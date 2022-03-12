from fastapi import FastAPI
from starlette.responses import JSONResponse
import uvicorn
import torch
import torch.nn as nn
import torch.nn.functional as F
import os.path
import json
from joblib import load
import pandas as pd
from pytorchmulticlass import PytorchMultiClass

app = FastAPI()

# Get files saved in previous work
def read_files(path="./"):
    
    file = open(f"{path}brewery_name_dict.json", "r")
    brewery_name_dict = json.load(file)
    file.close()
    
    file = open(f"{path}beer_style_dict.json", "r")
    beer_style_dict = json.load(file)
    file.close()
    
    numeric_scaler = load(f'{path}numeric_scaler.joblib')
    
    brewery_name_encoder = load(f'{path}brewery_name_encoder.joblib')

    model = PytorchMultiClass(6)
    model.load_state_dict(torch.load(f'{path}pytorch_nn_v3_3_dict.pt'))
    model.eval()

    return brewery_name_dict, beer_style_dict, numeric_scaler, brewery_name_encoder, model

brewery_name_dict, beer_style_dict, numeric_scaler, brewery_name_encoder, model = read_files()

# Function that returns a list of single prediction inputs
def format_features(brewery_name: str, review_aroma: float, review_appearance: float, review_palate: float, review_taste: float, beer_abv: float):
    
    # check for brewery_name validity:
    if brewery_name not in brewery_name_dict:
        return False

    # Transform inputs to a dataframe
    obs = pd.DataFrame({
        'brewery_name': [brewery_name],
        'review_aroma': [review_aroma],
        'review_appearance': [review_appearance],
        'review_palate': [review_palate],
        'review_taste': [review_taste],
        'beer_abv': [beer_abv]
    })

    # Encode and scale input data
    obs["brewery_name"] = brewery_name_encoder.transform(obs["brewery_name"])
    num_cols = [i for i in obs.columns]
    obs[num_cols] = numeric_scaler.transform(obs[num_cols])

    # Transform dataframe to list
    obs_list = obs.values.tolist()
    return obs_list

# Function that returns a list of single prediction inputs
def multi_format_features(brewery_name: str, review_aroma: str, review_appearance: str, review_palate: str, review_taste: str, beer_abv: str):
    
    # Split up inputs into lists
    brewery_name_list = brewery_name.split(", ")
    review_aroma_list = review_aroma.split(", ")
    review_appearance_list = review_appearance.split(", ")
    review_palate_list = review_palate.split(", ")
    review_taste_list = review_taste.split(", ")
    beer_abv_list = beer_abv.split(", ")

    # Check for input length by comparing length
    length_list = [len(brewery_name_list), len(review_aroma_list), len(review_appearance_list), len(review_palate_list), len(review_taste_list), len(beer_abv_list)]
    if len(set(length_list)) != 1:
        return "Length Error"

    # Check for brewery_name validity:
    for i in brewery_name_list:
        if i not in brewery_name_dict:
            return False
    
    # Transform inputs to a dataframe
    obs = pd.DataFrame({
        'brewery_name': [i for i in brewery_name_list],
        'review_aroma': [float(i) for i in review_aroma_list],
        'review_appearance': [float(i) for i in review_appearance_list],
        'review_palate': [float(i) for i in review_palate_list],
        'review_taste': [float(i) for i in review_taste_list],
        'beer_abv': [float(i) for i in beer_abv_list]
    })

    # Encode and scale input data
    obs["brewery_name"] = brewery_name_encoder.transform(obs["brewery_name"])
    num_cols = [i for i in obs.columns]
    obs[num_cols] = numeric_scaler.transform(obs[num_cols])

    # Transform dataframe to list
    obs_list = obs.values.tolist()
    return obs_list

@app.get("/")
def home():
    return {"Project Objectives": "Build and deploy an neural networks model that can accurately predict a type of beer based on some rating criterias",
        "List of Endpoints": {"'/'(GET)": "Displaying a brief description of the project objectives, list of endpoints, expected input parameters and output format of the model, link to the Github repo related to this project", 
                            "'/health/' (GET)": "Returning status code 200 with a string with a welcome message", 
                            "'/brewer/'(GET)": "Returning all avaialable brewers", 
                            "'/beer/type/'(POST)": "Returning prediction for a single input only", 
                            "'/beers/type/'(POST)": "Returning predictions for a multiple inputs", 
                            "'/model/architecture/' (GET)": "Displaying the architecture of your Neural Networks (listing of all layers with their types)"
                            },
        "Expected input parameters": {"brewery_name": "Name of brewery", 
                                    "review_aroma": "Score given by reviewer regarding beer aroma", 
                                    "review_appearance": "Score given by reviewer regarding beer appearance", 
                                    "review_palate": "Score given by reviewer regarding beer palate", 
                                    "review_taste": "Score given by reviewer regarding beer taste", 
                                    "beer_abv": "Alcohol by volume measure"},
        "Output format": "String of beer type",
        "Link to Github": "To be updated",
        }

@app.get('/health', status_code=200)
def healthcheck():
    return 'Neural Network Model is ready to go!'

@app.get("/brewer")
def brewer_list():
    return brewery_name_dict

@app.post("/beer/type")
def predict(brewery_name: str,	review_aroma: float, review_appearance: float, review_palate: float, review_taste: float, beer_abv: float):
    
    obs_list = format_features(brewery_name, review_aroma, review_appearance, review_palate, review_taste, beer_abv)
    
    if not obs_list:
        return "Please check your brewery_name input, it is probably wrong. Refer '/brewer/'(GET) for avalable brewer names"
    else:
        _, y_pred_tags = torch.max(model(torch.FloatTensor(obs_list)), dim=1)

        # Reverse beer_style_dict to find out corresponding beer_type according to prediction
        beer_style_dict_reverse = {k:v for v,k in beer_style_dict.items()}
        beer_type = beer_style_dict_reverse[int(y_pred_tags)]
        return beer_type
    
        

@app.post("/beers/type")
def predicts_separate_using_a_comma_and_a_blank_space(brewery_name: str,	review_aroma: str, review_appearance: str, review_palate: str, review_taste: str, beer_abv: str):
    
    obs_list = multi_format_features(brewery_name, review_aroma, review_appearance, review_palate, review_taste, beer_abv)
    
    if obs_list == "Length Error":
        return "Please ensure all blanks are inputted with same number of inputs. And make sure each input is separated by a comma and a blank space : 'input_1, input_2'"
    elif not obs_list:
        return "Please check your brewery_name input, it is probably wrong. Refer '/brewer/'(GET) for avalable brewer names"
    else:
        _, y_pred_tags = torch.max(model(torch.FloatTensor(obs_list)), dim=1)

        # Reverse beer_style_dict to find out corresponding beer_type according to prediction
        beer_style_dict_reverse = {k:v for v,k in beer_style_dict.items()}
        output=[]
        for i in range(len(y_pred_tags)):
            beer_type = beer_style_dict_reverse[int(y_pred_tags[i])]
            output.append(beer_type)
        
        # format output
        output_dict = dict()
        for i in range(len(output)):
            output_dict[f'beer_type for input {i+1}'] = output[i]
        return output_dict

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)