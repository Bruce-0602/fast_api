FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7

RUN pip install torch torchvision torchaudio

COPY requirements.txt .

RUN pip3 install -r requirements.txt

COPY ./app /app

COPY ./models /models

CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-c", "/gunicorn_conf.py", "main:app"]
