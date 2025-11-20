FROM python:3.11

WORKDIR /app

RUN mkdir -p /app/data
RUN mkdir -p /app/deployment
RUN mkdir -p /app/ml_pipeline
RUN mkdir -p /app/models
RUN mkdir -p /app/src

COPY deployment/run_server.py /app/deployment
COPY requirements.txt /app
COPY deployment/src /app/src
COPY models /app/models
COPY ml_pipeline /app/ml_pipeline

RUN pip install -r requirements.txt

EXPOSE 8082

CMD ["python", "-m", "ml_pipeline.serve.app"]
