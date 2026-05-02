FROM apache/spark-py:latest

USER root
RUN pip install --no-cache-dir numpy pymongo

COPY sentiment_job.py /spark_scripts/sentiment_job.py
COPY dataset_sentimientos_500.csv /data/dataset_sentimientos_500.csv

USER 185