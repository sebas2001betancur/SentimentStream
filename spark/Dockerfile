# Imagen base oficial de Spark con soporte Python
FROM bitnami/spark:3.5

USER root

# Instalar pymongo para que el job pueda escribir en MongoDB
RUN pip install pymongo==4.7.3 --no-cache-dir

# Directorio de trabajo
WORKDIR /app

# Copiar el job de PySpark
COPY sentiment_job.py .

# El conector de MongoDB para Spark se descarga en tiempo de ejecucion
# via --packages en el spark-submit
CMD [ \
  "spark-submit", \
  "--master", "local[*]", \
  "--packages", "org.mongodb.spark:mongo-spark-connector_2.12:10.2.1", \
  "--conf", "spark.mongodb.output.uri=mongodb://mongodb:27017/sentimentdb.predictions", \
  "--conf", "spark.driver.memory=2g", \
  "/app/sentiment_job.py" \
]
