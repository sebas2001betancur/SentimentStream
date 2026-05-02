Write-Host "Construyendo imagen de Spark..." -ForegroundColor Cyan
docker build -t sentimentstream-spark ./spark

Write-Host "Ejecutando job de Spark..." -ForegroundColor Cyan
docker run --rm `
  --network sentimentstream_default `
  -v "${PWD}/spark:/spark_scripts" `
  -v "${PWD}/data:/data" `
  sentimentstream-spark `
  /opt/spark/bin/spark-submit `
  --master "local[*]" `
  --conf "spark.jars.ivy=/tmp/.ivy2" `
  --packages "org.mongodb.spark:mongo-spark-connector_2.12:10.2.1" `
  /spark_scripts/sentiment_job.py

Write-Host "Listo! Abre http://localhost:5000/dashboard" -ForegroundColor Green