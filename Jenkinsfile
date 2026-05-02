pipeline {
    agent any

    stages {

        stage('Clonar repositorio') {
            steps {
                git branch: 'master', url: 'https://github.com/sebas2001betancur/SentimentStream.git'
            }
        }

        stage('Build imagenes Docker') {
            steps {
                sh 'docker compose build --no-cache'
            }
        }

        stage('Levantar infraestructura') {
            steps {
                sh 'docker compose up -d'
                sh 'sleep 10'
            }
        }

        stage('Ejecutar Spark NLP') {
            steps {
                sh '''
                    cp data/dataset_sentimientos_500.csv spark/
                    docker build -t sentimentstream-spark ./spark
                    docker run --rm \
                      --network sentimentstream_default \
                      sentimentstream-spark \
                      /opt/spark/bin/spark-submit \
                      --master local[*] \
                      --conf spark.jars.ivy=/tmp/.ivy2 \
                      --packages org.mongodb.spark:mongo-spark-connector_2.12:10.2.1 \
                      /spark_scripts/sentiment_job.py
                '''
            }
        }

        stage('Verificar API') {
            steps {
                sh 'curl -sf http://flask_api:5000/health && echo "✅ API OK"'
            }
        }

    } // <--- ESTA ES LA LLAVE QUE FALTABA PARA CERRAR "stages"

    post {
        success { 
            echo '✅ API disponible en http://localhost:5000/dashboard' 
        }
        failure { 
            echo '❌ Pipeline falló' 
        }
    }
}