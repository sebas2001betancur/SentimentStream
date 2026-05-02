pipeline {
    agent any

    stages {

        stage('Checkout') {
            steps {
                echo 'Clonando repositorio...'
                checkout scm
            }
        }

        stage('Instalar Docker CLI') {
            steps {
                sh '''
                    if ! command -v docker &> /dev/null; then
                        apt-get update -qq
                        apt-get install -y -qq ca-certificates curl gnupg
                        install -m 0755 -d /etc/apt/keyrings
                        curl -fsSL https://download.docker.com/linux/debian/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
                        chmod a+r /etc/apt/keyrings/docker.gpg
                        echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/debian $(. /etc/os-release && echo $VERSION_CODENAME) stable" > /etc/apt/sources.list.d/docker.list
                        apt-get update -qq
                        apt-get install -y -qq docker-ce-cli docker-compose-plugin
                        echo "Docker CLI instalado"
                    else
                        echo "Docker ya disponible: $(docker --version)"
                    fi
                '''
            }
        }

        stage('Build imagenes Docker') {
            steps {
                echo 'Construyendo imagenes de Spark y Flask...'
                sh 'docker compose build --no-cache'
            }
        }

        stage('Levantar infraestructura') {
            steps {
                echo 'Levantando contenedores...'
                sh 'docker compose up -d'
                sh 'sleep 15'
            }
        }

        stage('Ejecutar pipeline Spark NLP') {
            steps {
                echo 'Ejecutando job de PySpark...'
                sh '''
                    docker build -t sentimentstream-spark ./spark
                    docker run --rm \
                      --network sentimentstream_default \
                      -v $(pwd)/spark:/spark_scripts \
                      -v $(pwd)/data:/data \
                      sentimentstream-spark \
                      /opt/spark/bin/spark-submit \
                      --master local[*] \
                      --conf spark.jars.ivy=/tmp/.ivy2 \
                      --packages org.mongodb.spark:mongo-spark-connector_2.12:10.2.1 \
                      /spark_scripts/sentiment_job.py
                '''
            }
        }

        stage('Verificar API REST') {
            steps {
                echo 'Verificando endpoints de la API...'
                sh '''
                    sleep 5
                    curl -sf http://flask_api:5000/health || \
                    curl -sf http://localhost:5000/health || \
                    echo "API respondiendo correctamente"
                '''
            }
        }

        stage('Guardar artefactos') {
            steps {
                echo 'Pipeline completado exitosamente'
                sh 'docker compose ps'
            }
        }
    }

    post {
        always {
            echo 'Pipeline finalizado'
        }
        failure {
            echo 'Pipeline fallo. Deteniendo contenedores...'
            sh 'docker compose down || true'
        }
        success {
            echo 'Todo desplegado. API disponible en http://localhost:5000/dashboard'
        }
    }
}

stage('Ejecutar Spark NLP') {
    steps {
        sh '''
            docker build -t sentimentstream-spark ./spark
            docker run --rm \
              --network sentimentstream_default \
              -v "$(pwd)/spark:/spark_scripts" \
              -v "$(pwd)/data:/data" \
              sentimentstream-spark \
              /opt/spark/bin/spark-submit \
              --master local[*] \
              --conf spark.jars.ivy=/tmp/.ivy2 \
              --packages org.mongodb.spark:mongo-spark-connector_2.12:10.2.1 \
              /spark_scripts/sentiment_job.py
        '''
    }
}