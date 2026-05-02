pipeline {
    agent any

    environment {
        COMPOSE_FILE = 'docker-compose.yml'
    }

    stages {

        stage('Checkout') {
            steps {
                echo 'Clonando repositorio...'
                checkout scm
            }
        }

        stage('Build imagenes Docker') {
            steps {
                echo 'Construyendo imagenes de Spark y Flask...'
                // CORREGIDO: "docker compose" con espacio (v2) en lugar de "docker-compose" (v1)
                sh 'docker compose build --no-cache'
            }
        }

        stage('Levantar infraestructura') {
            steps {
                echo 'Iniciando MongoDB y API Flask...'
                sh 'docker compose up -d mongodb api'
                echo 'Esperando que MongoDB este listo (30s)...'
                sh '''
                    for i in $(seq 1 12); do
                        if docker exec mongodb mongosh --eval "db.adminCommand('ping')" > /dev/null 2>&1; then
                            echo "MongoDB listo en intento $i"
                            break
                        fi
                        echo "Intento $i/12 - esperando MongoDB..."
                        sleep 5
                    done
                '''
            }
        }

        stage('Ejecutar pipeline Spark NLP') {
            steps {
                echo 'Corriendo job de PySpark para analisis de sentimientos...'
                // CORREGIDO: "docker compose run" en lugar de "docker exec" sobre el master
                // El contenedor spark corre el job y termina automaticamente
                sh 'docker compose run --rm spark'
            }
        }

        stage('Verificar API REST') {
            steps {
                echo 'Verificando que la API responde correctamente...'
                sh 'sleep 5'
                sh 'curl -sf http://localhost:5000/health'
                sh 'curl -sf http://localhost:5000/stats'
                sh '''
                    curl -sf -X POST http://localhost:5000/predict \
                        -H "Content-Type: application/json" \
                        -d "{\"texto\": \"This product is absolutely amazing\"}"
                '''
                echo 'Todos los endpoints responden correctamente'
            }
        }

        stage('Guardar artefactos') {
            steps {
                echo 'Archivando outputs del job Spark...'
                sh 'mkdir -p data/outputs'
                archiveArtifacts artifacts: 'data/outputs/**', fingerprint: true, allowEmptyArchive: true
                echo 'Artefactos guardados'
            }
        }
    }

    post {
        success {
            echo 'Pipeline SentimentStream completado. API en http://localhost:5000/dashboard'
        }
        failure {
            echo 'Pipeline fallo. Deteniendo contenedores...'
            sh 'docker compose down || true'
        }
        always {
            echo 'Pipeline finalizado'
        }
    }
}
