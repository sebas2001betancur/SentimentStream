# ⚡ SentimentStream

Proyecto de **Big Data** para análisis de sentimientos en texto, utilizando **Apache Spark**, **MongoDB**, **Flask** y **Jenkins**.

---

## 📁 Estructura del proyecto

```
SentimentStream/
├── docker-compose.yml          ← Orquestación de todos los servicios
├── Jenkinsfile                 ← Pipeline CI/CD
├── README.md                   ← Este archivo
├── data/
│   └── dataset_sentimientos_500.csv   ← Dataset de entrada
├── spark/
│   └── sentiment_job.py        ← Job PySpark (NLP + Naive Bayes)
└── api/
    ├── Dockerfile
    ├── requirements.txt
    └── app.py                  ← API Flask con 5 endpoints + dashboard
```

---

## 🏗️ Arquitectura

```
CSV Dataset
    │
    ▼
[Apache Spark] ─── Pipeline NLP ──────────────────────────────────────────┐
  • Tokenización                                                            │
  • Stopwords removal                                                       │
  • TF-IDF                                                                  │
  • Naive Bayes                                                             │
    │                                                                       │
    ▼                                                                       ▼
[MongoDB] ◄────────────────────────── predictions / metrics ──────────────┘
    │
    ▼
[Flask API] ─── /sentiments | /stats | /predict | /dashboard | /health
    │
    ▼
[Jenkins] ─── CI/CD Pipeline (Build → Deploy → Spark → Verify)
```

---

## 🚀 Instrucciones de ejecución

### Prerrequisitos

- Docker Desktop instalado y corriendo
- Docker Compose v2+
- Puertos libres: `5000`, `8080`, `8081`, `27017`

### Paso 1 — Levantar la infraestructura

```bash
cd SentimentStream
docker-compose up -d --build
```

Espera ~30 segundos a que MongoDB esté listo.

### Paso 2 — Ejecutar el job de Spark

```bash
docker exec -it spark_master spark-submit \
  --master local[*] \
  --packages org.mongodb.spark:mongo-spark-connector_2.12:10.2.1 \
  /spark_scripts/sentiment_job.py
```

> El job tarda 1-3 minutos. Verás en consola el Accuracy y F1-Score al finalizar.

### Paso 3 — Verificar la API

Abre en el navegador:

| URL | Descripción |
|-----|-------------|
| `http://localhost:5000/health`      | Estado de la API |
| `http://localhost:5000/stats`       | Métricas y distribución |
| `http://localhost:5000/sentiments`  | Listado de predicciones |
| `http://localhost:5000/dashboard`   | Dashboard interactivo |

### Paso 4 — Jenkins (CI/CD)

1. Abre `http://localhost:8080`
2. Obtén la contraseña inicial:
   ```bash
   docker exec jenkins cat /var/jenkins_home/secrets/initialAdminPassword
   ```
3. Crea un pipeline nuevo apuntando al `Jenkinsfile` del repositorio.
4. Ejecuta el pipeline — automatiza todos los pasos anteriores.

---

## 📡 Endpoints de la API

### `GET /health`
```json
{
  "status": "ok",
  "mongodb": "conectado",
  "total_predicciones": 502,
  "timestamp": "2024-01-15T10:30:00"
}
```

### `GET /stats`
```json
{
  "total_registros": 502,
  "distribucion_real": { "positivo": 168, "negativo": 167, "neutral": 167 },
  "distribucion_predicha": { "positivo": 170, "negativo": 165, "neutral": 167 },
  "accuracy": 0.8761,
  "metricas_modelo": { "accuracy": 0.87, "f1_score": 0.87, "modelo": "NaiveBayes_multinomial" }
}
```

### `GET /sentiments?limit=10&page=1&etiqueta=positivo`
```json
{
  "page": 1,
  "limit": 10,
  "total": 168,
  "paginas": 17,
  "datos": [ { "id": 1, "texto": "Amazing experience...", "etiqueta": "positivo", ... } ]
}
```

### `POST /predict`
**Body:** `{ "texto": "This product is absolutely amazing!" }`
```json
{
  "texto": "This product is absolutely amazing!",
  "resultado": {
    "sentimiento": "positivo",
    "confianza": 0.75,
    "scores": { "positivo": 3, "negativo": 0, "neutral": 1 }
  },
  "timestamp": "2024-01-15T10:30:00"
}
```

### `GET /dashboard`
Retorna el dashboard HTML interactivo con gráficas y clasificador en vivo.

---

## 🛠️ Tecnologías utilizadas

| Componente | Tecnología | Versión |
|------------|-----------|---------|
| Procesamiento Big Data | Apache Spark / PySpark | 3.4.1 |
| Modelo ML | Naive Bayes Multinomial | MLlib |
| Base de datos | MongoDB | 6.0 |
| API REST | Flask + Flask-CORS | 3.0 |
| CI/CD | Jenkins | LTS |
| Orquestación | Docker Compose | v3.8 |

---

## 🔧 Comandos útiles

```bash
# Ver logs de la API
docker logs flask_api -f

# Ver logs de Spark
docker logs spark_master -f

# Conectarse a MongoDB directamente
docker exec -it mongodb mongosh sentimentdb

# Ver predicciones en Mongo
docker exec -it mongodb mongosh sentimentdb --eval "db.predictions.find().limit(5).pretty()"

# Apagar todo
docker-compose down

# Apagar y borrar datos
docker-compose down -v
```
