"""
SentimentStream — API Flask
============================
Endpoints disponibles:
  GET  /health          → Estado de la API y conexión a MongoDB
  GET  /sentiments      → Listado paginado de predicciones
  GET  /stats           → Distribución de sentimientos y métricas del modelo
  GET  /sentiment/<id>  → Detalle de una predicción por ID
  POST /predict         → Inferencia simulada sobre texto nuevo
  GET  /dashboard       → Dashboard HTML interactivo con gráficas
"""

import os
import re
from flask import Flask, jsonify, request, Response
from flask_cors import CORS
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from datetime import datetime

# ─────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

MONGO_URI = os.environ.get("MONGO_URI", "mongodb://mongodb:27017/")
DB_NAME   = "sentimentdb"

def get_db():
    """Retorna la base de datos MongoDB. Lanza excepción si no hay conexión."""
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    client.admin.command("ping")  # Valida la conexión
    return client[DB_NAME]


# ─────────────────────────────────────────────
# UTILIDADES NLP BÁSICAS (para /predict)
# ─────────────────────────────────────────────
POSITIVE_WORDS = {
    "amazing", "great", "excellent", "good", "wonderful", "fantastic",
    "love", "best", "happy", "perfect", "awesome", "brilliant", "outstanding",
    "recommend", "highly", "superb", "nice", "pleased", "satisfied", "enjoy",
    "helpful", "friendly", "fast", "quick", "easy", "smooth", "reliable"
}
NEGATIVE_WORDS = {
    "terrible", "bad", "poor", "awful", "horrible", "worst", "hate",
    "disappointed", "useless", "broken", "slow", "rude", "unhelpful",
    "frustrating", "annoying", "problematic", "failure", "wrong", "error",
    "difficult", "confusing", "expensive", "waste", "never", "not"
}
NEUTRAL_WORDS = {
    "updated", "maintenance", "system", "file", "uploaded", "scheduled",
    "information", "available", "completed", "processed", "configured",
    "database", "server", "status", "report", "version", "release"
}

def predict_sentiment(text: str) -> dict:
    """Clasificador léxico básico para el endpoint /predict."""
    tokens = re.findall(r"[a-zA-Z]+", text.lower())
    pos = sum(1 for t in tokens if t in POSITIVE_WORDS)
    neg = sum(1 for t in tokens if t in NEGATIVE_WORDS)
    neu = sum(1 for t in tokens if t in NEUTRAL_WORDS)
    total = max(pos + neg + neu, 1)

    if pos > neg and pos >= neu:
        sentiment = "positivo"
        confidence = round(pos / total, 2)
    elif neg > pos and neg >= neu:
        sentiment = "negativo"
        confidence = round(neg / total, 2)
    else:
        sentiment = "neutral"
        confidence = round(max(neu, 1) / total, 2)

    return {
        "sentimiento": sentiment,
        "confianza": confidence,
        "scores": {"positivo": pos, "negativo": neg, "neutral": neu}
    }


# ─────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    """Estado de salud de la API."""
    try:
        db = get_db()
        total = db["predictions"].count_documents({})
        return jsonify({
            "status": "ok",
            "mongodb": "conectado",
            "total_predicciones": total,
            "timestamp": datetime.utcnow().isoformat()
        }), 200
    except (ConnectionFailure, ServerSelectionTimeoutError):
        return jsonify({
            "status": "degraded",
            "mongodb": "desconectado",
            "mensaje": "La API funciona pero MongoDB no está disponible aún.",
            "timestamp": datetime.utcnow().isoformat()
        }), 200  # 200 para no romper healthcheck de Docker


@app.route("/sentiments", methods=["GET"])
def get_sentiments():
    """
    Retorna predicciones almacenadas con paginación y filtrado opcional.
    Query params:
      limit     (int, default=50, max=200)
      page      (int, default=1)
      etiqueta  (str: positivo | negativo | neutral)
    """
    try:
        limit    = min(int(request.args.get("limit", 50)), 200)
        page     = max(int(request.args.get("page", 1)), 1)
        etiqueta = request.args.get("etiqueta", None)
        skip     = (page - 1) * limit

        db = get_db()
        query = {}
        if etiqueta and etiqueta in ("positivo", "negativo", "neutral"):
            query["etiqueta"] = etiqueta

        total   = db["predictions"].count_documents(query)
        records = list(
            db["predictions"]
            .find(query, {"_id": 0})
            .skip(skip)
            .limit(limit)
        )

        # Convertir campos no serializables
        for r in records:
            for k, v in r.items():
                if hasattr(v, "isoformat"):
                    r[k] = v.isoformat()

        return jsonify({
            "page":    page,
            "limit":   limit,
            "total":   total,
            "paginas": -(-total // limit),  # ceil division
            "datos":   records
        }), 200

    except (ConnectionFailure, ServerSelectionTimeoutError):
        return jsonify({"error": "MongoDB no disponible"}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/stats", methods=["GET"])
def get_stats():
    """
    Estadísticas generales: distribución de sentimientos, accuracy y métricas.
    """
    try:
        db = get_db()

        # Distribución de etiquetas reales
        pipeline_dist = [
            {"$group": {"_id": "$etiqueta", "total": {"$sum": 1}}},
            {"$sort": {"_id": 1}}
        ]
        distribucion = {
            r["_id"]: r["total"]
            for r in db["predictions"].aggregate(pipeline_dist)
        }

        # Distribución de predicciones
        pipeline_pred = [
            {"$group": {"_id": "$sentimiento_predicho", "total": {"$sum": 1}}},
            {"$sort": {"_id": 1}}
        ]
        predicciones = {
            r["_id"]: r["total"]
            for r in db["predictions"].aggregate(pipeline_pred)
        }

        # Accuracy real (campo "correcto" calculado en Spark)
        total     = db["predictions"].count_documents({})
        correctos = db["predictions"].count_documents({"correcto": True})
        accuracy  = round(correctos / total, 4) if total > 0 else 0

        # Métricas del modelo guardadas por Spark
        metricas = {}
        metric_doc = db["metrics"].find_one({}, {"_id": 0})
        if metric_doc:
            for k, v in metric_doc.items():
                if hasattr(v, "isoformat"):
                    metric_doc[k] = v.isoformat()
            metricas = metric_doc

        return jsonify({
            "total_registros":     total,
            "distribucion_real":   distribucion,
            "distribucion_predicha": predicciones,
            "accuracy":            accuracy,
            "metricas_modelo":     metricas
        }), 200

    except (ConnectionFailure, ServerSelectionTimeoutError):
        return jsonify({"error": "MongoDB no disponible"}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/sentiment/<int:sentiment_id>", methods=["GET"])
def get_sentiment_by_id(sentiment_id):
    """Retorna una predicción específica por su ID."""
    try:
        db  = get_db()
        doc = db["predictions"].find_one({"id": sentiment_id}, {"_id": 0})
        if not doc:
            return jsonify({"error": f"Registro con id={sentiment_id} no encontrado"}), 404
        for k, v in doc.items():
            if hasattr(v, "isoformat"):
                doc[k] = v.isoformat()
        return jsonify(doc), 200
    except (ConnectionFailure, ServerSelectionTimeoutError):
        return jsonify({"error": "MongoDB no disponible"}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict", methods=["POST"])
def predict():
    """
    Clasifica un texto nuevo con el clasificador léxico.
    Body JSON: { "texto": "Tu texto aquí" }
    """
    try:
        body  = request.get_json(force=True, silent=True) or {}
        texto = body.get("texto", "").strip()

        if not texto:
            return jsonify({"error": "El campo 'texto' es requerido y no puede estar vacío"}), 400
        if len(texto) > 2000:
            return jsonify({"error": "El texto no puede superar los 2000 caracteres"}), 400

        resultado = predict_sentiment(texto)
        return jsonify({
            "texto":       texto,
            "resultado":   resultado,
            "timestamp":   datetime.utcnow().isoformat(),
            "nota":        "Clasificación léxica en tiempo real (sin Spark)"
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/dashboard", methods=["GET"])
def dashboard():
    """Dashboard HTML interactivo que consulta la propia API."""
    html = """<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>SentimentStream — Dashboard</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: 'Segoe UI', sans-serif; background: #0f172a; color: #e2e8f0; }
    header { background: #1e293b; padding: 1.5rem 2rem; border-bottom: 2px solid #3b82f6; }
    header h1 { font-size: 1.6rem; color: #60a5fa; }
    header p  { color: #94a3b8; font-size: 0.9rem; margin-top: 0.25rem; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1.25rem; padding: 2rem; }
    .card { background: #1e293b; border-radius: 12px; padding: 1.5rem; border: 1px solid #334155; }
    .card h2 { font-size: 0.8rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px; }
    .card .value { font-size: 2rem; font-weight: 700; margin-top: 0.5rem; color: #f1f5f9; }
    .card .sub { font-size: 0.8rem; color: #64748b; margin-top: 0.25rem; }
    .charts { display: grid; grid-template-columns: 1fr 1fr; gap: 1.25rem; padding: 0 2rem 2rem; }
    .chart-card { background: #1e293b; border-radius: 12px; padding: 1.5rem; border: 1px solid #334155; }
    .chart-card h3 { margin-bottom: 1rem; font-size: 1rem; color: #cbd5e1; }
    .predict-box { margin: 0 2rem 2rem; background: #1e293b; border-radius: 12px; padding: 1.5rem; border: 1px solid #334155; }
    .predict-box h3 { color: #cbd5e1; margin-bottom: 1rem; }
    textarea { width: 100%; background: #0f172a; color: #e2e8f0; border: 1px solid #334155; border-radius: 8px; padding: 0.75rem; font-size: 1rem; resize: vertical; }
    button { margin-top: 0.75rem; background: #3b82f6; color: white; border: none; border-radius: 8px; padding: 0.75rem 2rem; font-size: 1rem; cursor: pointer; }
    button:hover { background: #2563eb; }
    #pred-result { margin-top: 1rem; padding: 1rem; background: #0f172a; border-radius: 8px; font-family: monospace; white-space: pre-wrap; color: #a3e635; }
    .badge { display: inline-block; padding: 0.2rem 0.6rem; border-radius: 999px; font-size: 0.75rem; font-weight: 700; }
    .pos { background: #14532d; color: #4ade80; }
    .neg { background: #7f1d1d; color: #f87171; }
    .neu { background: #1e3a5f; color: #60a5fa; }
    @media (max-width: 768px) { .charts { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <header>
    <h1>⚡ SentimentStream Dashboard</h1>
    <p>Análisis de sentimientos en tiempo real — MongoDB + PySpark + Flask</p>
  </header>

  <div class="grid" id="kpis">
    <div class="card"><h2>Total Registros</h2><div class="value" id="total">—</div><div class="sub">procesados por Spark</div></div>
    <div class="card"><h2>Accuracy</h2><div class="value" id="accuracy">—</div><div class="sub">sobre conjunto de prueba</div></div>
    <div class="card"><h2>Positivos</h2><div class="value" id="pos" style="color:#4ade80">—</div><div class="sub">registros</div></div>
    <div class="card"><h2>Negativos</h2><div class="value" id="neg" style="color:#f87171">—</div><div class="sub">registros</div></div>
    <div class="card"><h2>Neutrales</h2><div class="value" id="neu" style="color:#60a5fa">—</div><div class="sub">registros</div></div>
  </div>

  <div class="charts">
    <div class="chart-card"><h3>📊 Distribución Real de Etiquetas</h3><canvas id="chartReal"></canvas></div>
    <div class="chart-card"><h3>🤖 Distribución de Predicciones</h3><canvas id="chartPred"></canvas></div>
  </div>

  <div class="predict-box">
    <h3>🔮 Clasificar texto nuevo</h3>
    <textarea id="texto-input" rows="3" placeholder="Escribe un texto en inglés para clasificarlo..."></textarea>
    <button onclick="classify()">Clasificar</button>
    <div id="pred-result" style="display:none"></div>
  </div>

  <script>
    const COLORS = ['#4ade80','#f87171','#60a5fa'];
    let chartReal, chartPred;

    async function loadStats() {
      const r = await fetch('/stats');
      const d = await r.json();
      document.getElementById('total').textContent    = d.total_registros ?? '—';
      document.getElementById('accuracy').textContent = d.accuracy ? (d.accuracy*100).toFixed(1)+'%' : '—';
      document.getElementById('pos').textContent = d.distribucion_real?.positivo ?? '—';
      document.getElementById('neg').textContent = d.distribucion_real?.negativo ?? '—';
      document.getElementById('neu').textContent = d.distribucion_real?.neutral  ?? '—';

      const real = d.distribucion_real || {};
      const pred = d.distribucion_predicha || {};
      const labels = ['positivo','negativo','neutral'];

      if (chartReal) chartReal.destroy();
      if (chartPred) chartPred.destroy();

      chartReal = new Chart(document.getElementById('chartReal'), {
        type: 'doughnut',
        data: { labels, datasets: [{ data: labels.map(l=>real[l]||0), backgroundColor: COLORS, borderWidth: 0 }] },
        options: { plugins: { legend: { labels: { color: '#cbd5e1' } } } }
      });
      chartPred = new Chart(document.getElementById('chartPred'), {
        type: 'doughnut',
        data: { labels, datasets: [{ data: labels.map(l=>pred[l]||0), backgroundColor: COLORS, borderWidth: 0 }] },
        options: { plugins: { legend: { labels: { color: '#cbd5e1' } } } }
      });
    }

    async function classify() {
      const texto = document.getElementById('texto-input').value.trim();
      if (!texto) return;
      const box = document.getElementById('pred-result');
      box.style.display = 'block';
      box.textContent = 'Clasificando...';
      const r = await fetch('/predict', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({texto})
      });
      const d = await r.json();
      box.textContent = JSON.stringify(d, null, 2);
    }

    loadStats();
    setInterval(loadStats, 30000);
  </script>
</body>
</html>"""
    return Response(html, mimetype="text/html")



@app.route("/wordcloud", methods=["GET"])
def wordcloud():
    """
    Palabras mas frecuentes por sentimiento.
    Query params:
      etiqueta  (str: positivo | negativo | neutral, default=positivo)
      top       (int, default=20)
    """
    try:
        etiqueta = request.args.get("etiqueta", "positivo")
        top_n    = min(int(request.args.get("top", 20)), 50)

        if etiqueta not in ("positivo", "negativo", "neutral"):
            return jsonify({"error": "etiqueta debe ser positivo, negativo o neutral"}), 400

        db   = get_db()
        docs = list(db["predictions"].find(
            {"etiqueta": etiqueta}, {"texto": 1, "_id": 0}
        ))

        if not docs:
            return jsonify({"etiqueta": etiqueta, "palabras": [], "total_docs": 0}), 200

        stopwords = {
            "the","is","a","an","it","this","was","for","to","of","and","in",
            "i","my","with","that","not","be","me","on","at","as","by","are",
            "were","has","had","have","been","will","would","could","should",
            "its","our","their","your","we","they","he","she","do","did",
            "but","so","if","or","no","up","out","about","what","when","how"
        }

        from collections import Counter
        todas = []
        for d in docs:
            palabras = re.findall(r"[a-zA-Z]+", d.get("texto", "").lower())
            todas.extend([p for p in palabras if p not in stopwords and len(p) > 2])

        conteo = Counter(todas).most_common(top_n)
        return jsonify({
            "etiqueta":    etiqueta,
            "total_docs":  len(docs),
            "top_n":       top_n,
            "palabras":    [{"palabra": w, "frecuencia": f} for w, f in conteo]
        }), 200

    except (ConnectionFailure, ServerSelectionTimeoutError):
        return jsonify({"error": "MongoDB no disponible"}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────
# ARRANQUE
# ─────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
