"""
SentimentStream — Tests de la API Flask
Ejecutar: python test_api.py
Requiere que la API este corriendo en localhost:5000
"""
import requests
import sys
import json

BASE_URL = "http://localhost:5000"
ERRORES  = 0

def check(nombre, condicion, detalle=""):
    global ERRORES
    if condicion:
        print(f"  OK  {nombre}")
    else:
        print(f"  FAIL {nombre} {detalle}")
        ERRORES += 1

def test_health():
    print("\n[1] GET /health")
    r = requests.get(f"{BASE_URL}/health", timeout=5)
    check("Status code 200", r.status_code == 200)
    data = r.json()
    check("Campo 'status' presente", "status" in data)
    check("Campo 'timestamp' presente", "timestamp" in data)

def test_sentiments():
    print("\n[2] GET /sentiments")
    r = requests.get(f"{BASE_URL}/sentiments?limit=10", timeout=5)
    check("Status code 200", r.status_code == 200)
    data = r.json()
    check("Campo 'total' presente", "total" in data)
    check("Campo 'datos' presente", "datos" in data)
    check("datos es lista", isinstance(data.get("datos"), list))

def test_sentiments_filtro():
    print("\n[3] GET /sentiments?etiqueta=positivo")
    r = requests.get(f"{BASE_URL}/sentiments?etiqueta=positivo&limit=5", timeout=5)
    check("Status code 200", r.status_code == 200)

def test_stats():
    print("\n[4] GET /stats")
    r = requests.get(f"{BASE_URL}/stats", timeout=5)
    check("Status code 200", r.status_code == 200)
    data = r.json()
    check("Campo 'total_registros' presente", "total_registros" in data)
    check("Campo 'distribucion_real' presente", "distribucion_real" in data)

def test_predict_positivo():
    print("\n[5] POST /predict - texto positivo")
    r = requests.post(f"{BASE_URL}/predict",
                      json={"texto": "This product is absolutely amazing and wonderful"},
                      timeout=5)
    check("Status code 200", r.status_code == 200)
    data = r.json()
    check("Campo 'resultado' presente", "resultado" in data)
    check("Detecta sentimiento positivo",
          data.get("resultado", {}).get("sentimiento") == "positivo",
          f"obtuvo: {data.get('resultado', {}).get('sentimiento')}")

def test_predict_negativo():
    print("\n[6] POST /predict - texto negativo")
    r = requests.post(f"{BASE_URL}/predict",
                      json={"texto": "Terrible experience, very bad service, horrible"},
                      timeout=5)
    check("Status code 200", r.status_code == 200)
    data = r.json()
    check("Detecta sentimiento negativo",
          data.get("resultado", {}).get("sentimiento") == "negativo",
          f"obtuvo: {data.get('resultado', {}).get('sentimiento')}")

def test_predict_sin_texto():
    print("\n[7] POST /predict - sin campo texto (debe dar error)")
    r = requests.post(f"{BASE_URL}/predict", json={}, timeout=5)
    check("Status code 400", r.status_code == 400)
    check("Mensaje de error presente", "error" in r.json())

def test_wordcloud():
    print("\n[8] GET /wordcloud")
    r = requests.get(f"{BASE_URL}/wordcloud?etiqueta=positivo&top=10", timeout=5)
    check("Status code 200", r.status_code == 200)
    data = r.json()
    check("Campo 'palabras' presente", "palabras" in data)
    check("Campo 'etiqueta' correcto", data.get("etiqueta") == "positivo")

def test_wordcloud_invalido():
    print("\n[9] GET /wordcloud?etiqueta=invalido (debe dar error)")
    r = requests.get(f"{BASE_URL}/wordcloud?etiqueta=invalido", timeout=5)
    check("Status code 400", r.status_code == 400)

def test_dashboard():
    print("\n[10] GET /dashboard")
    r = requests.get(f"{BASE_URL}/dashboard", timeout=5)
    check("Status code 200", r.status_code == 200)
    check("Responde HTML", "text/html" in r.headers.get("Content-Type", ""))
    check("Contiene SentimentStream", "SentimentStream" in r.text)

if __name__ == "__main__":
    print("=" * 50)
    print("  SentimentStream — Tests de la API")
    print(f"  URL base: {BASE_URL}")
    print("=" * 50)

    try:
        requests.get(f"{BASE_URL}/health", timeout=3)
    except Exception:
        print(f"\nERROR: No se puede conectar a {BASE_URL}")
        print("Asegurate de que la API este corriendo: docker compose up -d api")
        sys.exit(1)

    test_health()
    test_sentiments()
    test_sentiments_filtro()
    test_stats()
    test_predict_positivo()
    test_predict_negativo()
    test_predict_sin_texto()
    test_wordcloud()
    test_wordcloud_invalido()
    test_dashboard()

    print("\n" + "=" * 50)
    total = 10
    pasaron = total - ERRORES
    print(f"  Resultado: {pasaron}/{total} tests pasaron")
    if ERRORES == 0:
        print("  Todos los tests pasaron correctamente")
    else:
        print(f"  {ERRORES} test(s) fallaron")
    print("=" * 50)
    sys.exit(1 if ERRORES > 0 else 0)
