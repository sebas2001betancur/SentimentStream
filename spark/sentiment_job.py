"""
SentimentStream — Job de Procesamiento de Sentimientos con PySpark
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    current_timestamp, monotonically_increasing_id,
    col, regexp_replace, lower, trim, when
)
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    Tokenizer, StopWordsRemover, HashingTF, IDF, StringIndexer
)
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

spark = SparkSession.builder \
    .appName("SentimentStream") \
    .master("local[*]") \
    .config("spark.driver.memory", "2g") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")
print("=" * 60)
print("✅ SparkSession iniciada correctamente")
print("=" * 60)

print("\n📥 Leyendo dataset...")
df_raw = spark.read.csv("/data/dataset_sentimientos_500.csv", header=True, inferSchema=True)
df_raw.printSchema()
print(f"   Total registros: {df_raw.count()}")
df_raw.groupBy("etiqueta").count().orderBy("etiqueta").show()

print("\n🧹 Limpiando texto...")
df_clean = df_raw \
    .withColumn("texto", trim(lower(col("texto")))) \
    .withColumn("texto", regexp_replace(col("texto"), r"[^a-zA-Z\s]", "")) \
    .withColumn("texto", regexp_replace(col("texto"), r"\s+", " ")) \
    .dropna(subset=["texto", "etiqueta"]) \
    .filter(col("texto") != "")
print(f"   Registros tras limpieza: {df_clean.count()}")

train_df, test_df = df_clean.randomSplit([0.8, 0.2], seed=42)
print(f"\n✂️  Train: {train_df.count()} | Test: {test_df.count()}")

print("\n🔧 Construyendo pipeline NLP...")
stopwords_es = [
    "de","la","el","en","y","a","que","se","es","un","una","con",
    "los","las","por","para","del","al","su","no","como","lo","pero",
    "le","más","si","ya","mi","me","te","ser","hay","fue","era","son"
]
tokenizer = Tokenizer(inputCol="texto", outputCol="words")
remover = StopWordsRemover(
    inputCol="words", outputCol="filtered_words",
    stopWords=StopWordsRemover.loadDefaultStopWords("english") + stopwords_es
)
hashingTF = HashingTF(inputCol="filtered_words", outputCol="rawFeatures", numFeatures=5000)
idf = IDF(inputCol="rawFeatures", outputCol="features", minDocFreq=2)
indexer = StringIndexer(inputCol="etiqueta", outputCol="label", handleInvalid="keep")
nb = NaiveBayes(smoothing=1.0, modelType="multinomial", featuresCol="features", labelCol="label")
pipeline = Pipeline(stages=[tokenizer, remover, hashingTF, idf, indexer, nb])

print("\n🏋️  Entrenando modelo Naive Bayes...")
model = pipeline.fit(train_df)
print("   ✅ Modelo entrenado")

print("\n📊 Evaluando sobre conjunto de prueba...")
predictions_test = model.transform(test_df)
evaluator_acc = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
evaluator_f1  = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
accuracy = evaluator_acc.evaluate(predictions_test)
f1_score  = evaluator_f1.evaluate(predictions_test)
print(f"   Accuracy : {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"   F1-Score : {f1_score:.4f}")

print("\n🔮 Generando predicciones sobre todo el dataset...")
predictions_full = model.transform(df_clean)
indexer_model = model.stages[4]
labels = indexer_model.labels

def map_label(pred_col):
    expr = None
    for i, lbl in enumerate(labels):
        cond = col(pred_col) == float(i)
        expr = when(cond, lbl) if expr is None else expr.when(cond, lbl)
    return expr.otherwise("desconocido")

output_df = predictions_full \
    .withColumn("id", monotonically_increasing_id()) \
    .withColumn("fecha_proceso", current_timestamp()) \
    .withColumn("sentimiento_predicho", map_label("prediction")) \
    .withColumn("correcto", (col("etiqueta") == col("sentimiento_predicho")).cast("boolean")) \
    .select("id", "texto", "etiqueta", "sentimiento_predicho", "correcto", "fecha_proceso")

print(f"   Total predicciones generadas: {output_df.count()}")
output_df.show(10, truncate=50)

# ─────────────────────────────────────────────
# GUARDAR EN MONGODB con pymongo directamente
# ─────────────────────────────────────────────
print("\n💾 Guardando resultados en MongoDB...")
try:
    from pymongo import MongoClient
    client = MongoClient("mongodb://mongodb:27017/")
    db = client["sentimentdb"]

    rows = output_df.collect()
    docs = [{
        "id": int(row["id"]),
        "texto": row["texto"],
        "etiqueta": row["etiqueta"],
        "sentimiento_predicho": row["sentimiento_predicho"],
        "correcto": bool(row["correcto"]),
        "fecha_proceso": str(row["fecha_proceso"])
    } for row in rows]

    db["predictions"].drop()
    db["predictions"].insert_many(docs)
    print(f"   ✅ {len(docs)} predicciones guardadas en sentimentdb.predictions")

    db["metrics"].drop()
    db["metrics"].insert_one({
        "accuracy": float(accuracy),
        "f1_score": float(f1_score),
        "total_registros": int(df_clean.count()),
        "train_size": int(train_df.count()),
        "test_size": int(test_df.count()),
        "modelo": "NaiveBayes_multinomial"
    })
    print("   ✅ Métricas guardadas en sentimentdb.metrics")
    client.close()

except Exception as e:
    print(f"   ⚠️  Error guardando en MongoDB: {e}")

print("\n" + "=" * 60)
print("🏁 RESUMEN DEL PROCESAMIENTO")
print("=" * 60)
print(f"  Total registros  : {df_clean.count()}")
print(f"  Modelo           : Naive Bayes Multinomial")
print(f"  Accuracy         : {accuracy*100:.2f}%")
print(f"  F1-Score         : {f1_score:.4f}")
print(f"  Salida MongoDB   : sentimentdb.predictions")
print("=" * 60)
print("✅ Job completado exitosamente")
spark.stop()