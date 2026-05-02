"""
SentimentStream — Job de Procesamiento de Sentimientos con PySpark
===================================================================
Pipeline NLP completo:
  1. Ingesta del CSV
  2. Limpieza y tokenización de texto
  3. Eliminación de stopwords
  4. Vectorización TF-IDF
  5. Entrenamiento de modelo Naive Bayes
  6. Inferencia y métricas de evaluación
  7. Persistencia de resultados en MongoDB
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    current_timestamp,
    monotonically_increasing_id,
    col,
    regexp_replace,
    lower,
    trim,
    when
)
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    Tokenizer,
    StopWordsRemover,
    HashingTF,
    IDF,
    StringIndexer,
    IndexToString
)
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import json

# ─────────────────────────────────────────────
# 1. INICIALIZACIÓN DE SPARK
# ─────────────────────────────────────────────
spark = SparkSession.builder \
    .appName("SentimentStream") \
    .master("local[*]") \
    .config("spark.mongodb.output.uri", "mongodb://mongodb:27017/sentimentdb.predictions") \
    .config("spark.driver.memory", "2g") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")
print("=" * 60)
print("✅ SparkSession iniciada correctamente")
print("=" * 60)

# ─────────────────────────────────────────────
# 2. INGESTA DE DATOS
# ─────────────────────────────────────────────
print("\n📥 Leyendo dataset...")
df_raw = spark.read.csv(
    "/data/dataset_sentimientos_500.csv",
    header=True,
    inferSchema=True
)
df_raw.printSchema()
print(f"   Total registros: {df_raw.count()}")
print(f"   Distribución de etiquetas:")
df_raw.groupBy("etiqueta").count().orderBy("etiqueta").show()

# ─────────────────────────────────────────────
# 3. LIMPIEZA DE TEXTO
# ─────────────────────────────────────────────
print("\n🧹 Limpiando texto...")
df_clean = df_raw \
    .withColumn("texto", trim(lower(col("texto")))) \
    .withColumn("texto", regexp_replace(col("texto"), r"[^a-zA-Z\s]", "")) \
    .withColumn("texto", regexp_replace(col("texto"), r"\s+", " ")) \
    .dropna(subset=["texto", "etiqueta"]) \
    .filter(col("texto") != "")

print(f"   Registros tras limpieza: {df_clean.count()}")

# ─────────────────────────────────────────────
# 4. DIVISIÓN TRAIN / TEST (80% / 20%)
# ─────────────────────────────────────────────
train_df, test_df = df_clean.randomSplit([0.8, 0.2], seed=42)
print(f"\n✂️  Train: {train_df.count()} | Test: {test_df.count()}")

# ─────────────────────────────────────────────
# 5. PIPELINE DE NLP + MODELO
# ─────────────────────────────────────────────
print("\n🔧 Construyendo pipeline NLP...")

# Etapa 1: Tokenización
tokenizer = Tokenizer(inputCol="texto", outputCol="words")

# Etapa 2: Eliminación de stopwords (inglés y español)
stopwords_es = [
    "de","la","el","en","y","a","que","se","es","un","una","con",
    "los","las","por","para","del","al","su","no","como","lo","pero",
    "le","más","si","ya","mi","me","te","ser","hay","fue","era","son"
]
remover = StopWordsRemover(
    inputCol="words",
    outputCol="filtered_words",
    stopWords=StopWordsRemover.loadDefaultStopWords("english") + stopwords_es
)

# Etapa 3: TF (Term Frequency)
hashingTF = HashingTF(
    inputCol="filtered_words",
    outputCol="rawFeatures",
    numFeatures=5000
)

# Etapa 4: IDF (Inverse Document Frequency)
idf = IDF(
    inputCol="rawFeatures",
    outputCol="features",
    minDocFreq=2
)

# Etapa 5: Encoding de etiquetas
indexer = StringIndexer(
    inputCol="etiqueta",
    outputCol="label",
    handleInvalid="keep"
)

# Etapa 6: Modelo Naive Bayes
nb = NaiveBayes(
    smoothing=1.0,
    modelType="multinomial",
    featuresCol="features",
    labelCol="label"
)

# Etapa 7: Decodificación de predicciones a etiquetas originales
label_converter = IndexToString(
    inputCol="prediction",
    outputCol="sentimiento_predicho",
    labels=[]  # Se rellena en fit()
)

pipeline = Pipeline(stages=[
    tokenizer,
    remover,
    hashingTF,
    idf,
    indexer,
    nb
])

# ─────────────────────────────────────────────
# 6. ENTRENAMIENTO
# ─────────────────────────────────────────────
print("\n🏋️  Entrenando modelo Naive Bayes...")
model = pipeline.fit(train_df)
print("   ✅ Modelo entrenado")

# ─────────────────────────────────────────────
# 7. EVALUACIÓN
# ─────────────────────────────────────────────
print("\n📊 Evaluando sobre conjunto de prueba...")
predictions_test = model.transform(test_df)

evaluator_acc = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="accuracy"
)
evaluator_f1 = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="f1"
)

accuracy = evaluator_acc.evaluate(predictions_test)
f1_score  = evaluator_f1.evaluate(predictions_test)

print(f"   Accuracy : {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"   F1-Score : {f1_score:.4f}")

# ─────────────────────────────────────────────
# 8. INFERENCIA SOBRE TODO EL DATASET
# ─────────────────────────────────────────────
print("\n🔮 Generando predicciones sobre todo el dataset...")
predictions_full = model.transform(df_clean)

# Recuperar mapa label → etiqueta del StringIndexerModel
indexer_model = model.stages[4]
labels = indexer_model.labels  # ['negativo', 'neutral', 'positivo'] (orden alfabético Spark)

# Mapear prediction numérica a nombre de etiqueta
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
    .select(
        "id",
        "texto",
        "etiqueta",
        "sentimiento_predicho",
        "correcto",
        "fecha_proceso"
    )

print(f"   Total predicciones generadas: {output_df.count()}")
output_df.show(10, truncate=50)

# ─────────────────────────────────────────────
# 9. GUARDAR EN MONGODB
# ─────────────────────────────────────────────
print("\n💾 Guardando resultados en MongoDB...")
try:
    output_df.write \
        .format("com.mongodb.spark.sql.DefaultSource") \
        .mode("overwrite") \
        .option("uri", "mongodb://mongodb:27017/sentimentdb.predictions") \
        .save()
    print("   ✅ Predicciones guardadas en sentimentdb.predictions")
except Exception as e:
    print(f"   ⚠️  Error escribiendo en MongoDB: {e}")
    print("   💡 Guardando resultados en CSV como respaldo...")
    output_df.coalesce(1).write.csv(
        "/data/predictions_output",
        header=True,
        mode="overwrite"
    )
    print("   ✅ Respaldo guardado en /data/predictions_output/")

# Guardar métricas de evaluación en MongoDB
print("\n📈 Guardando métricas en MongoDB...")
try:
    metrics_df = spark.createDataFrame([{
        "accuracy": float(accuracy),
        "f1_score": float(f1_score),
        "total_registros": int(df_clean.count()),
        "train_size": int(train_df.count()),
        "test_size": int(test_df.count()),
        "modelo": "NaiveBayes_multinomial",
        "fecha_ejecucion": str(spark.sql("SELECT current_timestamp()").collect()[0][0])
    }])
    metrics_df.write \
        .format("com.mongodb.spark.sql.DefaultSource") \
        .mode("overwrite") \
        .option("uri", "mongodb://mongodb:27017/sentimentdb.metrics") \
        .save()
    print("   ✅ Métricas guardadas en sentimentdb.metrics")
except Exception as e:
    print(f"   ⚠️  No se pudo guardar métricas: {e}")

# ─────────────────────────────────────────────
# 10. RESUMEN FINAL
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("🏁 RESUMEN DEL PROCESAMIENTO")
print("=" * 60)
print(f"  Dataset          : /data/dataset_sentimientos_500.csv")
print(f"  Total registros  : {df_clean.count()}")
print(f"  Modelo           : Naive Bayes Multinomial")
print(f"  Accuracy         : {accuracy*100:.2f}%")
print(f"  F1-Score         : {f1_score:.4f}")
print(f"  Salida MongoDB   : sentimentdb.predictions")
print(f"  Métricas MongoDB : sentimentdb.metrics")
print("=" * 60)
print("✅ Job completado exitosamente")

spark.stop()
