from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer
from pyspark.ml.feature import CountVectorizer
from pyspark.ml.feature import IDF
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

def SentimentAnalysis(text):
    spark = SparkSession.builder.appName("SentimentAnalysis").getOrCreate()
    
    # Veri setini Spark DataFrame'e yükle
    dataset = spark.read.csv('hepsiburada.csv', header=True, inferSchema=True)
    
    # Veri setini inceleyelim
    dataset.show(5)
    
    # Gerekli sütunları seç
    dataset = dataset.select('Review', 'Rating')
    
    # Train-test bölünmesi
    (train_data, test_data) = dataset.randomSplit([0.8, 0.2], seed=123)
    
    # Tokenizasyon ve TF-IDF dönüşümü için bir pipeline oluştur
    tokenizer = Tokenizer(inputCol="Review", outputCol="words")
    cv = CountVectorizer(inputCol="words", outputCol="rawFeatures")
    idf = IDF(inputCol="rawFeatures", outputCol="features")
    
    # Label indexing
    indexer = StringIndexer(inputCol="Rating", outputCol="label")
    
    # Sınıflandırıcı modeli oluştur
    gbt = GBTClassifier(labelCol="label", featuresCol="features", maxIter=10)
    
    # Pipeline'ı oluştur
    pipeline = Pipeline(stages=[tokenizer, cv, idf, indexer, gbt])
    
    # Modeli eğit
    model = pipeline.fit(train_data)
    
    # Modeli değerlendir
    predictions = model.transform(test_data)
    
    # Tahmin yapmak için gelen metni işle
    text_df = spark.createDataFrame([(text,)], ["Review"])
    text_tokenized = tokenizer.transform(text_df)
    text_cv = cv.transform(text_tokenized)
    text_idf = idf.transform(text_cv)
    
    # Tahmini yap
    result = model.transform(text_idf)
    
    # Tahmin sonucunu değerlendir
    predicted_label = result.select("prediction").collect()[0][0]
    
    if predicted_label == 1.0:
        yorum_tipi = "olumlu"
    else:
        yorum_tipi = "olumsuz"
    
    spark.stop()
    return yorum_tipi

# Kullanım örneği
text = "güzel"
print(SentimentAnalysis(text))
