# Databricks notebook source
import re 

from pyspark.sql import SparkSession
from pyspark.ml.feature import RegexTokenizer,Tokenizer,StopWordsRemover,HashingTF,IDFModel
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import lit,regexp_replace,lower,explode,col
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import HashingTF, IDFModel, RegexTokenizer, StopWordsRemover
from pyspark.ml.classification import LogisticRegressionModel
from pyspark.sql import SparkSession
from pyspark.sql.functions import lower, regexp_replace
from pyspark.ml.classification import LogisticRegression, LogisticRegressionModel
import re

# spark = SparkSession.builder.master("local[*]").appName("MLlib lab").getOrCreate()
# sc = spark.sparkContext


fake_news_test_dataset = spark.read.csv('/mnt/2024-team2/dataset/news_data_test.csv', inferSchema=False, header=True)

fake_news_test_dataset=fake_news_test_dataset.select('text')
fake_news_test_dataset.show(5)


# COMMAND ----------

punctuation_chars = '!#$%&'
# punctuation_chars_braodcast = sc.broadcast(punctuation_chars)


def preprocessing_fake_news_text_data(dataset, punctuation_chars):
    # Convert text to lowercase
    dataset = dataset.withColumn('clean_text', lower(dataset['text']))
    
    # Remove URLs
    dataset = dataset.withColumn("clean_text", regexp_replace(dataset["clean_text"], r"http[s]?\://\S+", "")) 
    
    # Remove text within parentheses or square brackets
    dataset = dataset.withColumn("clean_text", regexp_replace(dataset["clean_text"], r"(\(.*\))|(\[.*\])", ""))
    
    # Remove words containing consecutive asterisks
    dataset = dataset.withColumn("clean_text", regexp_replace(dataset["clean_text"], r"\b\w+\*{2,3}\w*\b", ""))
    
    # Remove special characters and punctuation
    dataset = dataset.withColumn("clean_text", regexp_replace(dataset["clean_text"], r'[!#$%&\'()*+,-./:;<=>?@[\\]^_`{|}]+', ""))
    dataset = dataset.withColumn("clean_text", regexp_replace(dataset["clean_text"], r"[" + re.escape(punctuation_chars) + "]", ""))
    
    return dataset
fake_news_test_dataset = fake_news_test_dataset.drop("label")
fake_news_test_dataset = preprocessing_fake_news_text_data(fake_news_test_dataset,punctuation_chars)
fake_news_test_dataset.show(1,truncate=False)
# print(final_dataset.getNumPartitions())

# COMMAND ----------

#Load Our Transformer & Extractor Pkgs
from pyspark.ml.feature import Tokenizer,StopWordsRemover,CountVectorizer,IDF
from pyspark.ml.feature import StringIndexer

# COMMAND ----------


regextokenizer=RegexTokenizer.load("/mnt/2024-team2/tokenizer")
fake_news_test_dataset=regextokenizer.transform(fake_news_test_dataset)

# COMMAND ----------

stopwords_remover=StopWordsRemover.load("/mnt/2024-team2/stopwords_remover")
fake_news_test_dataset = stopwords_remover.transform(fake_news_test_dataset)

# COMMAND ----------

hashing_tf=HashingTF.load("/mnt/2024-team2/hashing_tf")
featurized_data = hashing_tf.transform(fake_news_test_dataset)

# COMMAND ----------



idf_vectorizer=IDFModel.load("/mnt/2024-team2/idf_model_path")
rescaled_data = idf_vectorizer.transform(featurized_data)

# COMMAND ----------



model = LogisticRegressionModel.load('/mnt/2024-team2/lr-model')
predict_saved = model.transform(rescaled_data)
predict_saved.select('prediction').show()