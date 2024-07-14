# Databricks notebook source
import re 

from pyspark.sql import SparkSession
from pyspark.ml.feature import (
    RegexTokenizer, Tokenizer, StopWordsRemover, CountVectorizer, IDF, StringIndexer, HashingTF
)  
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator 
from pyspark.sql.functions import lit, regexp_replace, lower, explode, col 
from pyspark.sql.types import StructType, StructField, StringType, IntegerType  
from pyspark.ml import PipelineModel 
from pyspark.ml.tuning import CrossValidator, TrainValidationSplit 
from pyspark.ml.param.shared import HasSeed 
from pyspark.ml.util import _jvm  
from pyspark.sql.types import ArrayType, StringType  
from pyspark.sql.functions import udf 
from statistics import mode 
from pyspark.sql.types import DoubleType
import pyspark.sql.functions as sql_f

# Initialize SparkSession with master node set as local
spark = SparkSession.builder.master("local[*]").appName("MLlib lab").getOrCreate()

# Get Spark context from Spark session for low-level operations
sc = spark.sparkContext




# COMMAND ----------

#Read fake new data from the delta table
fake_news_dataset = spark.read.format("delta").table("news_data")
fake_news_dataset.show(5)

# COMMAND ----------

# Split the 'fake_news_dataset' into training and test sets with a ratio of 70% to 30% respectively
(training_fake_news_data, test_fake_news_data) = fake_news_dataset.randomSplit([0.7, 0.3], seed=100)

#Print the columns
columnnames = training_fake_news_data.columns

# Print or retrieve the column names
columnnames

# COMMAND ----------

value_counts = training_fake_news_data.groupBy('label').count().orderBy("count", ascending=False)
value_counts.collect()

# COMMAND ----------

global punctuation_chars
punctuation_chars = '!#$%&'

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


training_fake_news_data = preprocessing_fake_news_text_data(training_fake_news_data,punctuation_chars)
training_fake_news_data.show(1,truncate=False)

# COMMAND ----------

# Initialize the RegexTokenizer object with configuration to tokenize the 'clean_text' column.

regex_tokenizer = RegexTokenizer(inputCol="clean_text", outputCol="words", pattern="\\W")

# Apply the tokenizer to the 'training_fake_news_data' DataFrame to create a new column 'words' 

training_fake_news_data = regex_tokenizer.transform(training_fake_news_data)

# Select and display the first 5 rows of the DataFrame showing the 'label', 'clean_text', and 'words' columns.

training_fake_news_data.select('label', 'clean_text', 'words').show(5)

# tokenizer.save("/mnt/2024-team2/tokenizer")

# COMMAND ----------

#Instantiate StopwordRemover object that will remove the stopwords from the words
stopwords_remover = StopWordsRemover(inputCol="words", outputCol="filtered")

#Apply the tokenizer to the 'training_fake_news_data' DataFrame to create a new column 'words' 
training_fake_news_data = stopwords_remover.transform(training_fake_news_data)
training_fake_news_data.show(10)

# COMMAND ----------

#Define HashingTF transformer
hashing_tf = HashingTF(inputCol="filtered", 
                       outputCol="raw_features",  
                       numFeatures=3000) 

#Transform the input data using HashingTF
featurized_data = hashing_tf.transform(training_fake_news_data)

#Show the first 10 rows of the transformed DataFrame
featurized_data.show(10)


# COMMAND ----------

#Create IDF transformer object
idf = IDF(inputCol="raw_features",
          outputCol="features")

#Fit IDF transformer to the featurized data
idf_vectorizer = idf.fit(featurized_data)

#Transform the featurized data using the trained IDF model
rescaled_data = idf_vectorizer.transform(featurized_data)

#displayed  the first 10 rows of the transformed DataFrame
rescaled_data.show(10)


# COMMAND ----------

#Created the logistic regression object
lr_model=LogisticRegression()
lr_model

# COMMAND ----------

models=[]
def bagging(trainingData, weak_learner,bootstrap_size,max_iter=10): 
  '''Create the bagging function that will create multiple classifier'''
  for iteration in range(max_iter): 
    bag=trainingData.sample(withReplacement=True,fraction=1.0)
    lr_model.setPredictionCol(f"prediction_{iteration}")
    lr_model.setProbabilityCol(f"probability_{iteration}")
    lr_model.setRawPredictionCol(f"rawPrediction_{iteration}")
    models.append(lr_model.fit(trainingData))
max_iter=10
bagging(rescaled_data,lr_model,1,max_iter)

# COMMAND ----------

#Created the multi class evaluator object
evaluator=MulticlassClassificationEvaluator(labelCol='label',metricName='accuracy')

# COMMAND ----------

def test_models(test_features,models): 
  'Created the test function to test the bagging model on test data '
  for model in range(0,len(models)): 
    prediction=models[model].transform(test_features)
    evaluator.setPredictionCol(f'prediction_{model}')
    acc=evaluator.evaluate(prediction)
    print(f'accuracy {model}:{acc}')

test_fake_news_data=test_fake_news_data.select('text','label')    
test_fake_news_data = preprocessing_fake_news_text_data(test_fake_news_data,punctuation_chars)
# trainingData.show(1,truncate=False)
test_fake_news_data = regex_tokenizer.transform(test_fake_news_data)
# Remove stopwords
test_fake_news_data = stopwords_remover.transform(test_fake_news_data)
# Convert text to features
test_fake_news_data = hashing_tf.transform(test_fake_news_data)
# Apply IDF
test_fake_news_data = idf_vectorizer.transform(test_fake_news_data)
test_fake_news_data.show()

# COMMAND ----------

models
test_models(test_fake_news_data,models)


# COMMAND ----------

pipeline_models=PipelineModel(stages=models)
prediction=pipeline_models.transform(test_fake_news_data)

# COMMAND ----------


ensemble=prediction.select(sql_f.array([f"prediction_{i}" for i in range(max_iter)]).alias('preds'), 'label')
ensemble.show(1)

# COMMAND ----------


mode_udf=sql_f.udf(mode,DoubleType())
prediction=ensemble.withColumn('prediction',mode_udf('preds'))
prediction.show()

# COMMAND ----------

#Calculate evaluation metrics 
evaluator.setPredictionCol('prediction')
evaluator.evaluate(prediction)
accuracy = evaluator.evaluate(prediction, {evaluator.metricName: "accuracy"})
precision = evaluator.evaluate(prediction, {evaluator.metricName: "weightedPrecision"})
recall = evaluator.evaluate(prediction, {evaluator.metricName: "weightedRecall"})
f1_score = evaluator.evaluate(prediction, {evaluator.metricName: "f1"})

# Print the confusion metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1_score)

# COMMAND ----------

spark.stop()