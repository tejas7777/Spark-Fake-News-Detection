# Databricks notebook source
import re 
#Import pyspark libraries
from pyspark.sql import SparkSession
from pyspark.ml.feature import RegexTokenizer, Tokenizer, StopWordsRemover, CountVectorizer, IDF, StringIndexer, HashingTF
from pyspark.ml.classification import LogisticRegression, LogisticRegressionModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator,Evaluator
from pyspark.sql.functions import lit, regexp_replace, lower, explode, col
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from pyspark.ml.tuning import CrossValidator, TrainValidationSplit
from pyspark.ml.param.shared import HasSeed
from pyspark.ml.util import _jvm
from pyspark.sql.types import ArrayType, StringType
from pyspark.sql.functions import udf

from pyspark.sql.functions import expr
import matplotlib.pyplot as plt
import seaborn as sns

import pyspark.ml.feature



# COMMAND ----------


#Read fake new data from the delta table
fake_news_dataset = spark.read.format("delta").table("news_data")

#Display top 5 row of dataframe
schema = StructType([
    StructField('title', StringType(), True),
    StructField('text', StringType(), True),
    StructField('date', StringType(), True)
])
fake_news_dataset=fake_news_dataset.select('text','label')


# COMMAND ----------

# Split the 'fake_news_dataset' into training and test sets with a ratio of 70% to 30% respectively

(training_fake_news_data, test_fake_news_data) = fake_news_dataset.randomSplit([0.7, 0.3], seed=100)

#Print the columns
columnnames = training_fake_news_data.columns

# Print or retrieve the column names
columnnames


# COMMAND ----------

# Group the data in 'training_fake_news_data' by the 'label' column and count the occurrences of each label
value_counts = training_fake_news_data.groupBy('label').count().orderBy("count", ascending=False)

# Collect the count data from the Spark DataFrame to the driver as a list of rows
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

# Regex Tokenizer breaks down text into individual words or tokens, useful for text processing tasks.
regex_tokenizer = RegexTokenizer(inputCol="clean_text", outputCol="words", pattern="\\W")

# Apply the tokenizer to the 'training_fake_news_data' DataFrame to create a new column 'words' 
training_fake_news_data = regex_tokenizer.transform(training_fake_news_data)

# Select and display the first 5 rows of the DataFrame showing the 'label', 'clean_text', and 'words' columns.
training_fake_news_data.select('label', 'clean_text', 'words').show(5)

# regex_tokenizer.save("/mnt/2024-team2/tokenizer")

# COMMAND ----------

#Instantiate StopwordRemover object that will remove the stopwords from the words
stopwords_remover = StopWordsRemover(inputCol="words", outputCol="filtered")

#Apply the tokenizer to the 'training_fake_news_data' DataFrame to create a new column 'words' 
training_fake_news_data = stopwords_remover.transform(training_fake_news_data)
training_fake_news_data.show(10)

# stopwords_remover.save(("/mnt/2024-team2/stopwords_remover")

# COMMAND ----------

#Define HashingTF transformer
hashing_tf = HashingTF(inputCol="filtered", 
                       outputCol="raw_features",  
                       numFeatures=3000) 

#Transform the input data using HashingTF
featurized_data = hashing_tf.transform(training_fake_news_data)

#Show the first 10 rows of the transformed DataFrame
featurized_data.show(10)

# hashing_tf.save("/mnt/2024-team2/hashing_tf")


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

# idf.save("/mnt/2024-team2/idf_model_path")


# COMMAND ----------


class CustomParamValidator:
    '''Created Param Validation class that will validate the model on different params and return the model with best params and model'''

    def __init__(self, estimator, customParamsList=None, evaluator=None, numFolds=3, seed=None):
        self.estimator = estimator
        self.customParamsList = customParamsList
        self.evaluator = evaluator
        
    def _fit(self, dataset):
        estimators = self.estimator
        params = self.customParamsList
        evaluator = self.evaluator
        
        #Custom cross-validation logic
        bestModel = None
        bestMetric = float('-inf')
        bestParams = None
        training_data, test_data = dataset.randomSplit([0.8, 0.2], seed=123)
        for param in params:
            model = estimators.fit(training_data, param)
            metric = evaluator.evaluate(model.transform(test_data))
            if metric > bestMetric:
              bestMetric = metric
              bestModel = model
              bestParams = params

        return bestModel,bestParams

# COMMAND ----------


estimator = LogisticRegression(featuresCol='features', labelCol='label')

customParamsList = [
    {estimator.maxIter: 10, estimator.regParam: 0.1},
    {estimator.maxIter: 20, estimator.regParam: 0.01},
]

#created binary class classsifier 
evaluator = BinaryClassificationEvaluator()

#Instantiate CustomParamValidator
customCrossValidator = CustomParamValidator(estimator=estimator, customParamsList=customParamsList, evaluator=evaluator)

bestModel, bestParams = customCrossValidator._fit(rescaled_data)
# bestModel.save('/mnt/2024-team2/lr-model')


# COMMAND ----------

#diplay best model
bestModel

# COMMAND ----------

#display bet params
bestParams

# COMMAND ----------

#Create a MulticlassClassificationEvaluator object for evaluating model performance
evaluator = MulticlassClassificationEvaluator(labelCol='label',predictionCol='prediction',metricName='accuracy')

# COMMAND ----------

#Preprocess the test data by cleaning the text and removing punctuation
test_fake_news_data = preprocessing_fake_news_text_data(test_fake_news_data, punctuation_chars)

#Tokenize the cleaned text using the regex tokenizer
test_fake_news_data = regex_tokenizer.transform(test_fake_news_data)

#Remove stop words from the tokenized text
test_fake_news_data = stopwords_remover.transform(test_fake_news_data)

#Transform the preprocessed test data into features
featurized_test_data = hashing_tf.transform(test_fake_news_data)

#Apply IDF transformation to the featurized test data
rescaled_test_data = idf_vectorizer.transform(featurized_test_data)


# COMMAND ----------

#Make predictions on the preprocessed and transformed test data using the best model obtained from custom param validation
predictions = bestModel.transform(rescaled_test_data)

# COMMAND ----------

#Evaluate the performance of the model by computing the accuracy on the predictions made for the test data
accuracy = evaluator.evaluate(predictions)

# COMMAND ----------

#display accuracy of the model 
accuracy

# COMMAND ----------

#calculate evaluaion metrics of the model 
accuracy = evaluator.evaluate(predictions, {evaluator.metricName: "accuracy"})
precision = evaluator.evaluate(predictions, {evaluator.metricName: "weightedPrecision"})
recall = evaluator.evaluate(predictions, {evaluator.metricName: "weightedRecall"})
f1_score = evaluator.evaluate(predictions, {evaluator.metricName: "f1"})

#Print the confusion metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1_score)


# COMMAND ----------

# predict_saved = bestModel.transform(rescaled_data)
# evaluator = MulticlassClassificationEvaluator(labelCol='label',predictionCol='prediction',metricName='accuracy')
# accuracy = evaluator.evaluate(predict_saved)

# COMMAND ----------





#Compute confusion metrics
confusion_metrics = predictions \
    .groupBy('prediction', 'label') \
    .count() \
    .withColumnRenamed('count', 'count') \
    .orderBy('prediction', 'label')

#Calculate true positives false positives true negatives and false negatives
true_positives = confusion_metrics.filter(expr('prediction == 1 AND label == 1')).select(expr('sum(count)')).collect()[0][0]
false_positives = confusion_metrics.filter(expr('prediction == 1 AND label == 0')).select(expr('sum(count)')).collect()[0][0]
true_negatives = confusion_metrics.filter(expr('prediction == 0 AND label == 0')).select(expr('sum(count)')).collect()[0][0]
false_negatives = confusion_metrics.filter(expr('prediction == 0 AND label == 1')).select(expr('sum(count)')).collect()[0][0]

#Output the confusion matrix
print("Confusion Matrix:")
print("True Positives:", true_positives)
print("False Positives:", false_positives)
print("True Negatives:", true_negatives)
print("False Negatives:", false_negatives)


#Define confusion matrix values
confusion_matrix = [[true_positives, false_negatives],
                    [false_positives, true_negatives]]
print(confusion_matrix)





# COMMAND ----------

#Stop Spark session
spark.stop()