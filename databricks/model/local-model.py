# Databricks notebook source
from pyspark.sql import SparkSession
from pyspark.ml.feature import RegexTokenizer, HashingTF, IDF, StopWordsRemover
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import lit, regexp_replace, lower, explode, col
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from sklearn.linear_model import LogisticRegression
import pandas as pd 
import numpy as np 
import re
import matplotlib.pyplot as plt

from pyspark.ml.evaluation import MulticlassClassificationEvaluator



# COMMAND ----------

fake_news_dataset = spark.read.format("delta").table("news_data")
common_schema = StructType([
    StructField('title', StringType(), True),
    StructField('text', StringType(), True),
    StructField('date', StringType(), True)
])

fake_news_dataset.cache()



# COMMAND ----------

#Selecting only the 'label' and 'text' columns from the fake_news_dataset DataFrame
fake_news_dataset = fake_news_dataset.select('label', 'text')

#Filling any missing values
fake_news_dataset = fake_news_dataset.fillna({'text': ''})


# COMMAND ----------

#Splitting the dataset into training and test datasets with a ratio of 70% for training and 30% for testing
#he seed parameter ensures reproducibility of the random split
(training_fake_news_data, test_fake_news_data) = fake_news_dataset.randomSplit([0.7, 0.3], seed=100)

#Getting the column names of the fake_news_dataset DataFrame
columnnames = fake_news_dataset.columns

#Printing the column names
columnnames


# COMMAND ----------

 
punctuation_chars = '!#$%&'

def preprocessing_text_data(dataset, punctuation_chars):
    #Convert text to lowercase
    dataset = dataset.withColumn('text', lower(dataset['text']))

    #Remove URLs
    dataset = dataset.withColumn("text", regexp_replace(dataset["text"], r"http[s]?\://\S+", ""))

    #Remove text within parentheses or square brackets
    dataset = dataset.withColumn("text", regexp_replace(dataset["text"], r"(\(.*\))|(\[.*\])", ""))

    #Remove words containing consecutive asterisks
    dataset = dataset.withColumn("text", regexp_replace(dataset["text"], r"\b\w+\*{2,3}\w*\b", ""))

    #Remove special characters and punctuation
    dataset = dataset.withColumn("text", regexp_replace(dataset["text"], r'[!#$%&\'()*+,-./:;<=>?@[\\]^_`{|}]+', ""))
    dataset = dataset.withColumn("text", regexp_replace(dataset["text"], r"[" + re.escape(punctuation_chars) + "]", ""))

    return dataset



training_fake_news_data = preprocessing_text_data(training_fake_news_data,punctuation_chars)
training_fake_news_data.show()

# COMMAND ----------

regex_tokenizer = RegexTokenizer(inputCol="text", outputCol="words", pattern="\\W")
training_fake_news_data = regex_tokenizer.transform(training_fake_news_data)
training_fake_news_data.select('label', 'text', 'words').show(5)
# tokenizer.save("/mnt/2024-team2/local/tokenizer")

# COMMAND ----------

stopwords_remover = StopWordsRemover(inputCol="words", outputCol="filtered")
training_fake_news_data = stopwords_remover.transform(training_fake_news_data)
training_fake_news_data.show(10)
# stopwords_remover.save("/mnt/2024-team2/local/stopwords_remover")

# COMMAND ----------

#Create a HashingTF (Term Frequency) instance with input and output columns specified, and the number of features set to 1000
hashing_tf = HashingTF(inputCol="filtered", outputCol="raw_features", numFeatures=3000)

#Transform the training_fake_news_data using the HashingTF model to obtain the raw feature vectors
featurized_data = hashing_tf.transform(training_fake_news_data)

# hashing_tf.save("/mnt/2024-team2/local/hashing_tf")

# COMMAND ----------

featurized_data.show()

# COMMAND ----------

#Create an IDF instance 
idf = IDF(inputCol="raw_features", outputCol="features")

#Fit the IDF model to the featurized data to compute the IDF weights
idf_vectorizer = idf.fit(featurized_data)

#Transform the featurized data using the IDF model to obtain the TF-IDF weighted vectors
rescaled_data = idf_vectorizer.transform(featurized_data)

#Display the transformed data with TF-IDF weighted vectors
rescaled_data.show()

# idf_vectorizer.save("/mnt/2024-team2/local/idf_vectorizer")
# rescaled_data.count()


# COMMAND ----------

rescaled_data.show()
type(rescaled_data)

# COMMAND ----------

#Global variable
global column_names
column_names = rescaled_data.columns

#Build a logistic regression model on each partition of data
def build_model(partition_data_it):
    try:
        print('Inside build_model')
        
        # Convert the partition data iterator to a pandas DataFrame with column names
        partition_data_it = pd.DataFrame(partition_data_it, columns=column_names)
        
        # Extract features and labels from the partition data
        X_train = list(partition_data_it['features'])
        Y_train = partition_data_it['label']
        
        # Initialize and train a logistic regression model
        clf = LogisticRegression()
        model = clf.fit(X_train, Y_train)
        
        # Return the trained model
        return [model]
    
    except Exception as e:
        print('Inside Exception')
        import traceback
        print(traceback.print_exc)
        print(e)

#Repartition the rescaled data RDD into 5 partitions
training_fake_news_rdd = rescaled_data.rdd.repartition(5)
type(training_fake_news_rdd)

#Apply the build_model function to each partition of the RDD
transformed_fake_news_rdd = training_fake_news_rdd.mapPartitions(build_model)

# collect the models generated from each partition
try:
    models = transformed_fake_news_rdd.collect()
    print("Transformation successful.")
except Exception as e:
    print("Error during transformation:", e)


# COMMAND ----------

models

# COMMAND ----------

#Define a function to predict the label for a given instance using a list of models
def predict(instance):
    # Predict the label for the instance using each model in the list of models
    # Return a list of predictions
    return [m.predict([instance['features']]) for m in models]

#Define a function to aggregate predictions and determine the final label
def agg_predictions(preds):
    #Initialize a dictionary to store the count of each label
    prediction = {0: 0, 1: 0}
    
    #Iterate over the predictions and update the count for each label
    for elem in preds:
        prediction[elem[0]] += 1
    
    #Return the label with the highest count
    return max(prediction, key=prediction.get)

#Preprocess the test data
test_fake_news_data = preprocessing_text_data(test_fake_news_data, punctuation_chars)
test_fake_news_data = regex_tokenizer.transform(test_fake_news_data)
test_fake_news_data = stopwords_remover.transform(test_fake_news_data)
featurized_data = hashing_tf.transform(test_fake_news_data)
test_fake_news_data = idf_vectorizer.transform(featurized_data)


# COMMAND ----------

from pyspark.sql.types import Row

#Define a function to transform each instance in the test data
def transform(instance):
    #Generate raw predictions for the instance using the agg_predictions function
    #Convert the result to a Row object
    return Row(**instance.asDict(), raw_prediction=agg_predictions(predict(instance)))

#Repartition the testData RDD into 10 partitions and apply the transform function to each instance
#Convert the transformed RDD to a DataFrame named 'prediction'
prediction = test_fake_news_data.rdd.repartition(10).map(transform).toDF()

#Show the contents of the 'prediction' DataFrame
prediction.show()


# COMMAND ----------

#This code is return to test the model on different partition of the dataset. 
partition_accuracy={}
for partition in range(5,20):
  training_fake_news_rdd = rescaled_data.rdd.repartition(partition)
  transformed_fake_news_rdd = training_fake_news_rdd.mapPartitions(build_model)
  try:
      models = transformed_fake_news_rdd.collect()
      print("Transformation successful.")
  except Exception as e:
    print("Error during transformation:", e) 
  prediction=test_fake_news_data.rdd.repartition(partition).map(transform).toDF()
  prediction_num=prediction.select((prediction['label']==0).cast('double').alias('label'),
                                  (prediction['raw_prediction']==0).cast('double').alias('raw_prediction'),
                                  )
  
  acc_evaluator=MulticlassClassificationEvaluator(metricName='accuracy',labelCol='label',predictionCol='raw_prediction')
  accuracy=acc_evaluator.evaluate(prediction_num)
  partition_accuracy[partition]=accuracy
print(partition_accuracy)

  


# COMMAND ----------

prediction.cache()
prediction_num=prediction.select((prediction['label']==0).cast('double').alias('label'),
                                 (prediction['raw_prediction']==0).cast('double').alias('raw_prediction'),
                                 )

acc_evaluator=MulticlassClassificationEvaluator(metricName='accuracy',labelCol='label',predictionCol='raw_prediction')
prediction_num.cache()
acc_evaluator.evaluate(prediction_num)

# COMMAND ----------


#Extracting partition values and corresponding accuracy values from the dictionary
parition_value = list(partition_accuracy.keys())
partition_accuracy = list(partition_accuracy.values())

#Plotting the accuracy vs number of partitions
plt.figure(figsize=(10, 6))
plt.plot(parition_value, partition_accuracy, marker='o', linestyle='-')
plt.title('Accuracy vs Number of Partitions')
plt.xlabel('Number of Partitions')
plt.ylabel('Accuracy')
plt.grid(True)
plt.xticks(parition_value)
plt.show()

# COMMAND ----------

models

# COMMAND ----------

#Calculate various evaluation metrics 
accuracy = acc_evaluator.evaluate(prediction_num, {acc_evaluator.metricName: "accuracy"})
precision = acc_evaluator.evaluate(prediction_num, {acc_evaluator.metricName: "weightedPrecision"})
recall = acc_evaluator.evaluate(prediction_num, {acc_evaluator.metricName: "weightedRecall"})
f1_score = acc_evaluator.evaluate(prediction_num, {acc_evaluator.metricName: "f1"})

#Print the confusion metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1_score)

# COMMAND ----------

