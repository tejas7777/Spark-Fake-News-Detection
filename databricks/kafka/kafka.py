# Databricks notebook source
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from pyspark.sql.functions import col, explode, from_json, lit, current_timestamp

spark = SparkSession.builder.appName('kafka-stream') \
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.1.2") \
    .getOrCreate()



# COMMAND ----------

#Kafka Schema including source
json_schema = "array<struct<title:string, text:string, label:int, id:string, date:string>>"

#Read from Kafka
kafka_df = (
    spark
    .readStream
    .format("kafka")
    .option("kafka.bootstrap.servers", "ec2-13-60-88-247.eu-north-1.compute.amazonaws.com:9092")
    .option("subscribe", "news_topic")
    .option("startingOffsets", "latest")
    .option("failOnDataLoss", "false") 
    .load()
    .selectExpr("CAST(value AS STRING) AS json_string")
)

#Parse the JSON data
parsed_df = kafka_df.select(
    explode(
        from_json("json_string", json_schema)
    ).alias("data")
)

final_df = parsed_df.select(
    col("data.title"),
    col("data.text"),
    col("data.label"),
    col("data.id"),
    lit("kafka").alias("source"),  #Include source as 'kafka'
)

#Check point file had to be stored in a S3 bucket due to some access issues on ADLSgen2
checkpoint_location = "/mnt/2024-team2-s3a/stream"

query = final_df.writeStream \
    .outputMode("append") \
    .format("delta") \
    .option("path", "/mnt/delta/news_data") \
    .option("checkpointLocation", checkpoint_location) \
    .start()