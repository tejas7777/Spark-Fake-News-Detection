# Databricks notebook source
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from pyspark.sql.functions import lit, col, trim, when, current_timestamp, sha1


#Define the schema for true and false data column
common_schema = StructType([
    StructField('title', StringType(), True),
    StructField('text', StringType(), True),
    StructField('date', StringType(), True)
])

schema1 = StructType([
    StructField('text', StringType(), True),
    StructField('label', IntegerType(), True)
])

schema2 = StructType([
    StructField('id', IntegerType(), True),
    StructField('title', StringType(), True),
    StructField('author', StringType(), True),
    StructField('text', StringType(), True),
    StructField('label', IntegerType(), True)
])


# COMMAND ----------

#Merge the two files
df1 = spark.read.schema(common_schema).csv("/mnt/2024-team2/dataset/data_true.csv").withColumn('label', lit(1))
df2 = spark.read.schema(common_schema).csv("/mnt/2024-team2/dataset/data_false.csv").withColumn('label', lit(0))
df1 = df1.drop('subject')
df2 = df2.drop('subject')

#Reading and modifying df3
df3 = spark.read.option("header", "true").option("delimiter", "\t").schema(schema1).csv("/mnt/2024-team2/dataset/dataset3.csv")
df3 = df3.withColumn("title", lit(None)).withColumn("date", lit(None))
df3 = df3.withColumn("label", when(col("label") == 1, 0).otherwise(1))

#Reading and modifying df4
df4 = spark.read.option("header", "true")\
                .option("sep", ",")\
                .option("quote", "\"")\
                .option("escape", "\"")\
                .option("multiLine", "true")\
                .csv("/mnt/2024-team2/dataset/dataset4.csv").select('title', 'text', 'label')\
                .withColumn("date", lit(None))
df4 = df4.withColumn("label", when(col("label") == 1, 0).otherwise(1))

#Combine all DataFrames into one unified DataFrame
df = df1.unionByName(df2).unionByName(df3).unionByName(df4)

#Show the combined DataFrame with all expected columns
df.show(10)


# COMMAND ----------

df1 = spark.read.option("header", "true").csv("/mnt/2024-team2/dataset/data_true.csv").withColumn("label", lit(1))
df2 = spark.read.option("header", "true").csv("/mnt/2024-team2/dataset/data_false.csv").withColumn("label", lit(0))

#Drop unnecessary columns if present
if "subject" in df1.columns:
    df1 = df1.drop("subject")
if "subject" in df2.columns:
    df2 = df2.drop("subject")

#Reading CSV files for df3 and df4
df3 = spark.read.option("header", "true").option("delimiter", "\t").csv("/mnt/2024-team2/dataset/dataset3.csv")
df4 = spark.read.option("header", "true").option("sep", ",").option("quote", "\"").option("escape", "\"").option("multiLine", "true").csv("/mnt/2024-team2/dataset/dataset4.csv")

#Adjust columns to match (add missing columns with default values)
df3 = df3.withColumn("date", lit(None)).withColumn("title", lit(None))
df4 = df4.withColumn("date", lit(None)).withColumn("title", lit(None))

#Optionally invert labels in df3 and df4 if required
df3 = df3.withColumn("label", when(col("label") == 1, 0).otherwise(1))
df4 = df4.withColumn("label", when(col("label") == 1, 0).otherwise(1))

#Select only necessary columns to ensure schema alignment
df1 = df1.select("title", "text", "date", "label")
df2 = df2.select("title", "text", "date", "label")
df3 = df3.select("title", "text", "date", "label")
df4 = df4.select("title", "text", "date", "label")

#Union all DataFrames
df_unioned = df1.unionByName(df2).unionByName(df3).unionByName(df4)

#Show the results
df_unioned.show(100)

# COMMAND ----------

#Uniform the third file
df5 = spark.read.csv("/mnt/2024-team2/dataset/welfake_data.csv",header=True, inferSchema=True)


df5 = df5.drop('_c0')

df5 = df5.withColumn('date', lit(None).cast(StringType()))
df5 = df5.withColumn('label', col('label').cast(IntegerType()))

df_combined = df_unioned.unionByName(df5)
df_combined.show(10)

# COMMAND ----------

df_combined.count()

# COMMAND ----------

df_cleaned = df_combined.filter(df_combined.text.isNotNull() & df_combined.label.isNotNull())
df_cleaned = df_cleaned.filter(col('label').isin([0, 1]))
df_cleaned.show(1)

# COMMAND ----------

df_cleaned_filtered = df_cleaned.filter(trim(df_cleaned.text) != "")

# COMMAND ----------

df_cleaned_filtered.schema
df = df_cleaned_filtered.withColumn("timestamp", current_timestamp())
df = df.withColumn("id", sha1("text"))
df.show(2)

# COMMAND ----------

df.count()

# COMMAND ----------

#Create delta table
delta_table_path = "/mnt/delta/news_delta"

# Write the DataFrame to the Delta format
df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(delta_table_path)

# COMMAND ----------

spark.sql("CREATE TABLE news_delta USING DELTA LOCATION '/mnt/delta/news_delta'")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(*) FROM news_data;