# Exercise #2 (Un-supervised learning clustering)
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, isnan, mean, min as sql_min, max as sql_max
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans

# Create Spark session
spark = SparkSession.builder.appName("WineClustering").getOrCreate()

####################### 1. Download the wine dataset wine.csv accompanied with
# this assignment and move it to a folder on your virtual machine.

####################### 2. Load the wine dataset into a data frame named wine_x1
wine_matheus1 = spark.read.csv("D:/pCloudFolder/Repositories/Centennial/Semester6/BigData/Assignment4/Exercise2/wine.csv", 
                              header=True, 
                              inferSchema=True,
                              sep=';')  # Specify semicolon as separator

####################### 3. Using spark high level api functions
# (i.e. not pandas), carry out some initial investigation and record the results
# in your analysis, at minimum provide the following:

##### a. Printout the names of columns
print("\na. Column Names:")
print(wine_matheus1.columns)

##### b. Printout the types of each column
print("\nb. Column Types:")
print(wine_matheus1.dtypes)

##### c. Printout the basic statistics mean, median, the four quartiles
print("\nc. Basic Statistics:")
wine_matheus1.describe().show()

##### d. Printout the minimum, maximum value for each column
print("\nd. Min and Max Values for each column:")
for column in wine_matheus1.columns:
    min_max = wine_matheus1.agg(
        sql_min(column).alias("min"),
        sql_max(column).alias("max")
    ).collect()[0]
    print(f"{column}: Min = {min_max['min']}, Max = {min_max['max']}")

##### e. Generate and printout a table showing the number of missing values for
# each column. (Hint: use isnan, when, count, col)
print("\ne. Missing Values Count:")
for column in wine_matheus1.columns:
    missing_count = wine_matheus1.filter(col(column).isNull() | isnan(col(column))).count()
    print(f"{column}: {missing_count} missing values")

###################### 4. Show all the distinct values in the "quality" column.
print("\nDistinct Quality Values:")
wine_matheus1.select("quality").distinct().orderBy("quality").show()

###################### 5. Show the mean of the various chemical compositions
# across samples for the different groups of the wine quality.
print("\nMean Chemical Composition by Quality:")
columns_to_analyze = wine_matheus1.columns
columns_to_analyze.remove("quality")

wine_matheus1.groupBy("quality").agg(
    *[mean(col).alias(col) for col in columns_to_analyze]
).orderBy("quality").show()

###################### 6. Re-load the wine dataset into a data frame named
# wine_x as you load add a new column named feature_x of vector type that
# contains four columns as follows:
# "citric acid", "volatile acidity", "chlorides", "sulphates"
# Spread the data frame across 3 RDD partitions. (Hint: use coalesce)
vector_assembler = VectorAssembler(
    inputCols=["citric acid", "volatile acidity", "chlorides", "sulphates"],
    outputCol="feature_matheus"
)

wine_matheus = spark.read.csv("D:/pCloudFolder/Repositories/Centennial/Semester6/BigData/Assignment4/Exercise2/wine.csv", 
                             header=True, 
                             inferSchema=True,
                             sep=';')  # Specify semicolon as separator
wine_matheus = vector_assembler.transform(wine_matheus)
wine_matheus = wine_matheus.coalesce(3)

###################### 7. Cache the dataframe.
wine_matheus.cache()

###################### 8. Define a estimator that uses K-means clustering to
# cluster all the wine instances into 6 clusters using the new feature_x vector
# column you added in step #6.
kmeans_6 = KMeans(k=6, featuresCol="feature_matheus")
model_6 = kmeans_6.fit(wine_matheus)

# Get cluster sizes
predictions_6 = model_6.transform(wine_matheus)
print("\nCluster Sizes (k=6):")
cluster_sizes_6 = predictions_6.groupBy("prediction").count().orderBy("prediction")
cluster_sizes_6.show()

print("\nCluster Centroids (k=6):")
for i, center in enumerate(model_6.clusterCenters()):
    print(f"Cluster {i} center: {center}")

##################### 9. Print the cluster sizes and the cluster centroids,
# record the results in your analysis report and write some conclusions.

##################### 10. Repeat steps 8&9 but set the number of k to 4.
kmeans_4 = KMeans(k=4, featuresCol="feature_matheus")
model_4 = kmeans_4.fit(wine_matheus)

# Get cluster sizes
predictions_4 = model_4.transform(wine_matheus)
print("\nCluster Sizes (k=4):")
cluster_sizes_4 = predictions_4.groupBy("prediction").count().orderBy("prediction")
cluster_sizes_4.show()

print("\nCluster Centroids (k=4):")
for i, center in enumerate(model_4.clusterCenters()):
    print(f"Cluster {i} center: {center}")

# Uncache the dataframe
wine_matheus.unpersist()

# Stop Spark session
spark.stop()