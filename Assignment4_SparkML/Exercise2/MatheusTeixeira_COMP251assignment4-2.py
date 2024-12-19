# Exercise #2 (Un-supervised learning clustering) 
from pyspark.sql import SparkSession
from pyspark.sql.functions import sum, format_string, col, isnan, mean
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans

# Create Spark session
spark = SparkSession.builder.appName("WineClustering").getOrCreate()

####################### 2. Load the wine dataset into a data frame named wine_x1
wine_matheus1 = spark.read.csv("D:/pCloudFolder/Repositories/Centennial/Semester6/BigData/Assignment4/Exercise2/wine.csv", 
                              header=True, 
                              inferSchema=True,
                              sep=';')

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
##### d. Printout the  minimum, maximum value for each column
print("\nc. Basic Statistics:")
wine_matheus1.summary().show()

##### e. Generate and printout a table showing the number of missing values for
# each column. (Hint: use isnan, when, count, col)
missing_df = spark.createDataFrame(
    [(column, wine_matheus1.filter(col(column).isNull() | isnan(col(column))).count())
     for column in wine_matheus1.columns],
    ["Column", "Missing_Count"]
)
missing_df.show(truncate=False)

###################### 4. Show all the distinct values in the “quality” column along with their count.
print("\nDistinct Quality Values with Counts:")
distinct_values_count = wine_matheus1.groupBy("quality").count().orderBy("quality")
distinct_values_count.show()

###################### 5. Show the mean of the various chemical compositions
# across samples for the different groups of the wine quality.
print("\nMean Chemical Composition by Quality:")
columns_to_analyze = wine_matheus1.columns
columns_to_analyze.remove("quality")

wine_matheus1.groupBy("quality").agg(
    *[mean(col).alias(col) for col in columns_to_analyze]
).orderBy("quality").show()


###################### 6. Re-load the wine dataset into a data frame named
# wine_x as you load add a new column named feature_x  of vector type that
# contains four columns as follows:
# “citric acid", "volatile acidity", "chlorides", "sulphates"
# Spread the data frame across 3 RDD partitions.  (Hint: use coalesce)
vector_assembler = VectorAssembler(
    inputCols=["citric acid", "volatile acidity", "chlorides", "sulphates"],
    outputCol="feature_matheus"
)

wine_matheus = spark.read.csv("D:/pCloudFolder/Repositories/Centennial/Semester6/BigData/Assignment4/Exercise2/wine.csv", 
                             header=True, 
                             inferSchema=True,
                             sep=';')
wine_matheus = vector_assembler.transform(wine_matheus)
wine_matheus = wine_matheus.coalesce(3)

###################### 7. Cache the dataframe. 
wine_matheus.cache()

###################### 8. Define a estimator that uses K-means clustering to
# cluster all the wine instances into 6 clusters using the new feature_x  vector
# column you added in step #6.
kmeans_6 = KMeans(k=6, featuresCol="feature_matheus")
model_6 = kmeans_6.fit(wine_matheus)

###################### 9. Print the cluster sizes and the cluster centroids,
# record the results in your analysis report and write some conclusions.
# Get cluster sizes
predictions_6 = model_6.transform(wine_matheus)
print("\nCluster Sizes (k=6):")
cluster_sizes_6 = predictions_6.groupBy("prediction").count().orderBy("prediction")
cluster_sizes_6.show()

print("\nCluster Centroids (k=6):")
for i, center in enumerate(model_6.clusterCenters()):
    print(f"Cluster {i} center: {center}")

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


####################################################################

def print_cluster_analysis(predictions_df, k):
    print(f"\nAnalysis for k={k} clusters:")

    # Table 1: Counts of wine per quality in each cluster
    print(f"\nTable 1: Counts of wine per quality in each cluster:")
    count_matrix = predictions_df.crosstab("prediction", "quality")
    # Rename columns to remove 'quality_' prefix
    for col_name in count_matrix.columns:
        if col_name.startswith('quality_'):
            count_matrix = count_matrix.withColumnRenamed(col_name, col_name.replace('quality_', ''))
    count_matrix.orderBy("prediction_quality").show()

    # Table 2: Percentage of wine per quality in each cluster
    print(f"\nTable 2: Percentage of wine per quality in each cluster:")
    total_counts = count_matrix.select([sum(col).alias(col) for col in count_matrix.columns if col != 'prediction_quality'])

    # Broadcast the total counts to normalize the values and calculate the percentages
    total_counts_dict = total_counts.collect()[0].asDict()
    percentage_matrix = count_matrix
    for col_name in count_matrix.columns:
        if col_name != 'prediction_quality':
            percentage_matrix = percentage_matrix.withColumn(
                col_name, 
                format_string("%d%%", (col(col_name) / total_counts_dict[col_name] * 100).cast('int'))
            )

    percentage_matrix.orderBy("prediction_quality").show()

# Print analysis for both k=6 and k=4
print("\n" + "="*50)
print("CLUSTERING ANALYSIS")
print("="*50)

print_cluster_analysis(predictions_6, 6)
print_cluster_analysis(predictions_4, 4)
