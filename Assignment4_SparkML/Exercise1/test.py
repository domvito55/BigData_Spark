from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Create Spark session
spark = SparkSession.builder.appName("DecisionTreeClassification").getOrCreate()

####################### 2. Load the data
# Load LIBSVM format data
df_john = spark.read.format("libsvm").load("/home/centos/data/sample_libsvm_data.txt")

####################### 3. Basic investigation
# Count records
num_records = df_john.count()
print(f"Number of records: {num_records}")

# Print schema
print("Inferred Schema:")
df_john.printSchema()

# Count columns
num_columns = len(df_john.columns)
print(f"Number of columns: {num_columns}")

####################### 4. Index labels
# Create StringIndexer for labels
labelIndexer_john = StringIndexer(
    inputCol="label",
    outputCol="indexedLabel_john",
    handleInvalid="error"
)

####################### 5. Index features
# Create VectorIndexer for features
featureIndexer_john = VectorIndexer(
    inputCol="features",
    outputCol="indexedFeatures_john",
    maxCategories=4
)

####################### 6. Print indexer information
# Fit the featureIndexer to get information
fitted_featureIndexer = featureIndexer_john.fit(df_john)

print("\nFeature Indexer Information:")
print(f"a. Input Column: {featureIndexer_john.getInputCol()}")
print(f"b. Output Column: {featureIndexer_john.getOutputCol()}")
print(f"c. Number of Features: {fitted_featureIndexer.numFeatures}")
print(f"d. Category Map: {fitted_featureIndexer.categoryMaps}")

####################### 7. Split data
# Split the data into training and testing sets
(training_john, testing_john) = df_john.randomSplit([0.65, 0.35], seed=42)

####################### 8. Create Decision Tree estimator
# Initialize DecisionTreeClassifier
DT_john = DecisionTreeClassifier(
    labelCol="indexedLabel_john",
    featuresCol="indexedFeatures_john"
)

####################### 9. Create pipeline
# Create the pipeline with all stages
pipeline_john = Pipeline(stages=[
    labelIndexer_john,
    featureIndexer_john,
    DT_john
])

####################### 10. Fit the model
# Train the model using the pipeline
model_john = pipeline_john.fit(training_john)

####################### 11. Make predictions
# Make predictions on test data
predictions_john = model_john.transform(testing_john)

####################### 12. Print predictions schema
print("\nPredictions Schema:")
predictions_john.printSchema()

####################### 13. Calculate accuracy and test error
# Create evaluator
evaluator = MulticlassClassificationEvaluator(
    labelCol="indexedLabel_john",
    predictionCol="prediction",
    metricName="accuracy"
)

# Calculate accuracy
accuracy = evaluator.evaluate(predictions_john)
test_error = 1.0 - accuracy

print(f"\nAccuracy: {accuracy}")
print(f"Test Error: {test_error}")

####################### 14. Show first 10 predictions
print("\nFirst 10 predictions:")
predictions_john.select("prediction", "indexedLabel_john", "features").show(10)

# Stop Spark session
spark.stop()