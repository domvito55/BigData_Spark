# Exercise #1 (Supervised learning decision trees) 
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Create Spark session
spark = SparkSession.builder.appName("DecisionTreeClassification").getOrCreate()

####################### 2. Load the data stored in the file
# “sample_libsvm_data.txt” from the data available on the VMware image under
# the directory /home/centos/data/ into a dataframe and name it df_x where x is
# your firstname. Use infer schema, notice that you need to use the format is
#LIBSVM when you create the dataframe.
df_matheus = spark.read.format("libsvm").load("/home/centos/data/sample_libsvm_data.txt")

####################### 3. Carry out some basic investigation:
# count the number of records
num_records = df_matheus.count()
print(f"Number of records: {num_records}")

# count the number of columns
num_columns = len(df_matheus.columns)
print(f"Number of columns: {num_columns}")

# print the inferred schema and
print("Inferred Schema:")
df_matheus.printSchema()

# explain what each column contains and
# record the results in your analysis report.

####################### 4. Use the StringIndexer to index labels, in other
# words you will add metadata to the label column. Name the output column
# "indexedLabel_x” where x is your first name. Store the result in a variable
# named labelIndexer_x where x is your first name.  To Learn more about the
# StringIndexer checkout the following link
# https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.feature.StringIndexer.html
labelIndexer_matheus = StringIndexer(
    inputCol="label",
    outputCol="indexedLabel_matheus",
    handleInvalid="error"
)

###################### 5. Use the VectorIndexer to automatically identify
# categorical features, and index them. Set the maxCategories to 4. Name the
# output column " indexedFeatures _x” where x is your first name. Store the
# result in a variable named featureIndexer _x where x is your first name.
# To Learn more about the VectorIndexer checkout the following link
# https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.feature.VectorIndexer.html
featureIndexer_matheus = VectorIndexer(
    inputCol="features",
    outputCol="indexedFeatures_matheus",
    maxCategories=4,
    handleInvalid="keep"  # Change this to "skip" if you prefer to skip invalid rows
)

##################### 6. Printout the following:
print("\nFeature Indexer Information:")
fitted_featureIndexer = featureIndexer_matheus.fit(df_matheus)

##### a. Name of input column
print(f"a. Input Column: {featureIndexer_matheus.getInputCol()}")

##### b. Name of output column
print(f"b. Output Column: {featureIndexer_matheus.getOutputCol()}")

##### c. # of features
print(f"c. Number of Features: {fitted_featureIndexer.numFeatures}")

##### d. Map of categories
print(f"d. Category Map: {fitted_featureIndexer.categoryMaps}")

# Also note the results in your written response.

###################### 7. Split your original data into 65% for training and
# 35% for testing and store the training data into a datafrmae named training_x
# and testing_x respectively  where x is your firstname.
(training_matheus, testing_matheus) = df_matheus.randomSplit([0.65, 0.35], seed=42)

###################### 8. Create an estimator object that contains a decision
# tree classifier make sure to set the correct input and output columns you
# created during the transformation steps 4 & 5 above. Name the estimator DT_x
# where x is your firstname. 
DT_matheus = DecisionTreeClassifier(
    labelCol="indexedLabel_matheus",
    featuresCol="indexedFeatures_matheus"
)

###################### 9. Create a pipeline object with three stages the first
# two are the transformers you defined in steps 4 & 5 and the third is the
# decision tree estimator you defined in step 8. Name the pipeline object
# pipeline_x where x is your firstname.
pipeline_matheus = Pipeline(stages=[
    labelIndexer_matheus,
    featureIndexer_matheus,
    DT_matheus
])

##################### 10. Fit the training data to the pipeline. Store the
# results into an object named model_x, where x is your first name.
model_matheus = pipeline_matheus.fit(training_matheus)

##################### 11. Using the model_x predict the testing data. Store the
# results into a dataframe named predictions_x where x is your firstname.
predictions_matheus = model_matheus.transform(testing_matheus)

##################### 12. Print the schema of the predictions and note the
# results into your analysis report.
print("\nPredictions Schema:")
predictions_matheus.printSchema()

##################### 13. Print the accuracy of your model and the test error
# and note the results in your analysis report.
evaluator = MulticlassClassificationEvaluator(
    labelCol="indexedLabel_matheus",
    predictionCol="prediction",
    metricName="accuracy"
)

# Calculate accuracy
accuracy = evaluator.evaluate(predictions_matheus)
test_error = 1.0 - accuracy

print(f"\nAccuracy: {accuracy}")
print(f"Test Error: {test_error}")

##################### 14. Show the first 10 predictions with the actual labels
# and features take a screenshot and add it to your analysis report.
print("\nFirst 10 predictions:")
predictions_matheus.select("prediction", "indexedLabel_matheus", "features").show(10)

