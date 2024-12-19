#### 1-Load two days of data from the retail sales data available on the VMware
# image under the directory /home/centos/data/retail-data/by-day/ into a
# dataframe and name it df_x where x is your firstname. (use infer schema)
# For students with firstname starting A-M please load the data for ninth and
# the tenth of December (2010-12-09.csv and 2010-12-10.csv)
# For students with firstname starting M-Z please load the data for ninth and
# the tenth of January

# Load the data into a DataFrame
df_Matheus = spark.read.format("csv") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .load(["/home/centos/data/retail-data/by-day/2010-12-09.csv",
           "/home/centos/data/retail-data/by-day/2010-12-10.csv"])

#### 2- Check the UI at the local host port 4040 or the port that spark connects
# to when launched and record the following in your analysis report:
# a. The time it took to load the data.
# b. The number of tasks and try to explain in your own words what happened
# in the analysis report.
# Take a screenshot of the DAG execution and add it to your analysis report.

# For points 3, 4, 5, 6, and 7 use the Dataframe high level API, make sure you
# show the full column content, i.e. no truncation.
spark.conf.set("spark.sql.repl.eagerEval.truncate", 1000)

#### 3- Carry out some basic investigation: count the number of records,
record_count = df_Matheus.count()
print(f"Number of records: {record_count}")

# print the inferred schema. Record the results in your analysis report.
df_Matheus.printSchema()

#### 4- Show all the transactions that are related to the purchase of stock id
# that starts with "227" with the type of product “ALARM CLOCK” mentioned as
# part of the description or a unit price greater than 5.
from pyspark.sql.functions import col

filtered_df = df_Matheus.filter((col("StockCode").startswith("227")) & (
    (col("Description").contains("ALARM CLOCK")) | (col("UnitPrice") > 5)))
filtered_df.show(n=filtered_df.count(), truncate=False)

#### 5- Store the results into a new dataframe name it df2_firstname.
# Store the results into a new dataframe
df2_Matheus = filtered_df

#### 6- Show the sum of the quantities ordered and the minimum quantity order
# and the maximum quantity order for the transactions you extracted in point 4
# above.
from pyspark.sql.functions import sum as _sum, min as _min, max as _max

df2_Matheus.select(
    _sum("Quantity").alias("Total Quantity"),
    _min("Quantity").alias("Minimum Quantity"),
    _max("Quantity").alias("Maximum Quantity")).show(truncate=False)

# Investigate the UI take a screenshot of the DAG plan and in your
# "Analysis report" add:

# the number of stages the job required with
# the total time required per stage in addition to
# the number of tasks required for each job.

# Finally, drill down on each stage and produce the DAG graph for each
# stage and analyze the statistics, note the shuffle size and the number of
# partitions in your report.

#### 7- Show all the transactions mentioned in point 4 above that have originated
# from outside the United Kingdom.
non_uk_transactions = df2_Matheus.filter(col("Country") != "United Kingdom")
non_uk_transactions.show(n=non_uk_transactions.count(), truncate=False)
