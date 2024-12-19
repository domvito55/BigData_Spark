from pyspark.sql import SparkSession
spark = SparkSession\
        .builder\
        .appName("Graph1")\
        .config("spark.jars", r"graphframes-0.8.1-spark3.0-s_2.12.jar")\
        .getOrCreate()
# Configure your session to run on 4 partitions. Add all the commands to the Python script. Take a full screenshot. (3 marks)
spark.conf.set("spark.sql.shuffle.partitions", "4")

# Represent the graph described below in a Spark GraphFrame object, and name the frame firstname_graph where first_name is your first name.
# Graph Description:
# Nodes:
# Emma, 34, Teacher
# Liam, 17, Student
# Mia, 40, Principal
# Noah, 15, Student
# Ava, 45, Librarian
# Lucas, 38, Sports Coach
v = spark.createDataFrame([
  ("Emma", 34, "Teacher"),
  ("Liam", 17, "Student"),
  ("Mia", 40, "Principal"),
  ("Noah", 15, "Student"),
  ("Ava", 45, "Librarian"),
  ("Lucas", 38, "Sports Coach")
], ["id", "age", "occupations"])
v.take(2)

# Edges:
# Emma teaches Liam and Noah.
# Mia supervises Emma and Ava.
# Ava assists Mia.
# Lucas coaches Liam and Noah.
e = spark.createDataFrame([
  ("Emma", "Liam", "teaches"),
  ("Emma", "Noah", "teaches"),
  ("Mia", "Emma", "supervises"),
  ("Mia", "Ava", "supervises"),
  ("Ava", "Mia", "assists"),
  ("Lucas", "Liam", "coaches"),
  ("Lucas", "Noah", "coaches")
], ["src", "dst", "relationship"])
e.take(2)

from graphframes import *
matheus_graph = GraphFrame(v, e)
# Take full screenshots of all the steps you took to create the GraphFrame object and add them to your word document. Add all the commands to the Python script. (15 marks)

# Query the graph to show the oldest person and display the result on the console.
oldest_person = matheus_graph.vertices.orderBy("age", ascending=False).first()
print(f"The oldest person is {oldest_person['id']} with age {oldest_person['age']}")
# Take full screenshots of all the steps you took to achieve the requirement and the output, and add them to your word document. Add all the commands to the Python script. (6 marks)

# Query the graph to show the names of all students who are directly or indirectly connected to a teacher.
students_connected_to_teacher = matheus_graph.bfs("occupations = 'Teacher'", "occupations = 'Student'")
students_connected_to_teacher.show()
# Take full screenshots of all the steps you took to achieve the requirement and the output, and add them to your word document. Add all the commands to the Python script. (6 marks)

# Query the graph to show the top two people who are most connected through indirect relationships (e.g., "friends of friends").
indirect_relationships = matheus_graph.find("(a)-[]->(b); (b)-[]->(c)")
indirect_relationships.show()

# Take full screenshots of all the steps you took to achieve the requirement and the output, and add them to your word document. Add all the commands to the Python script. (8 marks)

# Query the graph to identify the strongest components:
# Show a table indicating the components and the count of vertices in each component.
components = matheus_graph.stronglyConnectedComponents(maxIter=10)
# Show a table indicating the components and the count of vertices in each component
from pyspark.sql.functions import desc
component_counts = components.groupBy("component").count().orderBy(desc("count"))
component_counts.show()
# Get the component with the highest count
strongest_component_id = component_counts.first()["component"]

# Show the details of all the vertices in the largest component.
strongest_component_vertices = components.filter(components["component"] == strongest_component_id)
strongest_component_vertices.show()
# Take full screenshots of all the steps you took to achieve the requirement and the output, and add them to your word document. Add all the commands to the Python script. (9 marks)
