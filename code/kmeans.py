#import itertools
import numpy as np
import pyspark
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, ArrayType, DoubleType
from pyspark.sql.functions import split
from pyspark.ml.feature import CountVectorizer
from pyspark.sql import functions as f
from pyspark.ml.clustering import KMeans
#from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import ClusteringEvaluator
#from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import isnan, when, count, col, udf
import matplotlib.pyplot as plt


# create the connection to the spark cluster for RDD and DataFrame
# RUN ONLY ONCE - MUST STOP sc AND/OR spark ONCE STARTED TO RUN THIS BLOCK AGAIN
sc = pyspark.SparkContext(master='local[*]', appName='CollabFilter')
spark = pyspark.sql.SparkSession(sc)


# create schemas for the data to be loaded from file using the GitHub README doc
movieSchema = StructType([
    StructField("movie_id", IntegerType(), True),
    StructField("movie_title", StringType(), True),
    StructField("genres", StringType(), True)])


ratingSchema = StructType([
    StructField("user_id", IntegerType(), True),
    StructField("movie_id", IntegerType(), True),
    StructField("rating", IntegerType(), True),
    StructField("timestamp_epoch", IntegerType(), True)])


userSchema = StructType([
    StructField("user_id", IntegerType(), True),
    StructField("gender", StringType(), True),
    StructField("age", StringType(), True),
    StructField("occupation", IntegerType(), True),
    StructField("zip_code", IntegerType(), True)])


# read in the data files to spark datframes
# movies = spark.read.csv('../data/movies.dat', sep='::', header=False, schema=movieSchema)
# ratings = spark.read.csv('../data/ratings.dat', sep='::', header=False, schema=ratingSchema)
# users = spark.read.csv('../data/users.dat', sep='::', header=False, schema=userSchema)
movies = spark.read.csv('../movies.dat', sep='::', header=False, schema=movieSchema)
ratings = spark.read.csv('../ratings.dat', sep='::', header=False, schema=ratingSchema)
users = spark.read.csv('../users.dat', sep='::', header=False, schema=userSchema)


# view some info about the data files
#movies.show(10)
#movies.printSchema()
#ratings.show(10)
#ratings.printSchema()
#users.show(10)
#users.printSchema()


# split the genres on '|' and create a list individual genres
movies = movies.withColumn("genres", split("genres", "\|"))


# use countVectorizer to encode/binarize the categorical genre feature
cv = CountVectorizer(binary=True)
cv.setInputCol("genres")
cv.setOutputCol("features")
model = cv.fit(movies)
model.setInputCol("genres")
movies_transformed = model.transform(movies)


# run kmeans for various values of k, i.e. number of clusters
# store the silhouette score in a list for comparison
list_silhouette_scores = []
for k in range(2, 21):
    # run KMeans clustering algorithm with 'k' clusters
    kmeans = KMeans().setK(k).setSeed(42)
    model = kmeans.fit(movies_transformed.select('features')).setPredictionCol('cluster')
    results_kmeans = model.transform(movies_transformed)
    # evaluate the results of KMeans using silhouette
    evaluator = ClusteringEvaluator(predictionCol='cluster')
    list_silhouette_scores.append((k, evaluator.evaluate(results_kmeans)))


# # print the silhouette plot to identify an appropriate 'k' value visually
# plt.figure(figsize=(20,5))
# plt.plot(*zip(*list_silhouette_scores))
# plt.title('Silhouette Plot for KMeans', fontsize=16)
# plt.xlabel('K', fontsize=12)
# plt.ylabel('Silhouette Measure Using\nSquared Euclidean Distance', fontsize=12)
# x_ticks = range(2,21,2)
# plt.xticks(x_ticks)
# plt.vlines(x=9, ymin=0.49, ymax=0.52, color='red', label='Chosen K')
# plt.legend(loc='upper center');


# run KMeans again for the chosen "best" K = 9
kmeans = KMeans().setK(9).setSeed(42)
model = kmeans.fit(movies_transformed.select('features')).setPredictionCol('cluster')
results_kmeans = model.transform(movies_transformed)


# join the raings and kmeans results into one dataframe
ratings = ratings.alias('ratings')
results_kmeans = results_kmeans.alias('results_kmeans')
list_columns = ['ratings.user_id', 'ratings.movie_id', 'ratings.rating', 'results_kmeans.movie_title', 'results_kmeans.genres', 'results_kmeans.cluster']
ratings_clusters = ratings.join(results_kmeans, ratings.movie_id == results_kmeans.movie_id).select(list_columns)


# group by cluster and user id to find the average rating for user and cluster
user_averages = ratings_clusters.groupby(['cluster', 'user_id']).agg({'rating':'avg'})
#user_averages.show(5)


# # identify counts of genres in each clusert for understanding of what the clusters represent
# cluster0 = results_kmeans.filter(results_kmeans.cluster == 0).withColumn("Cluster 0", f.explode("genres").alias("token")).groupBy("Cluster 0").count().orderBy(f.desc("count"))
# cluster1 = results_kmeans.filter(results_kmeans.cluster == 1).withColumn("Cluster 1", f.explode("genres").alias("token")).groupBy("Cluster 1").count().orderBy(f.desc("count"))
# cluster2 = results_kmeans.filter(results_kmeans.cluster == 2).withColumn("Cluster 2", f.explode("genres").alias("token")).groupBy("Cluster 2").count().orderBy(f.desc("count"))
# cluster3 = results_kmeans.filter(results_kmeans.cluster == 3).withColumn("Cluster 3", f.explode("genres").alias("token")).groupBy("Cluster 3").count().orderBy(f.desc("count"))
# cluster4 = results_kmeans.filter(results_kmeans.cluster == 4).withColumn("Cluster 4", f.explode("genres").alias("token")).groupBy("Cluster 4").count().orderBy(f.desc("count"))
# cluster5 = results_kmeans.filter(results_kmeans.cluster == 5).withColumn("Cluster 5", f.explode("genres").alias("token")).groupBy("Cluster 5").count().orderBy(f.desc("count"))
# cluster6 = results_kmeans.filter(results_kmeans.cluster == 6).withColumn("Cluster 6", f.explode("genres").alias("token")).groupBy("Cluster 6").count().orderBy(f.desc("count"))
# cluster7 = results_kmeans.filter(results_kmeans.cluster == 7).withColumn("Cluster 7", f.explode("genres").alias("token")).groupBy("Cluster 7").count().orderBy(f.desc("count"))
# cluster8 = results_kmeans.filter(results_kmeans.cluster == 8).withColumn("Cluster 8", f.explode("genres").alias("token")).groupBy("Cluster 8").count().orderBy(f.desc("count"))


# # print top two genres for each cluster
# # use this to create aliases for the clusters, e.g. action/thriller
# cluster0.show(2)
# cluster1.show(2)
# cluster2.show(2)
# cluster3.show(2)
# cluster4.show(2)
# cluster5.show(2)
# cluster6.show(2)
# cluster7.show(2)
# cluster8.show(2)


# sort the ratings_cluster df by user_id and movie_id
list_order = ['user_id', 'movie_id']
ratings_cluster = ratings_clusters.orderBy(list_order, ascending=True)
#ratings_cluster.show(10)




# create a df of distinct pairs of users and movies for creating the ratings matrix
users_distinct = ratings_cluster.select('user_id').distinct()
movies_distinct = ratings_cluster.select(['movie_id', 'cluster']).distinct()
matrix_indices = users_distinct.crossJoin(movies_distinct)
#matrix_indices.show(10)


# print('Number of distinct users:', users_distinct.count())
# print('Number of distinct movies:', movies_distinct.count())
# print('Length of matrix_indeices:', matrix_indices.count())


# create a ratings matrix containing user ratings, as well as predicted user ratings based on KMeans clustering
ratings_matrix = matrix_indices.join(ratings.select('user_id', 'movie_id', 'rating'), ['user_id', 'movie_id'], 'outer').join(user_averages.select('user_id', 'cluster', 'avg(rating)'), ['user_id', 'cluster'], 'outer')
#ratings_matrix.show(10)


# filter the ratings matrix to only those records with actual ratings
# this is needed to calculate RMSE
evaluate_kmeans = ratings_matrix.filter(ratings_matrix.rating.isNotNull())


# calculate the RMSE of the KMeans model
evaluate_kmeans = evaluate_kmeans.withColumn('square_diff', f.pow(f.col('rating') - f.col('avg(rating)'), 2))
rmse = evaluate_kmeans.select(f.sqrt(f.avg(f.col('square_diff'))).alias('rmse'))


# print the RMSE of the best KMeans model
print('\n\n\n')
print(rmse.show())
print('\n\n\n')


# stop/close the connection
sc.stop()
spark.stop()
