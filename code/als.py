import itertools
#import numpy as np
import pyspark
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, ArrayType, DoubleType
from pyspark.sql.functions import split
from pyspark.ml.feature import CountVectorizer
from pyspark.sql import functions as f
#from pyspark.ml.clustering import KMeans
from pyspark.ml.recommendation import ALS
#from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import isnan, when, count, col, udf
#import matplotlib.pyplot as plt


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


# read in the data files to spark datframes
# movies = spark.read.csv('../data/movies.dat', sep='::', header=False, schema=movieSchema)
# ratings = spark.read.csv('../data/ratings.dat', sep='::', header=False, schema=ratingSchema)
movies = spark.read.csv('../movies.dat', sep='::', header=False, schema=movieSchema)
ratings = spark.read.csv('../ratings.dat', sep='::', header=False, schema=ratingSchema)


# view some info about the data files
#movies.show(10)
#movies.printSchema()
#ratings.show(10)
#ratings.printSchema()


# split the ratings dataset into train-test-validate subsets
(train_ratings, test_ratings, validate_ratings) = ratings.randomSplit([0.35, 0.35, 0.3], seed=42)


# run ALS using algorithm defined at
# https://github.com/databricks/spark-training/blob/master/website/movie-recommendation-with-mllib.md
ranks = [8, 12]
lambdas = [0.1, 0.3, 0.5, 0.7, 0.9]
numIters = [10, 25]
bestModel = None
bestValidationRmse = float("inf")
bestRank = 0
bestLambda = -1.0
bestNumIter = -1

for rank, lmbda, numIter in itertools.product(ranks, lambdas, numIters):
    als = ALS(rank = rank, maxIter = numIter, regParam = lmbda, userCol = 'user_id', itemCol = 'movie_id', ratingCol = 'rating', nonnegative= False, implicitPrefs= False, coldStartStrategy='drop')
    model = als.fit(train_ratings)
    predictions = model.transform(validate_ratings)
    evaluator = RegressionEvaluator(metricName = 'rmse', labelCol = 'rating', predictionCol = 'prediction')
    rmse = evaluator.evaluate(predictions)

    #print ("RMSE (validation) = %f for the model trained with " % rmse + "rank = %d, lambda = %.1f, and numIter = %d." % (rank, lmbda, numIter))

    if (rmse < bestValidationRmse):
        bestModel = model
        bestValidationRmse = rmse
        bestRank = rank
        bestLambda = lmbda
        bestNumIter = numIter

predictions_test = bestModel.transform(test_ratings)
evaluator_test = RegressionEvaluator(metricName = 'rmse', labelCol = 'rating', predictionCol = 'prediction')
testRmse = evaluator.evaluate(predictions_test)

# evaluate the best model on the test set
print("\n\n\n")
print ("The best model was trained with rank = %d and lambda = %.1f, " % (bestRank, bestLambda) + "and numIter = %d, and its RMSE on the test set is %f." % (bestNumIter, testRmse))
print("\n\n\n")

# stop/close the connection
sc.stop()
spark.stop()
