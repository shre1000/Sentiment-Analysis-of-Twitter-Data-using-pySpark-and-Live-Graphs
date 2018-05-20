import json
from pyspark import SparkContext, SparkConf
from pyspark.streaming import StreamingContext
import socket
from pyspark.sql import SQLContext 
from pyspark.sql import Row
import sys
import requests
import shutil
import nltk
from nltk.tokenize import word_tokenize
from pyspark.mllib.feature import HashingTF, IDF
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import NaiveBayes, NaiveBayesModel
from pyspark.mllib.classification import LogisticRegressionWithLBFGS, LogisticRegressionModel
from pyspark.mllib.classification import SVMWithSGD, SVMModel
from pyspark.mllib.tree import DecisionTree, DecisionTreeModel
from nltk.classify import ClassifierI
from statistics import mode


class VoteClassifier(ClassifierI):
    
    
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, transformer):
        votes = []
        for c in self._classifiers:
            v = c.predict(transformer)
            votes.append(v)
        return mode(votes)

conf = SparkConf()
conf.setAppName("TA") 
sc = SparkContext(conf=conf)
ssc = StreamingContext(sc,10)
htf = HashingTF(50000)

NB_output_dir = 'hdfs://master:9000/user/hadoop/NaiveBayes'
NB_load_model= NaiveBayesModel.load(sc, NB_output_dir)

LR_output_dir = 'hdfs://master:9000/user/hadoop/LogisticRegression'
LR_load_model= LogisticRegressionModel.load(sc, LR_output_dir)

DT_output_dir = 'hdfs://master:9000/user/hadoop/DT'
DT_load_model= DecisionTreeModel.load(sc, DT_output_dir)


voted_classifier = VoteClassifier(NB_load_model, LR_load_model, DT_load_model)

def sentiment(test_sample):
    
    test_sample_sp = test_sample.split(" ")
    trans = htf.transform(test_sample_sp)
    return voted_classifier.classify(trans)
   

lines = ssc.socketTextStream(socket.gethostbyname(socket.gethostname()), 10000)
lines.pprint()
tweets = lines.flatMap(lambda text :[(text)])
tweets.pprint()

def s(rdd):
  r3 = rdd.collect()
  r1 = map(lambda f :(f,sentiment(f)), r3)
  r5 = sc.parallelize(r1)
  process_rdd(r5) 
    
def get_sql_context_instance(spark_context):
        if ('sqlContextSingletonInstance' not in globals()):
           globals()['sqlContextSingletonInstance'] = SQLContext(spark_context)
        return globals()['sqlContextSingletonInstance']

    def process_rdd(rdd):
        
        try:
        
          sql_context = get_sql_context_instance(rdd.context)
        
          row_rdd = rdd.map(lambda w: Row(text=w[0], senti=w[1]))
       
          
          hashtags_df = sql_context.createDataFrame(row_rdd)
          print hashtags_df.collect()
       
          hashtags_df.registerTempTable("hashtags")

          hashtag_counts_df = sql_context.sql("select text, senti from hashtags")
          
          hashtag_counts_df.show()
          
        except:
          e = sys.exc_info()[0]
          print("Error: %s" % e)
            
tweets.foreachRDD(lambda r : s(r))
ssc.start()
ssc.awaitTermination()
