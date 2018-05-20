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
#import classifier_2 as s

class VoteClassifier(ClassifierI):
    #global _init_
    #global classify
    
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, transformer):
        votes = []
        for c in self._classifiers:
            v = c.predict(transformer)
            votes.append(v)
        return mode(votes)

    #def confidence(self, transformer):
       # votes = []
        #for c in self._classifiers:
        #    v = c.predict(transformer)
        #    votes.append(v)

        #choice_votes = votes.count(mode(votes))
        #conf = choice_votes / len(votes)
        #return conf



#spark = SparkSession.builder \
#    .master("yarn") \
#    .appName("AspectDetector") \
#    .enableHiveSupport() \
#    .getOrCreate()
conf = SparkConf()
conf.setAppName("TA") 
sc = SparkContext(conf=conf)
#sqlContext = SQLContext(spark)
#sc = SparkContext("local[2]", "TA")
ssc = StreamingContext(sc,10)
#sqlContext = SQLContext(sc)
#hiveContext = HiveContext(sc)
htf = HashingTF(50000)

NB_output_dir = 'hdfs://master:9000/user/hadoop/NaiveBayes'
NB_load_model= NaiveBayesModel.load(sc, NB_output_dir)

LR_output_dir = 'hdfs://master:9000/user/hadoop/LogisticRegression'
LR_load_model= LogisticRegressionModel.load(sc, LR_output_dir)

#SVM_output_dir = 'hdfs://master:9000/user/hadoop/SVM'
#SVM_load_model= SVMModel.load(sc, SVM_output_dir)

DT_output_dir = 'hdfs://master:9000/user/hadoop/DT'
DT_load_model= DecisionTreeModel.load(sc, DT_output_dir)

#a = sc.broadcast(NB_output_dir)
#b = sc.broadcast(LR_output_dir)
#c = sc.broadcast(DT_output_dir)
voted_classifier = VoteClassifier(NB_load_model, LR_load_model, DT_load_model)
#p = voted_classifier.classify()
#broadcast_var = sc.broadcast(voted_classifier)
#voted_classifier= VoteClassifier(a,b,c)
def sentiment(test_sample):
    #t = test_sample
    test_sample_sp = test_sample.split(" ")
    trans = htf.transform(test_sample_sp)
    return voted_classifier.classify(trans)
    #return [(t, voted_classifier.classify(trans))]
    #return broadcast_var.classify(trans)
#count_positive = 0
#count_negative = 0
#htf = HashingTF(50000)

#def senti(test_sample):
    #test_sample_sp = test_sample.split(" ")
    #NB_result = NB_load_model.predict(htf.transform(test_sample_sp))
    #LR_result = LR_load_model.predict(htf.transform(test_sample_sp))
    #DT_result = DT_load_model.predict(htf.transform(test_sample_sp))
    #if NB_result == 1:
      #count_positive = count_positive + 1
    #else:
     # count_negative = count_negative + 1

    #if LR_result == 1:
      #count_positive = count_positive + 1
    #else:
     # count_negative = count_negative + 1

    #if DT_result == 1:
      #count_positive = count_positive + 1
    #else:
      #count_negative = count_negative + 1
    
    #if count_positive > count_negative:
     # return 1
      #count_positive = 0
    #else:
      #return 0
      #count_negative = 0
lines = ssc.socketTextStream(socket.gethostbyname(socket.gethostname()), 10000)
#lines = lines2.window(20)
#lines_collect = lines.collect()
#answer = lines_collect.map(lambda x : sentiment(x))
#ans_rdd = sc.parallelize(answer)
lines.pprint()
tweets = lines.flatMap(lambda text :[(text)])
#tweets1 = lines.flatMap(lambda text :[(text)])
#tweets0= lines.flatMap(lambda rdd :[senti(rdd)])
#tweets10 = tweets_old.zip(tweets_new)
#tweets = lines.zip(ans_rdd)
tweets.pprint()
#tweets1.pprint()
#tweet0.pprint()
#new_rdd = rdd.flatMap(lambda x : [sentiment(x)])
#fields = ("text", "sentiment")
#Tweet = namedtuple('Tweet', fields)
#lines.flatMap(lambda text : (text, 1)).map(lambda rec : Tweet (rec[0], rec[1])).foreachRDD(lambda rdd : rdd.toDF().registerTempTable("flight"))
#p = l.collect()
#.toPandas().to_csv('mycsv.csv'))
#lines.saveAsTextFiles()
#Person = Row('name', 'age')
#person = lines1.map(lambda r: Person(*r))
#person.foreachRDD(lambda rdd : sqlContext.createDataFrame(rdd))
#l = df2.collect()
#print l
#sqlContext.createDataFrame(l, ['name', 'age']).collect()
#t = sqlContext.sql('select * from flight')
#t.show()
#ssc.start()
#ssc.awaitTermination()
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
        #print("----------- %s -----------" % str(time))
        try:
        #  new_rdd = rdd.flatMap(lambda x : [(x,senti(x))])
        # Get spark sql singleton context from the current context
          sql_context = get_sql_context_instance(rdd.context)
        # convert the RDD to Row RDD
          row_rdd = rdd.map(lambda w: Row(text=w[0], senti=w[1]))
        # create a DF from the Row RDD
          #row_rdd.pprint()
          hashtags_df = sql_context.createDataFrame(row_rdd)
          print hashtags_df.collect()
        # Register the dataframe as table
          hashtags_df.registerTempTable("hashtags")
        # get the top 10 hashtags from the table using SQL and print them
          hashtag_counts_df = sql_context.sql("select text, senti from hashtags")
          #hashtags_df.write.csv('hdfs://master:9000/user/hadoop/mycsv.csv')
          hashtag_counts_df.show()
          #hashtag_counts_df.write.json("hdfs://mater:9000/user/hadoop")
        # call this method to prepare top 10 hashtags DF and send them
        #send_df_to_dashboard(hashtag_counts_df)
        except:
          e = sys.exc_info()[0]
          print("Error: %s" % e)
tweets.foreachRDD(lambda r : s(r))
#tweets.foreachRDD(lambda r : process_rdd(r))
ssc.start()
ssc.awaitTermination()
