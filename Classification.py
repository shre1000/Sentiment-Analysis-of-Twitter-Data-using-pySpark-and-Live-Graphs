import sys
import shutil
import nltk
from pyspark import SparkConf, SparkContext
from nltk.tokenize import word_tokenize
from pyspark.mllib.feature import HashingTF, IDF
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import NaiveBayes, NaiveBayesModel
from pyspark.mllib.classification import LogisticRegressionWithLBFGS, LogisticRegressionModel
from pyspark.mllib.classification import SVMWithSGD, SVMModel
from pyspark.mllib.tree import DecisionTree, DecisionTreeModel

if __name__ == "__main__":

  conf = SparkConf()
  conf.setAppName("SentimentAnalysis")
  sc = SparkContext(conf=conf)

  pos = sc.textFile("hdfs://master:9000/user/hadoop/pos.txt")
  neg = sc.textFile("hdfs://master:9000/user/hadoop/neg.txt")

  pos_sp = pos.flatMap(lambda line: line.split("\n")).collect()
  neg_sp = neg.flatMap(lambda line: line.split("\n")).collect()

  all_words = []
  documents = []
  allowed = ["J", "R", "V", "N"]

  for p in pos_sp:
    documents.append({"text": p , "label": 1})

  for p in neg_sp:
    documents.append({"text": p , "label": 0})

  def wc(data):
    words = word_tokenize(data)
    tag = nltk.pos_tag(words)
    for w in tag:
      if w[1][0] in allowed:
         all_words.append(w[0].lower())
    return all_words

  raw_data = sc.parallelize(documents)
  raw_tokenized = raw_data.map(lambda dic : {"text": wc(dic["text"]) , "label" : dic["label"]})

  htf = HashingTF(50000)
  raw_hashed = raw_tokenized.map(lambda dic : LabeledPoint(dic["label"], htf.transform(dic["text"])))
  raw_hashed.persist()

  trained_hashed, test_hashed = raw_hashed.randomSplit([0.7, 0.3])

  NB_model = NaiveBayes.train(trained_hashed)
  NB_prediction_and_labels = test_hashed.map(lambda point : (NB_model.predict(point.features), point.label))
  NB_correct = NB_prediction_and_labels.filter(lambda (predicted, actual): predicted == actual)
  NB_accuracy = NB_correct.count() / float(test_hashed.count())
  print "NB training accuracy:" + str(NB_accuracy * 100) + " %"
  NB_output_dir = 'hdfs://master:9000/user/hadoop/NaiveBayes'
  shutil.rmtree("hdfs://master:9000/user/hadoop/NaiveBayes/metadata", ignore_errors=True)
  NB_model.save(sc, NB_output_dir)
  
  
  LR_model = LogisticRegressionWithLBFGS.train(trained_hashed)
  LR_prediction_and_labels = test_hashed.map(lambda point : (LR_model.predict(point.features), point.label))
  LR_correct = LR_prediction_and_labels.filter(lambda (predicted, actual): predicted == actual)
  LR_accuracy = LR_correct.count() / float(test_hashed.count())
  print "LR training accuracy:" + str(LR_accuracy * 100) + " %"
  LR_output_dir = 'hdfs://master:9000/user/hadoop/LogisticRegression'
  shutil.rmtree("hdfs://master:9000/user/hadoop/LogisticRegression/metadata", ignore_errors=True)
  LR_model.save(sc, LR_output_dir)

  SVM_model = SVMWithSGD.train(trained_hashed, iterations = 10)
  SVM_prediction_and_labels = test_hashed.map(lambda point : (SVM_model.predict(point.features), point.label))
  SVM_model.clearThreshold()
  SVM_correct = SVM_prediction_and_labels.filter(lambda (predicted, actual): predicted == actual)
  SVM_accuracy = SVM_correct.count() / float(test_hashed.count())
  print "SVM training accuracy:" + str(SVM_accuracy * 100) + " %"
  SVM_output_dir = 'hdfs://master:9000/user/hadoop/SVM'
  shutil.rmtree("hdfs://master:9000/user/hadoop/SVM/metadata", ignore_errors=True)
  SVM_model.save(sc, SVM_output_dir)

  model = DecisionTree.trainClassifier(trained_hashed, numClasses=2, categoricalFeaturesInfo={},
                                     impurity='gini', maxDepth=5, maxBins=32)
  predictions = model.predict(test_hashed.map(lambda x: x.features))
  labelsAndPredictions = test_hashed.map(lambda lp: lp.label).zip(predictions)
  testErr = labelsAndPredictions.filter(
    lambda lp: lp[0] != lp[1]).count() / float(test_hashed.count())
  print('Test Error = ' + str(testErr))
  print('Learned classification tree model:')
  print(model.toDebugString())
  model.save(sc, "hdfs:///user/hadoop/DT")
  
