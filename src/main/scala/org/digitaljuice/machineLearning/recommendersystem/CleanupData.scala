package org.digitaljuice.itemrecommender

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.recommendation.ALS
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel
import org.apache.spark.mllib.recommendation.Rating
import org.apache.spark.sql.SparkSession
import org.apache.spark.SparkContext._
import it.nerdammer.spark.hbase._
object RecommenderTrainer {
  def main( args: Array[String] ){
    val dldDataPath = "hdfs:///user/DownloadsTracker2"          
    val conf = new SparkConf().setAppName("Model Trainer for Recommender Ending")
    conf.set("spark.hbase.host", "10.60.0.155")
    val sc = new SparkContext(conf)

    val userDownloadData = sc.textFile(dldDataPath)     
    
    val ratings = userDownloadData.map(_.split(',') match { case Array(user, item, rate) =>
     Rating(user.toInt, item.toInt, rate.toDouble)
    })
    
    val ranks = List(8, 12)
    val lambdas = List(1.0, 10.0)
    val numIters = List(10, 20)
    var bestModel: Option[MatrixFactorizationModel] = None
    var bestValidationRmse = Double.MaxValue
    var bestRank = 0
    var bestLambda = -1.0
    var bestNumIter = -1
    val alpha = 0.01

    for (rank <- ranks; lambda <- lambdas; numIter <- numIters) {
      val model = ALS.trainImplicit(ratings, rank, numIter, lambda, alpha)
      val validationRmse = computeRmse(model, validation, numValidation)
      println("RMSE (validation) = " + validationRmse + " for the model trained with rank = "
        + rank + ", lambda = " + lambda + ", and numIter = " + numIter + ".")
      if (validationRmse < bestValidationRmse) {
        bestModel = Some(model)
        bestValidationRmse = validationRmse
        bestRank = rank
        bestLambda = lambda
        bestNumIter = numIter
      }
    }

    // Evaluate the model on rating data
    val usersProducts = ratings.map { case Rating(user, product, rate) =>
      (user, product)
    }
    
    val predictions =
       model.predict(usersProducts).map { case Rating(user, product, rate) =>
        ((user, product), rate)
      }

    val ratesAndPreds = ratings.map { case Rating(user, product, rate) =>
       ((user, product), rate)
    }.join(predictions)

    val MSE = ratesAndPreds.map { case ((user, product), (r1, r2)) =>
      val err = (r1 - r2)
      err * err
    }.mean()

    //println("Mean Squared Error = " + MSE)

    // Save and load model
    model.save(sc, "hdfs:///user/myCollaborativeFilter")
    //val sameModel = MatrixFactorizationModel.load(sc, "target/tmp/myCollaborativeFilter")

   }
}
