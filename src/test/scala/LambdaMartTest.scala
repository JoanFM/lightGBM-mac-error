import java.io.{File, PrintWriter}
import java.util.Random
import java.util.function.Consumer

import com.holdenkarau.spark.testing.DataFrameSuiteBase
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.Row
import org.apache.spark.sql.functions.{asc, desc}
import org.scalatest.Matchers._
import org.scalatest.{FlatSpec, GivenWhenThen}

class LambdaMartTest extends FlatSpec with GivenWhenThen with DataFrameSuiteBase {

  private def writeRandomData(dataFile: File, seed: Long) {

    // feature 1 is the only good one
    val out = new PrintWriter(dataFile)
    val rand = new Random(seed)

    for (i <- 1 to 3) {
      for (j <- 1 to 100) {
        val w1 = if (rand.nextBoolean()) "-1.0" else "1.0"
        val w2 = if (rand.nextBoolean()) "-1.0" else "1.0"
        out.println("1 qid:" + Integer.toString(i) + " 1:1.0 2:" + w1 + " # P" + Integer.toString(j))
        out.println("0 qid:" + Integer.toString(i) + " 1:0.9 2:" + w2 + " # N" + Integer.toString(j))
      }
    }
    out.close()
  }

  private def getTempFile(prefix: String): File = {
    val file = File.createTempFile(prefix, ".tmp")
    file.deleteOnExit()
    file
  }

  "LambdaMart" should "separate noisy data" in {
    spark.sparkContext.setLogLevel("INFO")
    Logger.getLogger("logger").setLevel(Level.DEBUG)

    Given("train data file")
    val trainDataFile = getTempFile("train")
    writeRandomData(trainDataFile, 42L)

    And("validation data file")
    val validationDataFile = getTempFile("validation")
    writeRandomData(validationDataFile, 84L)

    And("Config")
    val debugVerbosity = 2
    val config = new LambdaMartConfig(trainDataFile.getAbsolutePath,
      validationDataFile.getAbsolutePath,
      featuresColumn = "Features",
      labelColumn = "Label",
      evalAt = 100,
      numIterations = 2,
      numLeaves = 2,
      learningRate = 0.05,
      earlyStoppingRound = 20,
      treeParallelism = "voting_parallel",
      verbosity = debugVerbosity,
      modelSavePath = ""
    )
    val trainer = new LambdaMart(config, spark)

    When("training is done")
    val model = trainer.trainRankerModel()

    Then("The model correctly ranks documents using first feature")
    val modelPath = getTempFile("model")
    model.write.overwrite().save(modelPath.getAbsolutePath)

    val validation = LambdaMart.loadLibSVMDataset(spark, config.validatePath, isValidation = true)
    model.transform(validation)
      .select("QID", "DocID", "prediction")
      .orderBy(asc("QID"), desc("prediction"))
      .toLocalIterator()
      .forEachRemaining(new Consumer[Row]{
        override def accept(row: Row): Unit = {
          val docId = row.getAs[String]("DocID")
          val score = row.getAs[Double]("prediction")
          if (docId.charAt(0) == 'P') {
            score should be > 0.0
          } else {
            score should be < 0.0
          }
        }
     })
    val importance = model.getFeatureImportances("split")
    importance(0) should be >= 2.0
    importance(1) should be <= 0.0
  }
}
