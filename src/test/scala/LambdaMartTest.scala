import java.io.{File, PrintWriter}
import java.util.Random
import java.util.function.Consumer

import org.apache.spark.sql.functions.{asc, desc}
import org.apache.spark.sql.{Row, SparkSession}
import org.scalatest.Matchers._
import org.scalatest.{FlatSpec, GivenWhenThen}

class LambdaMartTest extends FlatSpec with GivenWhenThen {

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

  private def runTest(spark: SparkSession) = {
    val trainDataFile = getTempFile("train")
    writeRandomData(trainDataFile, 42L)

    val validationDataFile = getTempFile("validation")
    writeRandomData(validationDataFile, 84L)

    val trainer = new LambdaMart(spark, trainDataFile.getAbsolutePath, validationDataFile.getAbsolutePath)

    val model = trainer.trainRankerModel()

    val modelPath = getTempFile("model")
    model.write.overwrite().save(modelPath.getAbsolutePath)

    val validation = LambdaMart.loadLibSVMDataset(spark, validationDataFile.getAbsolutePath, isValidation = true)
    model.transform(validation)
      .select("q", "d", "prediction")
      .orderBy(asc("q"), desc("prediction"))
      .toLocalIterator()
      .forEachRemaining(new Consumer[Row]{
        override def accept(row: Row): Unit = {
          val description = row.getAs[String]("d")
          val prediction = row.getAs[Double]("prediction")
          if (description.contains("# P")) {
            prediction should be > 0.0
          } else {
            prediction should be < 0.0
          }
        }
      })
    val importance = model.getFeatureImportances("split")
    importance(0) should be >= 2.0
    importance(1) should be <= 0.0
  }

  it should "work with several threads" in {
    // NOTE: works fine with 1 thread
    val spark = SparkSession.builder().appName("testApp").master("local[16]").getOrCreate()
    runTest(spark)
  }
}
