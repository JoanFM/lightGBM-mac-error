import java.util.StringTokenizer

import com.microsoft.ml.spark.lightgbm.{LightGBMRanker, LightGBMRankerModel}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.functions.lit
import org.apache.spark.sql.{Column, Dataset, Row, SparkSession}

import scala.collection.mutable

case class RankedPoint(Label: Double, QID: Int, Features: Vector, DocID: String)

class LambdaMart(config: LambdaMartConfig, sparkSession: SparkSession) {

  import LambdaMart._
  private val gradientBoostedModel = new LightGBMRanker()
  gradientBoostedModel
    .setFeaturesCol(config.featuresColumn)
    .setLabelCol(config.labelColumn)
    .setGroupCol(qidColumn)
    .setValidationIndicatorCol("Validation")
  gradientBoostedModel
    .setNumIterations(config.numIterations)
    .setNumLeaves(config.numLeaves)
    .setLearningRate(config.learningRate)
    .setMaxPosition(config.evalAt)
    .setEarlyStoppingRound(config.earlyStoppingRound)
  gradientBoostedModel.setParallelism(config.treeParallelism)
  gradientBoostedModel.setVerbosity(config.verbosity)

  val train = loadLibSVMDataset(sparkSession, config.trainPath, isValidation = false)
  val validation = loadLibSVMDataset(sparkSession, config.validatePath, isValidation = true)

  /**
   *
   * @return Trained model
   */
  def trainRankerModel(): LightGBMRankerModel = {
    gradientBoostedModel.fit(train.union(validation))
  }
}

object LambdaMart {

  private final val qidColumn = "QID"

  def parseLine(line: String): RankedPoint = {
    val st = new StringTokenizer(line, " ")
    val label = st.nextToken.toDouble
    val qid   = st.nextToken.replace("qid:", "").toInt
    val features = mutable.ListBuffer[Double]()

    var featureString = st.nextToken
    do {
      val splitIndex = featureString.indexOf(":") + 1
      features += featureString.substring(splitIndex).toDouble
      featureString = st.nextToken
    } while (featureString != "#")
    val doc = st.nextToken
    RankedPoint(label, qid, Vectors.dense(features.toArray), doc)
  }

  def loadLibSVMDataset(sparkSession: SparkSession, path: String, isValidation: Boolean): Dataset[Row] = {
    import sparkSession.implicits._
    val rawData = sparkSession.read.text(path)
    val parsed = rawData.map(row => parseLine(row.getString(0)))
      .withColumn("Validation", lit(isValidation))
      .repartition(new Column(qidColumn))
    parsed
  }
}
