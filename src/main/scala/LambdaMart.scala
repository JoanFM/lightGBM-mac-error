import ciir.umass.edu.learning.DenseDataPoint
import com.microsoft.ml.spark.lightgbm.{LightGBMRanker, LightGBMRankerModel}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.functions.lit
import org.apache.spark.sql.{Column, Dataset, Row, SparkSession}

case class Point(l: Double, q: Int, f: Vector, d: String)

class LambdaMart(sparkSession: SparkSession, trainPath: String, validatePath: String) {

  import LambdaMart._
  private val gradientBoostedModel = new LightGBMRanker()
    .setFeaturesCol("f")
    .setLabelCol("l")
    .setGroupCol(queryColumn)
    .setValidationIndicatorCol("Validation")
    .setNumIterations(2)
    .setNumLeaves(2)

  private val train = loadLibSVMDataset(sparkSession, trainPath, isValidation = false)
  private val validation = loadLibSVMDataset(sparkSession, validatePath, isValidation = true)

  def trainRankerModel(): LightGBMRankerModel = {
    gradientBoostedModel.fit(train.union(validation))
  }
}

object LambdaMart {

  private final val queryColumn = "q"

  def parseLine(line: String): Point = {
    val dataPoint = new DenseDataPoint(line)
    val features = Array.fill[Float](dataPoint.getFeatureCount){0.0f}
    Array.copy(dataPoint.getFeatureVector, 1, features, 0, features.length)
    Point(dataPoint.getLabel, dataPoint.getID.toInt,
      Vectors.dense(features.map(_.toDouble)), dataPoint.getDescription)
  }

  def loadLibSVMDataset(sparkSession: SparkSession, path: String, isValidation: Boolean): Dataset[Row] = {
    import sparkSession.implicits._
    val rawData = sparkSession.read.text(path)
    val parsed = rawData.map(row => parseLine(row.getString(0)))
      .withColumn("Validation", lit(isValidation))
      .repartition(new Column(queryColumn))
    parsed
  }
}
