package nm.handwritten.nn

import org.apache.spark.ml.classification.MultilayerPerceptronClassificationModel
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.Row
import org.apache.spark.sql.SparkSession
import org.slf4j.LoggerFactory
import nm.handwritten.data.IdxReader
import nm.handwritten.data.LabeledImage


class NeuralNetwork {
    private var sparkSession: SparkSession? = null
    private var model: MultilayerPerceptronClassificationModel? = null
    fun init() {
        initSparkSession()
        if (model == null) {
            LOGGER.info("Loading the Neural Network from saved model ... ")
            model = MultilayerPerceptronClassificationModel.load("resources/nnTrainedModels/ModelWith60000")
            LOGGER.info("Loading from saved model is done")
        }
    }

    fun train(trainData: Int?, testFieldValue: Int?) {
        initSparkSession()
        val labeledImages = IdxReader.loadData(trainData!!)
        val testLabeledImages = IdxReader.loadTestData(testFieldValue!!)
        val train = sparkSession!!.createDataFrame(labeledImages, LabeledImage::class.java).checkpoint()
        val test = sparkSession!!.createDataFrame(testLabeledImages, LabeledImage::class.java).checkpoint()
        val layers = intArrayOf(784, 128, 64, 10)
        val trainer = MultilayerPerceptronClassifier()
            .setLayers(layers)
            .setBlockSize(128)
            .setSeed(1234L)
            .setMaxIter(100)
        model = trainer.fit(train)
        evalOnTest(test)
        evalOnTest(train)
    }

    private fun evalOnTest(test: Dataset<Row>) {
        val result = model!!.transform(test)
        val predictionAndLabels = result.select("prediction", "label")
        val evaluator = MulticlassClassificationEvaluator()
            .setMetricName("accuracy")
        LOGGER.info("Test set accuracy = " + evaluator.evaluate(predictionAndLabels))
    }

    private fun initSparkSession() {
        if (sparkSession == null) {
            sparkSession = SparkSession.builder()
                .master("local[*]")
                .appName("Digit Recognizer")
                .orCreate
        }
        sparkSession!!.sparkContext().setCheckpointDir("checkPoint")
    }

    fun predict(labeledImage: LabeledImage): LabeledImage {
        val predict = model!!.predict(labeledImage.features)
        labeledImage.label = predict
        return labeledImage
    }

    companion object {
        private val LOGGER = LoggerFactory.getLogger(NeuralNetwork::class.java)
    }
}