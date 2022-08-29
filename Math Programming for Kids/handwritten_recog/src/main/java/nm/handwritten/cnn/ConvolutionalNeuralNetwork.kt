package nm.handwritten.cnn

import java.lang.Thread
import kotlin.Throws
import java.io.IOException
import kotlin.jvm.JvmStatic
import nm.handwritten.cnn.EdgeDetection
import java.awt.image.BufferedImage
import javax.imageio.ImageIO
import nm.handwritten.cnn.LenetMnistExample
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator
import org.deeplearning4j.nn.conf.MultiLayerConfiguration
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.LearningRatePolicy
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.util.ModelSerializer
import org.deeplearning4j.earlystopping.scorecalc.ScoreCalculator
import nm.handwritten.cnn.AccuracyCalculator
import nm.handwritten.cnn.ConvolutionalNeuralNetwork
import nm.handwritten.data.LabeledImage
import org.nd4j.linalg.factory.Nd4j
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration
import org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition
import org.deeplearning4j.earlystopping.termination.MaxTimeIterationTerminationCondition
import java.util.concurrent.TimeUnit
import org.deeplearning4j.earlystopping.saver.LocalFileModelSaver
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingTrainer
import org.deeplearning4j.earlystopping.EarlyStoppingResult
import org.deeplearning4j.nn.api.Model
import org.deeplearning4j.nn.conf.Updater
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer
import org.nd4j.linalg.activations.Activation
import org.slf4j.LoggerFactory
import java.io.File
import java.lang.Exception
import kotlin.Any as Any1


class ConvolutionalNeuralNetwork {
    private var preTrainedModel: MultiLayerNetwork? = null
    @Throws(IOException::class)
    fun init() {
        preTrainedModel = ModelSerializer.restoreMultiLayerNetwork(File(TRAINED_MODEL_FILE))
    }

    fun predict(labeledImage: LabeledImage): Int {
        val pixels = labeledImage.pixels
        if (pixels != null) {
            for (i in pixels.indices) {
                pixels[i] = pixels[i] / 255.0
            }
        }
        val predict = preTrainedModel!!.predict(Nd4j.create(pixels))
        return predict[0]
    }

    @Throws(IOException::class)
    fun train(trainDataSize: Int?, testDataSize: Int?) {
        val nChannels = 1 // Number of input channels
        val outputNum = 10 // The number of possible outcomes
        val batchSize = 64 // Test batch size
        val nEpochs = 20 // Number of training epochs
        val iterations = 1 // Number of training iterations
        val seed = 123 //
        val mnistTrain = MnistDataSetIterator(batchSize, trainDataSize!!, false, true, true, 12345)
        val conf = NeuralNetConfiguration.Builder()
            .seed(seed)
            .iterations(iterations)
            .regularization(false)
            .learningRate(0.01)
            .weightInit(WeightInit.XAVIER)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .updater(Updater.NESTEROVS)
            .list()
            .layer(
                0, ConvolutionLayer.Builder(5, 5)
                    .nIn(nChannels)
                    .stride(1, 1)
                    .nOut(20)
                    .activation(Activation.IDENTITY)
                    .build()
            )
            .layer(
                1, SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                    .kernelSize(2, 2)
                    .stride(2, 2)
                    .build()
            )
            .layer(
                2, ConvolutionLayer.Builder(5, 5)
                    .nIn(20)
                    .stride(1, 1)
                    .nOut(50)
                    .activation(Activation.IDENTITY)
                    .build()
            )
            .layer(
                3, SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                    .kernelSize(2, 2)
                    .stride(2, 2)
                    .build()
            )
            .layer(
                4, DenseLayer.Builder().activation(Activation.RELU)
                    .nIn(800)
                    .nOut(128).build()
            )
            .layer(
                5, DenseLayer.Builder().activation(Activation.RELU)
                    .nIn(128)
                    .nOut(64).build()
            )
            .layer(
                6, OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                    .nOut(outputNum)
                    .activation(Activation.SOFTMAX)
                    .build()
            )
            .setInputType(InputType.convolutionalFlat(28, 28, 1))
            .backprop(true).pretrain(false).build()
        val esConf: Unit = EarlyStoppingConfiguration.Builder<Model?>()
            .epochTerminationConditions(MaxEpochsTerminationCondition(nEpochs))
            .iterationTerminationConditions(MaxTimeIterationTerminationCondition(75, TimeUnit.MINUTES))
            .scoreCalculator(
                /* scoreCalculator = */ AccuracyCalculator(
                    MnistDataSetIterator(testDataSize!!, testDataSize, false, false, true, 12345)
                )
            )
            .evaluateEveryNEpochs(1)
            .modelSaver(LocalFileModelSaver(OUT_DIR))
            .build()
        val trainer = EarlyStoppingTrainer(esConf, conf, mnistTrain)
        val result: EarlyStoppingResult<*>

    }

    private fun EarlyStoppingTrainer(
        esConf: Unit,
        conf: MultiLayerConfiguration?,
        mnistTrain: MnistDataSetIterator
    ): kotlin.Any {
        TODO("Not yet implemented")
    }

    companion object {
        private const val OUT_DIR = "resources/cnnCurrentTrainingModels"
        private const val TRAINED_MODEL_FILE = "resources/cnnTrainedModels/bestModel.bin"
        private val LOG = LoggerFactory.getLogger(ConvolutionalNeuralNetwork::class.java)
        @Throws(Exception::class)
        @JvmStatic
        fun main(args: Array<String>) {
            ConvolutionalNeuralNetwork().train(60000, 1000)
        }
    }
}

private fun Model.fit() {

}

private fun Unit.build() {
}


private fun Unit.modelSaver(localFileModelSaver: LocalFileModelSaver) {

}

private fun Unit.evaluateEveryNEpochs(i: Int) {

}

private fun <T : Model?> EarlyStoppingConfiguration.Builder<T>.scoreCalculator(accuracyCalculator: AccuracyCalculator) {
}