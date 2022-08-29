package nm.handwritten.data

import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.linalg.Vectors
import java.io.Serializable


class LabeledImage(label: Int, pixels: DoubleArray) : Serializable {
    private val meanNormalizedPixel: DoubleArray
    val pixels: DoubleArray
    var label: Double
    val features: Vector
    private fun meanNormalizeFeatures(pixels: DoubleArray): DoubleArray {
        var min = Double.MAX_VALUE
        var max = Double.MIN_VALUE
        var sum = 0.0
        for (pixel in pixels) {
            sum = sum + pixel
            if (pixel > max) {
                max = pixel
            }
            if (pixel < min) {
                min = pixel
            }
        }
        val mean = sum / pixels.size
        val pixelsNorm = DoubleArray(pixels.size)
        for (i in pixels.indices) {
            pixelsNorm[i] = (pixels[i] - mean) / (max - min)
        }
        return pixelsNorm
    }

    override fun toString(): String {
        return "LabeledImage{" +
                "label=" + label +
                '}'
    }

    init {
        meanNormalizedPixel = meanNormalizeFeatures(pixels)
        this.pixels = pixels
        features = Vectors.dense(meanNormalizedPixel)
        this.label = label.toDouble()
    }
}