package nm.handwritten.cnn

import java.awt.Color
import java.awt.image.BufferedImage
import java.io.File
import java.io.IOException
import javax.imageio.ImageIO

/**
 * Created by Tanmay and Gaurav
 */
object EdgeDetection {
    private val FILTER_VERTICAL =
        arrayOf(doubleArrayOf(1.0, 0.0, -1.0), doubleArrayOf(1.0, 0.0, -1.0), doubleArrayOf(1.0, 0.0, -1.0))
    private val FILTER_HORIZONTAL =
        arrayOf(doubleArrayOf(1.0, 1.0, 1.0), doubleArrayOf(0.0, 0.0, 0.0), doubleArrayOf(-1.0, -1.0, -1.0))
    private val FILTER_SOBEL =
        arrayOf(doubleArrayOf(1.0, 0.0, -1.0), doubleArrayOf(2.0, 0.0, -2.0), doubleArrayOf(1.0, 0.0, -1.0))
    private const val INPUT_IMAGE = "resources/smallGirl.png"
    private var count = 1
    @Throws(IOException::class)
    @JvmStatic
    fun main(args: Array<String>) {
        detectVerticalEdges()
        detectHorizontalEdges()
        detectSobelEdges()
    }

    @Throws(IOException::class)
    private fun detectSobelEdges() {
        val bufferedImage = ImageIO.read(File(INPUT_IMAGE))
        val image = transformImageToArray(bufferedImage)
        val finalConv = applyConvolution(bufferedImage.width, bufferedImage.height, image, FILTER_SOBEL)
        reCreateOriginalImageFromMatrix(bufferedImage, finalConv)
    }

    @Throws(IOException::class)
    private fun detectHorizontalEdges() {
        val bufferedImage = ImageIO.read(File(INPUT_IMAGE))
        val image = transformImageToArray(bufferedImage)
        val finalConv = applyConvolution(bufferedImage.width, bufferedImage.height, image, FILTER_HORIZONTAL)
        reCreateOriginalImageFromMatrix(bufferedImage, finalConv)
    }

    @Throws(IOException::class)
    private fun detectVerticalEdges() {
        val bufferedImage = ImageIO.read(File(INPUT_IMAGE))
        val image = transformImageToArray(bufferedImage)
        val finalConv = applyConvolution(bufferedImage.width, bufferedImage.height, image, FILTER_VERTICAL)
        reCreateOriginalImageFromMatrix(bufferedImage, finalConv)
    }

    private fun transformImageToArray(bufferedImage: BufferedImage): Array<Array<DoubleArray>> {
        val width = bufferedImage.width
        val height = bufferedImage.height
        return transformImageToArray(bufferedImage, width, height)
    }

    private fun applyConvolution(
        width: Int,
        height: Int,
        image: Array<Array<DoubleArray>>,
        filter: Array<DoubleArray>
    ): Array<DoubleArray> {
        val convolution = Convolution()
        val redConv = convolution.convolutionType2(image[0], height, width, filter, 3, 3, 1)
        val greenConv = convolution.convolutionType2(image[1], height, width, filter, 3, 3, 1)
        val blueConv = convolution.convolutionType2(image[2], height, width, filter, 3, 3, 1)
        val finalConv = Array(redConv.size) { DoubleArray(redConv[0].size) }
        for (i in redConv.indices) {
            for (j in redConv[i].indices) {
                finalConv[i][j] = redConv[i][j] + greenConv[i][j] + blueConv[i][j]
            }
        }
        return finalConv
    }

    private fun transformImageToArray(
        bufferedImage: BufferedImage,
        width: Int,
        height: Int
    ): Array<Array<DoubleArray>> {
        val image = Array(3) { Array(height) { DoubleArray(width) } }
        for (i in 0 until height) {
            for (j in 0 until width) {
                val color = Color(bufferedImage.getRGB(j, i))
                image[0][i][j] = color.red.toDouble()
                image[1][i][j] = color.green.toDouble()
                image[2][i][j] = color.blue.toDouble()
            }
        }
        return image
    }

    @Throws(IOException::class)
    private fun reCreateOriginalImageFromMatrix(originalImage: BufferedImage, imageRGB: Array<DoubleArray>) {
        val writeBackImage = BufferedImage(originalImage.width, originalImage.height, BufferedImage.TYPE_INT_RGB)
        for (i in imageRGB.indices) {
            for (j in imageRGB[i].indices) {
                val color = Color(
                    fixOutOfRangeRGBValues(imageRGB[i][j]),
                    fixOutOfRangeRGBValues(imageRGB[i][j]),
                    fixOutOfRangeRGBValues(imageRGB[i][j])
                )
                writeBackImage.setRGB(j, i, color.rgb)
            }
        }
        val outputFile = File("edges" + count++ + ".png")
        ImageIO.write(writeBackImage, "png", outputFile)
    }

    private fun fixOutOfRangeRGBValues(value: Double): Int {
        var value = value
        if (value < 0.0) {
            value = -value
        }
        return if (value > 255) {
            255
        } else {
            value.toInt()
        }
    }
}