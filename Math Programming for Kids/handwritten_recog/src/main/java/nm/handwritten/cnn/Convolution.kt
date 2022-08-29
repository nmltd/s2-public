package nm.handwritten.cnn

import java.awt.Color
import java.lang.Thread

/**

 * @author: Tanmay and Gaurav
 */
class Convolution

    : Thread() {

    fun convolutionType1(
        input: Array<DoubleArray>,
        width: Int, height: Int,
        kernel: Array<DoubleArray>,
        kernelWidth: Int, kernelHeight: Int,
        iterations: Int
    ): Array<DoubleArray> {
        var width = width
        var height = height
        var newInput = input.clone()
        var output = input.clone()
        for (i in 0 until iterations) {
            val smallWidth = width - kernelWidth + 1
            val smallHeight = height - kernelHeight + 1
            output = convolution2D(newInput, width, height,
                kernel, kernelWidth, kernelHeight)
            width = smallWidth
            height = smallHeight
            newInput = output.clone()
        }
        return output
    }


    fun convolutionType2(
        input: Array<DoubleArray>,
        width: Int, height: Int,
        kernel: Array<DoubleArray>,
        kernelWidth: Int, kernelHeight: Int,
        iterations: Int
    ): Array<DoubleArray> {
        var newInput = input.clone()
        var output = input.clone()
        for (i in 0 until iterations) {
            output = Array(width) { DoubleArray(height) }
            //System.out.println("Iter: "+i+" conIN(50,50): "+newInput[50][50]);
            output = convolution2DPadded(newInput, width, height,
                kernel, kernelWidth, kernelHeight)
            //System.out.println("conOUT(50,50): "+output[50][50]);
            newInput = output.clone()
        }
        return output
    }

    companion object {

        fun singlePixelConvolution(
            input: Array<DoubleArray>,
            x: Int, y: Int,
            k: Array<DoubleArray>,
            kernelWidth: Int,
            kernelHeight: Int
        ): Double {
            var output = 0.0
            for (i in 0 until kernelWidth) {
                for (j in 0 until kernelHeight) {
                    output = output + input[x + i][y + j] * k[i][j]
                }
            }
            return output
        }

        fun applyConvolution(
            input: Array<IntArray>,
            x: Int, y: Int,
            k: Array<DoubleArray>,
            kernelWidth: Int,
            kernelHeight: Int
        ): Int {
            var output = 0
            for (i in 0 until kernelWidth) {
                for (j in 0 until kernelHeight) {
                    output = output + Math.round(input[x + i][y + j] * k[i][j]).toInt()
                }
            }
            return output
        }


        fun convolution2D(
            input: Array<DoubleArray>,
            width: Int, height: Int,
            kernel: Array<DoubleArray>,
            kernelWidth: Int,
            kernelHeight: Int
        ): Array<DoubleArray> {
            val smallWidth = width - kernelWidth + 1
            val smallHeight = height - kernelHeight + 1
            val output = Array(smallWidth) { DoubleArray(smallHeight) }
            for (i in 0 until smallWidth) {
                for (j in 0 until smallHeight) {
                    output[i][j] = 0.0
                }
            }
            for (i in 0 until smallWidth) {
                for (j in 0 until smallHeight) {
                    output[i][j] = singlePixelConvolution(input, i, j, kernel,
                        kernelWidth, kernelHeight)
                    //if (i==32- kernelWidth + 1 && j==100- kernelHeight + 1) System.out.println("Convolve2D: "+output[i][j]);
                }
            }
            return output
        }


        fun convolution2DPadded(
            input: Array<DoubleArray>,
            width: Int, height: Int,
            kernel: Array<DoubleArray>,
            kernelWidth: Int,
            kernelHeight: Int
        ): Array<DoubleArray> {
            val smallWidth = width - kernelWidth + 1
            val smallHeight = height - kernelHeight + 1
            val top = kernelHeight / 2
            val left = kernelWidth / 2
            var small = Array(smallWidth) { DoubleArray(smallHeight) }
            small = convolution2D(input, width, height,
                kernel, kernelWidth, kernelHeight)
            val large = Array(width) { DoubleArray(height) }
            for (j in 0 until height) {
                for (i in 0 until width) {
                    large[i][j] = 0.0
                }
            }
            for (j in 0 until smallHeight) {
                for (i in 0 until smallWidth) {
//if (i+left==32 && j+top==100) System.out.println("Convolve2DP: "+small[i][j]);
                    large[i + left][j + top] = small[i][j]
                }
            }
            return large
        }


        fun convolutionDouble(
            input: Array<DoubleArray>,
            width: Int, height: Int,
            kernel: Array<DoubleArray>,
            kernelWidth: Int, kernelHeight: Int
        ): DoubleArray {
            val smallWidth = width - kernelWidth + 1
            val smallHeight = height - kernelHeight + 1
            var small = Array(smallWidth) { DoubleArray(smallHeight) }
            small = convolution2D(input, width, height, kernel, kernelWidth, kernelHeight)
            val result = DoubleArray(smallWidth * smallHeight)
            for (j in 0 until smallHeight) {
                for (i in 0 until smallWidth) {
                    result[j * smallWidth + i] = small[i][j]
                }
            }
            return result
        }


        fun convolutionDoublePadded(
            input: Array<DoubleArray>,
            width: Int, height: Int,
            kernel: Array<DoubleArray>,
            kernelWidth: Int,
            kernelHeight: Int
        ): DoubleArray {
            var result2D = Array(width) { DoubleArray(height) }
            result2D = convolution2DPadded(input, width, height,
                kernel, kernelWidth, kernelHeight)
            val result = DoubleArray(width * height)
            for (j in 0 until height) {
                for (i in 0 until width) {
                    result[j * width + i] = result2D[i][j]
                    //if (i==32 && j==100) System.out.println("ConvolveDP: "+result[j*width +i]+" "+result2D[i][j]);
                }
            }
            return result
        }


        fun doublesToValidPixels(greys: DoubleArray): IntArray {
            val result = IntArray(greys.size)
            var grey: Int
            for (i in greys.indices) {
                grey = if (greys[i] > 255) {
                    255
                } else if (greys[i] < 0) {
                    0
                } else {
                    Math.round(greys[i]).toInt()
                }
                result[i] = Color(grey, grey, grey).rgb
            }
            return result
        }


        fun convolution_image(
            input: IntArray, width: Int, height: Int,
            kernel: Array<DoubleArray>,
            kernelWidth: Int, kernelHeight: Int,
            scale: Double, offset: Double
        ): IntArray {
            val input2D = Array(width) { DoubleArray(height) }
            var output = DoubleArray(width * height)
            for (j in 0 until height) {
                for (i in 0 until width) {
                    input2D[i][j] = Color(input[j * width + i]).red.toDouble()
                }
            }
            output = convolutionDoublePadded(input2D, width, height,
                kernel, kernelWidth, kernelHeight)
            val outputInts = IntArray(width * height)
            for (i in outputInts.indices) {
                outputInts[i] = Math.round(output[i] * scale + offset).toInt()
                if (outputInts[i] > 255) outputInts[i] = 255
                if (outputInts[i] < 0) outputInts[i] = 0
                val g = outputInts[i]
                outputInts[i] = Color(g, g, g).rgb
            }
            return outputInts
        }
    }
}