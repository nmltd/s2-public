package nm.handwritten.data

import org.slf4j.LoggerFactory
import nm.handwritten.data.LabeledImage
import nm.handwritten.data.IdxReader
import java.io.FileInputStream
import java.lang.Exception
import java.lang.RuntimeException
import java.util.ArrayList

object IdxReader {
    private val LOGGER = LoggerFactory.getLogger(IdxReader::class.java)
    const val INPUT_IMAGE_PATH = "resources/train-images.idx3-ubyte"
    const val INPUT_LABEL_PATH = "resources/train-labels.idx1-ubyte"
    const val INPUT_IMAGE_PATH_TEST_DATA = "resources/t10k-images.idx3-ubyte"
    const val INPUT_LABEL_PATH_TEST_DATA = "resources/t10k-labels.idx1-ubyte"
    const val VECTOR_DIMENSION = 784 //square 28*28 as from data set -> array 784 items

    /**
     * @param size
     * @return
     */
    @JvmStatic
    fun loadData(size: Int): List<LabeledImage> {
        return getLabeledImages(INPUT_IMAGE_PATH, INPUT_LABEL_PATH, size)
    }

    /**
     * @param size
     * @return
     */
    @JvmStatic
    fun loadTestData(size: Int): List<LabeledImage> {
        return getLabeledImages(INPUT_IMAGE_PATH_TEST_DATA, INPUT_LABEL_PATH_TEST_DATA, size)
    }

    private fun getLabeledImages(
        inputImagePath: String,
        inputLabelPath: String,
        amountOfDataSet: Int
    ): List<LabeledImage> {
        val labeledImageArrayList: MutableList<LabeledImage> = ArrayList(amountOfDataSet)
        try {
            FileInputStream(inputImagePath).use { inImage ->
                FileInputStream(inputLabelPath).use { inLabel ->

                    // just skip the amount of a data
                    // see the test and description for dataset
                    inImage.skip(16)
                    inLabel.skip(8)
                    LOGGER.debug("Available bytes in inputImage stream after read: " + inImage.available())
                    LOGGER.debug("Available bytes in inputLabel stream after read: " + inLabel.available())

                    //empty array for 784 pixels - the image from 28x28 pixels in a single row
                    val imgPixels = DoubleArray(VECTOR_DIMENSION)
                    LOGGER.info("Creating ADT filed with Labeled Images ...")
                    val start = System.currentTimeMillis()
                    for (i in 0 until amountOfDataSet) {
                        if (i % 1000 == 0) {
                            LOGGER.info("Number of images extracted: $i")
                        }
                        //it fills the array of pixels
                        for (index in 0 until VECTOR_DIMENSION) {
                            imgPixels[index] = inImage.read().toDouble()
                        }
                        //it creates a label for that
                        val label = inLabel.read()
                        //it creates a compound object and adds them to a list
                        labeledImageArrayList.add(LabeledImage(label, imgPixels))
                    }
                    LOGGER.info("Time to load LabeledImages in seconds: " + (System.currentTimeMillis() - start) / 1000.0)
                }
            }
        } catch (e: Exception) {
            LOGGER.error("Smth went wrong: \n$e")
            throw RuntimeException(e)
        }
        return labeledImageArrayList
    }
}