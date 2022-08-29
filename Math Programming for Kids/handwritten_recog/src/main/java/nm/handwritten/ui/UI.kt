package nm.handwritten.ui

import com.mortennobel.imagescaling.ResampleFilters
import com.mortennobel.imagescaling.ResampleOp
import org.slf4j.LoggerFactory
import nm.handwritten.cnn.ConvolutionalNeuralNetwork
import nm.handwritten.nn.NeuralNetwork
import nm.handwritten.data.LabeledImage
import nm.handwritten.ui.DrawArea
import java.awt.*
import java.awt.event.ActionEvent
import java.awt.event.ActionListener
import java.awt.event.WindowAdapter
import java.awt.event.WindowEvent
import java.awt.image.BufferedImage
import java.io.IOException
import java.util.concurrent.Executors
import javax.swing.*
import javax.swing.plaf.FontUIResource

class UI {
    private val neuralNetwork: NeuralNetwork = NeuralNetwork()
    private val convolutionalNeuralNetwork: ConvolutionalNeuralNetwork = ConvolutionalNeuralNetwork()
    private var drawArea: DrawArea? = null
    private var mainFrame: JFrame? = null
    private var mainPanel: JPanel? = null
    private var drawAndDigitPredictionPanel: JPanel? = null
    private var modelTrainSize: SpinnerNumberModel? = null
    private var trainField: JSpinner? = null
    private val TRAIN_SIZE = 30000
    private val sansSerifBold = Font("SansSerif", Font.BOLD, 18)
    private val TEST_SIZE = 10000
    private var modelTestSize: SpinnerNumberModel? = null
    private var testField: JSpinner? = null
    private var resultPanel: JPanel? = null
    fun initUI() {
        // create main frame
        mainFrame = createMainFrame()
        mainPanel = JPanel()
        mainPanel!!.setLayout(BorderLayout())
        addTopPanel()
        drawAndDigitPredictionPanel = JPanel(GridLayout())
        addActionPanel()
        addDrawAreaAndPredictionArea()
        mainPanel!!.add(drawAndDigitPredictionPanel, BorderLayout.CENTER)
        addSignature()
        mainFrame!!.add(mainPanel, BorderLayout.CENTER)
        mainFrame!!.setVisible(true)
    }

    private fun addActionPanel() {
        val recognize = JButton("Recognize Digit With Simple NN")
        val recognizeCNN = JButton("Recognize Digit With Conv NN")
        recognize.addActionListener(ActionListener { e: ActionEvent? ->
            val drawImage = drawArea!!.image
            val sbi: BufferedImage = toBufferedImage(drawImage)
            val scaled: Image = scale(sbi)
            val scaledBuffered: BufferedImage = toBufferedImage(scaled)
            val scaledPixels = transformImageToOneDimensionalVector(scaledBuffered)
            val labeledImage = LabeledImage(0, scaledPixels)
            val predict: LabeledImage = neuralNetwork.predict(labeledImage)
            val predictNumber = JLabel("" + predict.label.toInt())
            predictNumber.setForeground(Color.RED)
            predictNumber.setFont(Font("SansSerif", Font.BOLD, 128))
            resultPanel?.removeAll()
            resultPanel?.add(predictNumber)
            resultPanel?.updateUI()
        })
        recognizeCNN.addActionListener(ActionListener { e: ActionEvent? ->
            val drawImage = drawArea!!.image
            val sbi: BufferedImage = toBufferedImage(drawImage)
            val scaled: Image = scale(sbi)
            val scaledBuffered: BufferedImage = toBufferedImage(scaled)
            val scaledPixels = transformImageToOneDimensionalVector(scaledBuffered)
            val labeledImage = LabeledImage(0, scaledPixels)
            val predict: Int = convolutionalNeuralNetwork.predict(labeledImage)
            val predictNumber = JLabel("" + predict)
            predictNumber.setForeground(Color.RED)
            predictNumber.setFont(Font("SansSerif", Font.BOLD, 128))
            resultPanel?.removeAll()
            resultPanel?.add(predictNumber)
            resultPanel?.updateUI()
        })
        val clear = JButton("Clear")
        clear.addActionListener(ActionListener { e: ActionEvent? ->
            drawArea!!.image = null
            drawArea!!.repaint()
            drawAndDigitPredictionPanel?.updateUI()
        })
        val actionPanel = JPanel(GridLayout(8, 1))
        actionPanel.add(recognizeCNN)
        actionPanel.add(recognize)
        actionPanel.add(clear)
        drawAndDigitPredictionPanel?.add(actionPanel)
    }

    private fun addDrawAreaAndPredictionArea() {
        drawArea = DrawArea()
        drawAndDigitPredictionPanel?.add(drawArea)
        resultPanel = JPanel()
        resultPanel!!.setLayout(GridBagLayout())
        drawAndDigitPredictionPanel?.add(resultPanel)
    }

    private fun addTopPanel() {
        val topPanel = JPanel(FlowLayout())
        val trainNN = JButton("Train NN")
        trainNN.addActionListener(ActionListener { e: ActionEvent? ->
            val i: Int = JOptionPane.showConfirmDialog(mainFrame, "Are you sure this may take some time to train?")
            if (i == JOptionPane.OK_OPTION) {
                val progressBar = mainFrame?.let { ProgressBar(it) }
                SwingUtilities.invokeLater(Runnable { progressBar?.showProgressBar("Training may take one or two minutes...") })
                Executors.newCachedThreadPool().submit(Runnable {
                    try {
                        LOGGER.info("Start of train Neural Network")
                        neuralNetwork.train(trainField?.getValue() as Int, testField?.getValue() as Int)
                        LOGGER.info("End of train Neural Network")
                    } finally {
                        progressBar?.setVisible(false)
                    }
                })
            }
        })
        val trainCNN = JButton("Train Convolutional NN")
        trainCNN.addActionListener(ActionListener { e: ActionEvent? ->
            val i: Int = JOptionPane.showConfirmDialog(mainFrame,
                "Are you sure, training requires >10GB memory and more than 1 hour?")
            if (i == JOptionPane.OK_OPTION) {
                val progressBar = mainFrame?.let { ProgressBar(it) }
                SwingUtilities.invokeLater(Runnable { progressBar?.showProgressBar("Training may take a while...") })
                Executors.newCachedThreadPool().submit(Runnable {
                    try {
                        LOGGER.info("Start of train Convolutional Neural Network")
                        convolutionalNeuralNetwork.train(trainField?.getValue() as Int, testField?.getValue() as Int)
                        LOGGER.info("End of train Convolutional Neural Network")
                    } catch (e1: IOException) {
                        LOGGER.error("CNN not trained $e1")
                        throw RuntimeException(e1)
                    } finally {
                        progressBar?.setVisible(false)
                    }
                })
            }
        })
        topPanel.add(trainCNN)
        topPanel.add(trainNN)
        val tL = JLabel("Training Data")
        tL.setFont(sansSerifBold)
        topPanel.add(tL)
        modelTrainSize = SpinnerNumberModel(TRAIN_SIZE, 10000, 60000, 1000)
        trainField = JSpinner(modelTrainSize)
        trainField!!.setFont(sansSerifBold)
        topPanel.add(trainField)
        val ttL = JLabel("Test Data")
        ttL.setFont(sansSerifBold)
        topPanel.add(ttL)
        modelTestSize = SpinnerNumberModel(TEST_SIZE, 1000, 10000, 500)
        testField = JSpinner(modelTestSize)
        testField!!.setFont(sansSerifBold)
        topPanel.add(testField)
        mainPanel?.add(topPanel, BorderLayout.NORTH)
    }

    private fun createMainFrame(): JFrame {
        val mainFrame = JFrame()
        mainFrame.setTitle("Digit Recognizer")
        mainFrame.setDefaultCloseOperation(WindowConstants.DISPOSE_ON_CLOSE)
        mainFrame.setSize(FRAME_WIDTH, FRAME_HEIGHT)
        mainFrame.setLocationRelativeTo(null)
        mainFrame.addWindowListener(object : WindowAdapter() {
            override fun windowClosed(e: WindowEvent) {
                System.exit(0)
            }
        })
        val imageIcon = ImageIcon("icon.png")
        mainFrame.setIconImage(imageIcon.getImage())
        return mainFrame
    }

    private fun addSignature() {
        val signature = JLabel("Tanmay_Gaurav.tech", JLabel.HORIZONTAL)
        signature.setFont(Font(Font.SANS_SERIF, Font.ITALIC, 20))
        signature.setForeground(Color.BLUE)
        mainPanel?.add(signature, BorderLayout.SOUTH)
    }

    companion object {
        private val LOGGER = LoggerFactory.getLogger(UI::class.java)
        private const val FRAME_WIDTH = 1200
        private const val FRAME_HEIGHT = 628
        private fun scale(imageToScale: BufferedImage): BufferedImage {
            val resizeOp = ResampleOp(28, 28)
            resizeOp.filter = ResampleFilters.getLanczos3Filter()
            return resizeOp.filter(imageToScale, null)
        }

        private fun toBufferedImage(img: Image?): BufferedImage {
            // Create a buffered image with transparency
            val bimage = BufferedImage(img!!.getWidth(null), img.getHeight(null), BufferedImage.TYPE_INT_ARGB)

            // Draw the image on to the buffered image
            val bGr: Graphics2D = bimage.createGraphics()
            bGr.drawImage(img, 0, 0, null)
            bGr.dispose()

            // Return the buffered image
            return bimage
        }

        private fun transformImageToOneDimensionalVector(img: BufferedImage): DoubleArray {
            val imageGray = DoubleArray(28 * 28)
            val w: Int = img.getWidth()
            val h: Int = img.getHeight()
            var index = 0
            for (i in 0 until w) {
                for (j in 0 until h) {
                    val color = Color(img.getRGB(j, i), true)
                    val red = color.red
                    val green = color.green
                    val blue = color.blue
                    val v = 255 - (red + green + blue) / 3.0
                    imageGray[index] = v
                    index++
                }
            }
            return imageGray
        }
    }

    init {
        UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName())
        UIManager.put("Button.font", FontUIResource(Font("Dialog", Font.BOLD, 18)))
        UIManager.put("ProgressBar.font", FontUIResource(Font("Dialog", Font.BOLD, 18)))
        neuralNetwork.init()
        convolutionalNeuralNetwork.init()
    }
}