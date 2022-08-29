package nm.handwritten

import org.slf4j.LoggerFactory
import nm.handwritten.ui.ProgressBar
import javax.swing.JFrame
import kotlin.Throws
import kotlin.jvm.JvmStatic
import nm.handwritten.ui.UI
import java.lang.Exception
import java.util.concurrent.Executors
import java.lang.Runnable


object Run {
    private val LOGGER = LoggerFactory.getLogger(Run::class.java)
    private val mainFrame = JFrame()
    @Throws(Exception::class)
    @JvmStatic
    fun main(args: Array<String>) {
        LOGGER.info("Application is starting ... ")
        val progressBar = ProgressBar(mainFrame, true)
        progressBar.showProgressBar("Collecting data this make take several seconds!")
        val ui = UI()
        Executors.newCachedThreadPool().submit {
            try {
                ui.initUI()
            } finally {
                progressBar.setVisible(false)
                mainFrame.dispose()
            }
        }
    }
}