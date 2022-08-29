package nm.handwritten.ui

import javax.swing.JFrame
import javax.swing.JProgressBar
import javax.swing.SwingUtilities
import java.lang.Runnable
import java.awt.BorderLayout


class ProgressBar {
    private val mainFrame: JFrame
    private var progressBar: JProgressBar
    private var unDecoreate = false

    constructor(mainFrame: JFrame) {
        this.mainFrame = mainFrame
        progressBar = createProgressBar(mainFrame)
    }

    constructor(mainFrame: JFrame, unDecoreate: Boolean) {
        this.mainFrame = mainFrame
        progressBar = createProgressBar(mainFrame)
        this.unDecoreate = unDecoreate
    }

    fun showProgressBar(msg: String?) {
        SwingUtilities.invokeLater {
            if (unDecoreate) {
                mainFrame.setLocationRelativeTo(null)
                mainFrame.isUndecorated = true
            }
            progressBar = createProgressBar(mainFrame)
            progressBar.string = msg
            progressBar.isStringPainted = true
            progressBar.isIndeterminate = true
            progressBar.isVisible = true
            mainFrame.add(progressBar, BorderLayout.NORTH)
            if (unDecoreate) {
                mainFrame.pack()
                mainFrame.isVisible = true
            }
            mainFrame.repaint()
        }
    }

    private fun createProgressBar(mainFrame: JFrame): JProgressBar {
        val jProgressBar = JProgressBar(JProgressBar.HORIZONTAL)
        jProgressBar.isVisible = false
        mainFrame.add(jProgressBar, BorderLayout.NORTH)
        return jProgressBar
    }

    fun setVisible(visible: Boolean) {
        progressBar.isVisible = visible
    }
}