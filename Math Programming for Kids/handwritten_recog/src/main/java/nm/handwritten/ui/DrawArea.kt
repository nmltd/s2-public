package nm.handwritten.ui

import java.awt.*
import javax.swing.JComponent
import javax.swing.BorderFactory
import javax.swing.border.TitledBorder
import java.awt.event.MouseAdapter
import java.awt.event.MouseMotionAdapter
import java.awt.event.MouseEvent
// for drawing operations
class DrawArea : JComponent() {
    private val sansSerifBold = Font("Times New Roman", Font.BOLD, 18)

    // for drwaing image
    var image: Image? = null

    // grahics 2D will be used to draw on
    private var g2: Graphics2D? = null

    // to capture mouse position by detecting coordinates of it
    private var currentX = 0
    private var currentY = 0
    private var oldX = 0
    private var oldY = 0

    init {
        isDoubleBuffered = false
        border = BorderFactory.createTitledBorder(
            BorderFactory.createEtchedBorder(),
            "Draw any one Digit from 0-9",
            TitledBorder.LEFT,
            TitledBorder.TOP, sansSerifBold, Color.BLUE
        )
        addMouseListener(object : MouseAdapter() {
            override fun mousePressed(e: MouseEvent) {
                // When mouse is pressed it saves the X, Y chords
                oldX = e.x
                oldY = e.y
            }
        })
        addMouseMotionListener(object : MouseMotionAdapter() {
            override fun mouseDragged(e: MouseEvent) {
                // this section captures the X,Y chords when mouse is dragged
                currentX = e.x
                currentY = e.y
                if (g2 != null) {
                    g2!!.stroke = BasicStroke(10F)
                    // it draw the line if g2 context not null
                    g2!!.drawLine(oldX, oldY, currentX, currentY)
                    // reset the area for new shape
                    repaint()
                    // store current coords x,y as olds x,Y
                    oldX = currentX
                    oldY = currentY
                }
            }
        })
    }

    override fun paintComponent(g: Graphics) {
        if (image == null) {
            //image to draw null
            image = createImage(size.width, size.height)
            (image!!.getGraphics() as Graphics2D).also { g2 = it }
            // enable antialiasing
            g2!!.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON)
            // it simply clears drawing area
            clear()
        }
        g.drawImage(image, 0, 0, null)
    }

    fun clear() {
        g2!!.paint = Color.white
        // it simply draws color white to clear the background
        g2!!.fillRect(0, 0, size.width, size.height)
        g2!!.paint = Color.black
        repaint()
    }
}