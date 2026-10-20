package com.huaan.sales.myapplication6.vision

import com.huaan.sales.myapplication6.domain.DocumentDetection
import com.huaan.sales.myapplication6.domain.NormalizedPoint
import com.huaan.sales.myapplication6.domain.QualityIssue
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.MatOfDouble
import org.opencv.core.Point
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.abs
import kotlin.math.max

interface PreviewDetector {
    fun detect(gray: Mat, rotationDegrees: Int): DocumentDetection

    /** BGR input. Still capture detectors override this instead of replicating gray channels. */
    fun detectColor(bgr: Mat): DocumentDetection {
        val gray = Mat()
        return try {
            Imgproc.cvtColor(bgr, gray, Imgproc.COLOR_BGR2GRAY)
            detect(gray, 0)
        } finally {
            gray.release()
        }
    }
}

class OpenCvDocumentDetector(
    private val maxAnalysisEdge: Int = 640,
) : PreviewDetector {
    override fun detect(gray: Mat, rotationDegrees: Int): DocumentDetection {
        val started = System.nanoTime()
        val oriented = orient(gray, rotationDegrees)
        val scale = minOf(1.0, maxAnalysisEdge.toDouble() / max(oriented.cols(), oriented.rows()))
        val working = Mat()
        if (scale < 1.0) {
            Imgproc.resize(oriented, working, Size(), scale, scale, Imgproc.INTER_AREA)
        } else {
            oriented.copyTo(working)
        }

        val blurred = Mat()
        val edges = Mat()
        Imgproc.GaussianBlur(working, blurred, Size(5.0, 5.0), 0.0)
        Imgproc.Canny(blurred, edges, 55.0, 155.0)
        val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(5.0, 5.0))
        Imgproc.morphologyEx(edges, edges, Imgproc.MORPH_CLOSE, kernel)

        val contours = mutableListOf<MatOfPoint>()
        val hierarchy = Mat()
        Imgproc.findContours(edges, contours, hierarchy, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE)
        hierarchy.release()
        val imageArea = working.cols().toDouble() * working.rows()
        var best: List<Point>? = null
        var bestScore = 0.0
        contours.forEach { contour ->
            val area = abs(Imgproc.contourArea(contour))
            if (area < imageArea * 0.012 || area > imageArea * 0.995) return@forEach
            val curve = MatOfPoint2f(*contour.toArray())
            val approximation = MatOfPoint2f()
            Imgproc.approxPolyDP(curve, approximation, Imgproc.arcLength(curve, true) * 0.025, true)
            val points = approximation.toArray().toList()
            val polygon = MatOfPoint(*approximation.toArray())
            if (points.size == 4 && Imgproc.isContourConvex(polygon)) {
                val rectangularity = area / boundingArea(points)
                val score = (area / imageArea) * rectangularity.coerceIn(0.0, 1.0)
                if (score > bestScore) {
                    bestScore = score
                    best = orderCorners(points)
                }
            }
            polygon.release()
            curve.release()
            approximation.release()
        }

        val issues = qualityIssues(working, best, bestScore).toMutableSet()
        best?.let { points ->
            if (abs(points.indices.sumOf { i ->
                    val a = points[i]; val b = points[(i + 1) % 4]
                    a.x * b.y - b.x * a.y
                }) / 2.0 < imageArea * 0.10) issues.add(QualityIssue.TOO_SMALL)
        }
        val corners = best.orEmpty().map {
            NormalizedPoint(
                x = (it.x / working.cols()).toFloat().coerceIn(0f, 1f),
                y = (it.y / working.rows()).toFloat().coerceIn(0f, 1f),
            )
        }
        val confidence = when {
            best == null -> 0f
            else -> (bestScore / 0.72).toFloat().coerceIn(0f, 1f)
        }

        listOf(oriented, working, blurred, edges, kernel).forEach { it.release() }
        contours.forEach { it.release() }
        return DocumentDetection(
            contour = corners,
            corners = corners,
            confidence = confidence,
            qualityIssues = issues,
            rotationDegrees = rotationDegrees,
            sourceWidth = if (rotationDegrees % 180 == 0) gray.cols() else gray.rows(),
            sourceHeight = if (rotationDegrees % 180 == 0) gray.rows() else gray.cols(),
            inferenceMillis = (System.nanoTime() - started) / 1_000_000,
        )
    }

    private fun qualityIssues(
        gray: Mat,
        corners: List<Point>?,
        score: Double,
    ): Set<QualityIssue> = buildSet {
        val mean = Core.mean(gray).`val`[0]
        if (mean < 52) add(QualityIssue.TOO_DARK)
        if (mean > 220) add(QualityIssue.TOO_BRIGHT)

        val laplacian = Mat()
        Imgproc.Laplacian(gray, laplacian, CvType.CV_64F)
        val meanValue = MatOfDouble()
        val standardDeviation = MatOfDouble()
        Core.meanStdDev(laplacian, meanValue, standardDeviation)
        val blurVariance = standardDeviation.get(0, 0)[0].let { it * it }
        if (blurVariance < 55.0) add(QualityIssue.TOO_BLURRY)
        laplacian.release()
        meanValue.release()
        standardDeviation.release()

        val highlight = Mat()
        Imgproc.threshold(gray, highlight, 248.0, 255.0, Imgproc.THRESH_BINARY)
        val highlightRatio = Core.countNonZero(highlight).toDouble() / gray.total()
        // A reflection can cover more than the old 8% upper bound. Exclude an almost
        // completely white frame because that is already reported as TOO_BRIGHT/NO_DOCUMENT.
        if (highlightRatio >= 0.008 && highlightRatio < 0.85) add(QualityIssue.GLARE)
        highlight.release()

        if (corners == null) {
            add(QualityIssue.NO_DOCUMENT)
        } else {
            if (score < 0.32) add(QualityIssue.LOW_CONFIDENCE)
            val marginX = gray.cols() * 0.025
            val marginY = gray.rows() * 0.025
            if (corners.any { it.x < marginX || it.x > gray.cols() - marginX || it.y < marginY || it.y > gray.rows() - marginY }) {
                add(QualityIssue.EDGE_CLIPPED)
            }
        }
    }

    private fun orient(source: Mat, rotationDegrees: Int): Mat {
        val output = Mat()
        when ((rotationDegrees % 360 + 360) % 360) {
            90 -> Core.rotate(source, output, Core.ROTATE_90_CLOCKWISE)
            180 -> Core.rotate(source, output, Core.ROTATE_180)
            270 -> Core.rotate(source, output, Core.ROTATE_90_COUNTERCLOCKWISE)
            else -> source.copyTo(output)
        }
        return output
    }

    private fun boundingArea(points: List<Point>): Double {
        val width = points.maxOf { it.x } - points.minOf { it.x }
        val height = points.maxOf { it.y } - points.minOf { it.y }
        return max(1.0, width * height)
    }

    private fun orderCorners(points: List<Point>): List<Point> {
        val topLeft = points.minBy { it.x + it.y }
        val bottomRight = points.maxBy { it.x + it.y }
        val topRight = points.maxBy { it.x - it.y }
        val bottomLeft = points.minBy { it.x - it.y }
        return listOf(topLeft, topRight, bottomRight, bottomLeft)
    }
}

class DetectionSmoother(
    private val alpha: Float = 0.45f,
    private val holdMillis: Long = 1_200L,
) {
    private var previous: DocumentDetection? = null
    private var pending: DocumentDetection? = null
    private var pendingCount = 0
    private var lastSeenMillis = 0L

    fun reset() {
        previous = null
        pending = null
        pendingCount = 0
        lastSeenMillis = 0L
    }

    fun update(current: DocumentDetection, nowMillis: Long): DocumentDetection {
        previous?.let {
            if (it.sourceWidth != current.sourceWidth || it.sourceHeight != current.sourceHeight ||
                it.rotationDegrees != current.rotationDegrees) reset()
        }
        val old = previous
        if (current.corners.size != 4) {
            pending = null
            pendingCount = 0
            if (old != null && nowMillis - lastSeenMillis <= holdMillis) return uncertain(old, current)
            previous = null
            return current
        }

        // A new page or a large motion needs two consecutive agreeing observations.
        // Never interpolate between unrelated rectangles: that creates a nonexistent page.
        if (old == null || displacement(old, current) > 0.045f) {
            pendingCount = if (pending?.let { displacement(it, current) <= 0.025f } == true)
                pendingCount + 1 else 1
            pending = current
            if (pendingCount < 2) {
                return uncertain(if (old != null && nowMillis - lastSeenMillis <= holdMillis) old else current, current)
            }
            pending = null
            pendingCount = 0
            previous = current
            lastSeenMillis = nowMillis
            return current
        }

        pending = null
        pendingCount = 0
        val trackingAlpha = if (displacement(old, current) > 0.012f) max(alpha, 0.70f) else alpha
        val points = current.corners.indices.map { index ->
            val before = old.corners[index]
            val after = current.corners[index]
            if (before.distanceTo(after) < 0.004f) before else NormalizedPoint(
                before.x * (1f - trackingAlpha) + after.x * trackingAlpha,
                before.y * (1f - trackingAlpha) + after.y * trackingAlpha,
            )
        }
        val smoothed = current.copy(corners = points, contour = points)
        previous = smoothed
        lastSeenMillis = nowMillis
        return smoothed
    }

    private fun displacement(a: DocumentDetection, b: DocumentDetection): Float =
        a.corners.indices.maxOf { a.corners[it].distanceTo(b.corners[it]) }

    private fun uncertain(shown: DocumentDetection, current: DocumentDetection) = shown.copy(
        confidence = minOf(current.confidence, 0.54f),
        qualityIssues = current.qualityIssues + QualityIssue.LOW_CONFIDENCE,
        inferenceMillis = current.inferenceMillis,
    )
}
