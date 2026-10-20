package com.huaan.sales.myapplication6.vision

import com.huaan.sales.myapplication6.domain.DocumentGeometry
import com.huaan.sales.myapplication6.ml.DewarpGrid
import com.huaan.sales.myapplication6.ml.OnnxDewarper
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Rect
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.roundToInt

/** Compose learned reverse coordinates and user geometry, sampling the original only once. */
class OriginalResolutionWarp(
    private val source: Mat,
    private val geometry: DocumentGeometry,
    private val perspectiveEnabled: Boolean = true,
) : AutoCloseable {
    private val corners = geometry.corners.let { points ->
        if (perspectiveEnabled) points else {
            val left = points.minOf { it.x }; val right = points.maxOf { it.x }
            val top = points.minOf { it.y }; val bottom = points.maxOf { it.y }
            listOf(com.huaan.sales.myapplication6.domain.NormalizedPoint(left, top),
                com.huaan.sales.myapplication6.domain.NormalizedPoint(right, top),
                com.huaan.sales.myapplication6.domain.NormalizedPoint(right, bottom),
                com.huaan.sales.myapplication6.domain.NormalizedPoint(left, bottom))
        }
    }.map {
        Point(it.x.toDouble() * (source.cols() - 1), it.y.toDouble() * (source.rows() - 1))
    }
    val width = max(distance(corners[0], corners[1]), distance(corners[3], corners[2])).roundToInt().coerceAtLeast(64)
    val height = max(distance(corners[0], corners[3]), distance(corners[1], corners[2])).roundToInt().coerceAtLeast(64)
    private val matrix: Mat
    private val coefficients: DoubleArray
    private val offsets: List<Point>

    init {
        val polygon = org.opencv.core.MatOfPoint(*corners.toTypedArray())
        try {
            require(Imgproc.isContourConvex(polygon) && kotlin.math.abs(Imgproc.contourArea(polygon)) > 64) {
                "边框交叉或面积过小，请重新调整四角"
            }
        } finally { polygon.release() }
        val unit = MatOfPoint2f(Point(0.0, 0.0), Point(1.0, 0.0), Point(1.0, 1.0), Point(0.0, 1.0))
        val target = MatOfPoint2f(*corners.toTypedArray())
        matrix = try { Imgproc.getPerspectiveTransform(unit, target) } finally { unit.release(); target.release() }
        coefficients = DoubleArray(9).also { matrix.get(0, 0, it) }
        offsets = (0..3).map { index ->
            // A default midpoint is straight in source pixels. Inverse homography
            // need not place u=.5 at that Euclidean midpoint, so subtract the linear
            // source midpoint, preserving the exact planar transform by default.
            val a = corners[index]; val b = corners[(index + 1) % 4]
            val p = geometry.edgeMidpoints[index]
            if (!perspectiveEnabled) Point(0.0, 0.0) else
                Point(p.x * (source.cols() - 1) - (a.x + b.x) / 2,
                    p.y * (source.rows() - 1) - (a.y + b.y) / 2)
        }
    }

    fun render(dewarper: OnnxDewarper?, strength: Float): Mat {
        val grid = if (dewarper != null && strength > 0f) {
            val modelInput = remap(488, 712, null, 0f)
            try { dewarper.predict(modelInput) } finally { modelInput.release() }
        } else null
        return remap(width, height, grid, strength.coerceIn(0f, 1f))
    }

    private fun writeSourcePoint(u: Float, v: Float, xs: FloatArray, ys: FloatArray, index: Int) {
        val m = coefficients
        val denominator = m[6] * u + m[7] * v + m[8]
        require(kotlin.math.abs(denominator) > 1e-8) { "边框透视变换无效" }
        val px = (m[0] * u + m[1] * v + m[2]) / denominator
        val py = (m[3] * u + m[4] * v + m[5]) / denominator
        fun edgeParameter(s: Float, start: Double, end: Double): Float =
            (s * end / ((1 - s) * start + s * end)).toFloat()
        val top = edgeParameter(u, m[8], m[6] + m[8])
        val bottom = edgeParameter(u, m[7] + m[8], m[6] + m[7] + m[8])
        val left = edgeParameter(v, m[8], m[7] + m[8])
        val right = edgeParameter(v, m[6] + m[8], m[6] + m[7] + m[8])
        val w0 = 4 * top * (1 - top) * (1 - v)
        val w1 = 4 * right * (1 - right) * u
        val w2 = 4 * bottom * (1 - bottom) * v
        val w3 = 4 * left * (1 - left) * (1 - u)
        xs[index] = (px + w0 * offsets[0].x + w1 * offsets[1].x + w2 * offsets[2].x + w3 * offsets[3].x).toFloat()
        ys[index] = (py + w0 * offsets[0].y + w1 * offsets[1].y + w2 * offsets[2].y + w3 * offsets[3].y).toFloat()
    }

    private fun remap(outWidth: Int, outHeight: Int, grid: DewarpGrid?, strength: Float): Mat {
        val output = Mat(outHeight, outWidth, source.type())
        try {
            if (grid == null && offsets.all { kotlin.math.abs(it.x) < 0.01 && kotlin.math.abs(it.y) < 0.01 }) {
                // OpenCV evaluates the planar homography natively, preserving original pixels
                // without constructing maps or running a Kotlin loop for every output pixel.
                val inverse = Mat(3, 3, CvType.CV_64FC1)
                try {
                    val m = coefficients
                    inverse.put(0, 0, *doubleArrayOf(
                        m[0] / (outWidth - 1), m[1] / (outHeight - 1), m[2],
                        m[3] / (outWidth - 1), m[4] / (outHeight - 1), m[5],
                        m[6] / (outWidth - 1), m[7] / (outHeight - 1), m[8],
                    ))
                    Imgproc.warpPerspective(source, output, inverse, Size(outWidth.toDouble(), outHeight.toDouble()),
                        Imgproc.INTER_LINEAR or Imgproc.WARP_INVERSE_MAP, Core.BORDER_REPLICATE)
                } finally { inverse.release() }
                return output
            }
            // Bound the auxiliary maps to a strip, including for high-resolution photos.
            for (top in 0 until outHeight step 64) {
                val rows = minOf(64, outHeight - top)
                val xs = FloatArray(outWidth * rows)
                val ys = FloatArray(xs.size)
                for (r in 0 until rows) for (x in 0 until outWidth) {
                    val u = x.toFloat() / (outWidth - 1)
                    val v = (top + r).toFloat() / (outHeight - 1)
                    val uv = grid?.sample(u, v)
                    writeSourcePoint(if (uv != null) u + strength * (uv.x - u) else u,
                        if (uv != null) v + strength * (uv.y - v) else v, xs, ys, r * outWidth + x)
                }
                val mapX = Mat(rows, outWidth, CvType.CV_32FC1)
                val mapY = Mat(rows, outWidth, CvType.CV_32FC1)
                val roi = output.submat(Rect(0, top, outWidth, rows))
                try {
                    mapX.put(0, 0, xs); mapY.put(0, 0, ys)
                    Imgproc.remap(source, roi, mapX, mapY, Imgproc.INTER_LINEAR, Core.BORDER_REPLICATE)
                } finally { mapX.release(); mapY.release(); roi.release() }
            }
            return output
        } catch (error: Throwable) {
            output.release()
            throw error
        }
    }

    override fun close() = matrix.release()
    private fun distance(a: Point, b: Point) = hypot(a.x - b.x, a.y - b.y)
}
