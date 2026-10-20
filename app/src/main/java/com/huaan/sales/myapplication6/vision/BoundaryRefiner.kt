package com.huaan.sales.myapplication6.vision

import com.huaan.sales.myapplication6.domain.DocumentDetection
import com.huaan.sales.myapplication6.domain.NormalizedPoint
import com.huaan.sales.myapplication6.domain.QualityIssue
import com.huaan.sales.myapplication6.ml.CornerSearchGeometry
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.*

/** Local evidence may refine a page; it may never choose a different paper layer. */
class BoundaryRefiner {
    private data class Line(val ax: Double, val ay: Double, val bx: Double, val by: Double,
        val evidence: Float, val reliable: Boolean)

    fun refine(image: Mat, detection: DocumentDetection): DocumentDetection {
        if (detection.corners.size != 4) return detection
        val gray = Mat(); val resized = Mat()
        try {
            Imgproc.cvtColor(image, gray, Imgproc.COLOR_BGR2GRAY)
            val scale = min(1.0, 1024.0 / max(image.cols(), image.rows()))
            Imgproc.resize(gray, resized, Size(max(2, (gray.cols() * scale).roundToInt()).toDouble(),
                max(2, (gray.rows() * scale).roundToInt()).toDouble()), 0.0, 0.0, Imgproc.INTER_AREA)
            val w = resized.cols(); val h = resized.rows()
            val bytes = ByteArray(w * h).also { resized.get(0, 0, it) }
            val corners = detection.corners
            val lines = (0..3).map { i -> search(bytes, w, h, corners[i], corners[(i + 1) % 4]) }
            val proposed = corners.indices.map { i ->
                val before = lines[(i + 3) % 4]; val after = lines[i]
                // A weak side cannot be pulled around by its two neighbouring sides.
                if (!before.reliable || !after.reliable) corners[i] else intersection(before, after, w, h) ?: corners[i]
            }
            val maxMove = hypot(w.toDouble(), h.toDouble()) * .012
            val valid = CornerSearchGeometry.valid(proposed) && proposed.indices.all { i ->
                hypot((proposed[i].x - corners[i].x) * (w - 1).toDouble(),
                    (proposed[i].y - corners[i].y) * (h - 1).toDouble()) <= maxMove
            } && abs(CornerSearchGeometry.area(proposed) / CornerSearchGeometry.area(corners) - 1f) <= .06f
            val weak = lines.indices.filter { !lines[it].reliable }.toSet()
            val review = weak.isNotEmpty() || !valid || detection.requiresConfirmation
            return detection.copy(
                corners = if (valid) proposed else corners,
                contour = if (valid) proposed else detection.contour,
                edgeEvidence = lines.map { it.evidence },
                invisibleEdges = detection.invisibleEdges + weak,
                requiresConfirmation = review,
                qualityIssues = if (review) detection.qualityIssues + QualityIssue.LOW_CONFIDENCE else detection.qualityIssues,
            )
        } finally { gray.release(); resized.release() }
    }

    private fun search(bytes: ByteArray, w: Int, h: Int, a: NormalizedPoint, b: NormalizedPoint): Line {
        val ax = a.x * (w - 1).toDouble(); val ay = a.y * (h - 1).toDouble()
        val bx = b.x * (w - 1).toDouble(); val by = b.y * (h - 1).toDouble()
        val length = hypot(bx - ax, by - ay)
        val original = Line(ax, ay, bx, by, 0f, false)
        if (length < 24) return original
        val nx = -(by - ay) / length; val ny = (bx - ax) / length
        val radius = (min(w, h) * .012).roundToInt().coerceIn(2, 12)
        fun pixel(x: Double, y: Double): Double = (bytes[y.roundToInt().coerceIn(0, h - 1) * w +
            x.roundToInt().coerceIn(0, w - 1)].toInt() and 255).toDouble()
        fun contrast(x: Double, y: Double): Double {
            val inner = pixel(x + nx * 2, y + ny * 2)
            val outer = pixel(x - nx * 2, y - ny * 2)
            val wide = abs(pixel(x + nx * 4, y + ny * 4) - pixel(x - nx * 4, y - ny * 4))
            val texture = abs(pixel(x + nx * 2 + ny * 2, y + ny * 2 - nx * 2) - inner) +
                abs(pixel(x - nx * 2 + ny * 2, y - ny * 2 - nx * 2) - outer)
            return abs(inner - outer) * .55 + wide * .35 + min(texture, 20.0) * .10
        }
        data class Vote(val offset: Int, val tilt: Int, val score: Double, val support: Float)
        val votes = mutableListOf<Vote>()
        for (offset in -radius..radius) for (tilt in -radius / 2..radius / 2 step 2) {
            var score = 0.0; var supported = 0; var visible = 0
            for (sample in 0 until 48) {
                val t = .08 + sample * .84 / 47
                val shift = offset + tilt * (t - .5) * 2
                val x = ax + (bx - ax) * t + nx * shift
                val y = ay + (by - ay) * t + ny * shift
                if (x < 5 || y < 5 || x > w - 6 || y > h - 6) continue
                visible++
                val value = contrast(x, y)
                if (value >= 8) supported++
                score += min(value, 64.0)
            }
            if (visible >= 40) votes += Vote(offset, tilt,
                score / visible - abs(offset) * .18 - abs(tilt) * .10, supported / 48f)
        }
        val best = votes.maxByOrNull { it.score } ?: return original
        val runner = votes.filter { abs(it.offset - best.offset) >= 4 }.maxByOrNull { it.score }
        val ambiguous = runner != null && runner.score > best.score * .88 && runner.support >= .65f
        val reliable = best.score >= 9 && best.support >= .75f && !ambiguous
        if (!reliable) return original.copy(evidence = best.support)
        // Small gains do not justify moving a correctly predicted edge.
        val center = votes.filter { it.offset == 0 && abs(it.tilt) <= 1 }.maxOfOrNull { it.score } ?: 0.0
        if (best.score < center + 2) return original.copy(evidence = best.support, reliable = true)
        return Line(ax + nx * (best.offset - best.tilt), ay + ny * (best.offset - best.tilt),
            bx + nx * (best.offset + best.tilt), by + ny * (best.offset + best.tilt), best.support, true)
    }

    private fun intersection(a: Line, b: Line, w: Int, h: Int): NormalizedPoint? {
        val dx = a.bx - a.ax; val dy = a.by - a.ay
        val ex = b.bx - b.ax; val ey = b.by - b.ay
        val determinant = dx * ey - dy * ex
        if (abs(determinant) < 1e-6) return null
        val t = ((b.ax - a.ax) * ey - (b.ay - a.ay) * ex) / determinant
        val x = (a.ax + t * dx) / (w - 1); val y = (a.ay + t * dy) / (h - 1)
        if (x !in 0.0..1.0 || y !in 0.0..1.0) return null
        return NormalizedPoint(x.toFloat(), y.toFloat())
    }
}
