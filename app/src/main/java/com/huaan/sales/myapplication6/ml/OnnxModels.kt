package com.huaan.sales.myapplication6.ml

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import com.huaan.sales.myapplication6.domain.DocumentDetection
import com.huaan.sales.myapplication6.domain.DetectionSource
import com.huaan.sales.myapplication6.domain.NormalizedPoint
import com.huaan.sales.myapplication6.domain.QualityIssue
import com.huaan.sales.myapplication6.vision.PreviewDetector
import org.opencv.core.CvType
import org.opencv.core.Core
import org.opencv.core.Mat
import org.opencv.core.Rect
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import java.nio.FloatBuffer
import kotlin.math.ceil
import kotlin.math.abs
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.roundToInt

/** Owns a verified CPU session. Callers serialize inference and close. */
class OnnxImageModel(private val session: OrtSession) : AutoCloseable {
    @Synchronized
    fun run(bgr: Mat, width: Int, height: Int, rgb: Boolean): Pair<FloatArray, LongArray> {
        require(bgr.type() == CvType.CV_8UC3) { "模型需要三通道 BGR 图像" }
        val resized = Mat()
        try {
            Imgproc.resize(bgr, resized, Size(width.toDouble(), height.toDouble()), 0.0, 0.0, Imgproc.INTER_LINEAR)
            val bytes = ByteArray(width * height * 3)
            resized.get(0, 0, bytes)
            val plane = width * height
            val values = FloatArray(plane * 3)
            for (i in 0 until plane) {
                for (c in 0..2) {
                    values[c * plane + i] = (bytes[i * 3 + if (rgb) 2 - c else c].toInt() and 255) / 255f
                }
            }
            OnnxTensor.createTensor(OrtEnvironment.getEnvironment(), FloatBuffer.wrap(values),
                longArrayOf(1, 3, height.toLong(), width.toLong())).use { input ->
                session.run(mapOf(session.inputNames.single() to input)).use { results ->
                    val output = results[0] as OnnxTensor
                    val buffer = output.floatBuffer
                    val floats = FloatArray(buffer.remaining())
                    buffer.get(floats)
                    require(floats.all { it.isFinite() }) { "模型输出包含无效数值" }
                    return floats to output.info.shape
                }
            }
        } finally {
            resized.release()
        }
    }

    @Synchronized
    override fun close() = session.close()
}

class OnnxCornerDetector(
    private val model: OnnxImageModel,
    private val fallback: PreviewDetector,
    private val source: DetectionSource = DetectionSource.LCNET,
) : PreviewDetector, AutoCloseable {
    // Cache transformations only. Both predictions are recomputed from the CURRENT frame.
    @Volatile private var previewContexts: Pair<Float, Float>? = null
    override fun detect(gray: Mat, rotationDegrees: Int): DocumentDetection = fallback.detect(gray, rotationDegrees)

    /** The camera throttles this work; OpenCV supplies quality diagnostics, never an unverified fast quad. */
    fun detectPreviewColor(bgr: Mat): DocumentDetection = detectColorChecked(bgr, still = false)

    override fun detectColor(bgr: Mat): DocumentDetection = detectColorChecked(bgr, still = true)

    private fun detectColorChecked(bgr: Mat, still: Boolean): DocumentDetection {
        val started = System.nanoTime()
        val baseline = fallback.detectColor(bgr)
        val selected = search(bgr, still)

        val corners = selected?.points.orEmpty()
        val evidence = if (selected != null) boundaryEvidence(bgr, corners) else emptyList()
        val score = selected?.let { if (it.needsReview) minOf(it.score, 0.54f) else it.score } ?: 0f
        val issues = baseline.qualityIssues - setOf(QualityIssue.NO_DOCUMENT,
            QualityIssue.LOW_CONFIDENCE, QualityIssue.EDGE_CLIPPED, QualityIssue.TOO_SMALL) + buildSet {
            if (selected == null) add(QualityIssue.NO_DOCUMENT)
            if (score < 0.55f) add(QualityIssue.LOW_CONFIDENCE)
            if (selected?.clipped == true) add(QualityIssue.EDGE_CLIPPED)
            if (selected != null && CornerSearchGeometry.area(corners) < 0.10f) add(QualityIssue.TOO_SMALL)
            // Keep the model quad available for manual editing, but do not silently export
            // a boundary that has little two-sided image evidence (white-on-white and glare).
            if (evidence.any { it < 0.16f }) {
                add(QualityIssue.LOW_CONFIDENCE)
            }
        }
        return baseline.copy(corners = corners, contour = corners, confidence = score,
            qualityIssues = issues, inferenceMillis = (System.nanoTime() - started) / 1_000_000,
            edgeEvidence = evidence, cornerUncertainty = selected?.uncertainty.orEmpty(),
            detectionSource = source,
            requiresConfirmation = selected?.needsReview == true || QualityIssue.LOW_CONFIDENCE in issues,
            invisibleEdges = evidence.indices.filter { evidence[it] < 0.16f }.toSet(),
            outOfFrame = selected?.clipped == true)
    }

    private data class Candidate(val points: List<NormalizedPoint>, val score: Float,
        val clipped: Boolean, val needsReview: Boolean = false, val context: Float = 0f,
        val uncertainty: List<Float> = emptyList())

    private fun search(bgr: Mat, still: Boolean): Candidate? {
        if (!still) previewContexts?.let { (first, second) ->
            val a = contextCandidate(bgr, first)
            val b = contextCandidate(bgr, second)
            if (a != null && b != null && CornerSearchGeometry.consensusAgrees(a.points, b.points)) {
                val best = if (a.score >= b.score) a else b
                return best.copy(score = minOf(a.score, b.score))
            }
            previewContexts = null
        }
        val raw = candidate(bgr)
        // Verify the original foreground page before adding a canvas that can introduce another layer.
        if (raw != null && !CornerSearchGeometry.touchesFrame(raw.points, 0.04f)) {
            val crop = cropCandidate(bgr, raw.points)
            if (crop != null && CornerSearchGeometry.consensusAgrees(raw.points, crop.points)) {
                return crop.copy(score = minOf(raw.score, crop.score))
            }
        }
        val candidates = mutableListOf<Candidate>().apply { raw?.let(::add) }
        val fractions = mutableListOf(0.08f, 0.15f, 0.25f, 0.40f, 0.60f, 0.75f)
        if (raw != null && !CornerSearchGeometry.touchesFrame(raw.points, 0.04f) &&
            CornerSearchGeometry.area(raw.points) in 0.15f..<0.5f) fractions.add(1, 0.12f)
        for (fraction in fractions) {
            val current = paddedCandidate(bgr, fraction)
            current?.let(candidates::add)
            // A nearly full-frame page needs a nearby scale, not a larger artificial canvas.
            if (fraction == 0.15f && current != null && CornerSearchGeometry.area(current.points) > 0.70f) {
                paddedCandidate(bgr, 0.10f)?.let(candidates::add)
            }
            val supported = supported(candidates)
            val foreground = firstInterior(candidates)
            supported.firstOrNull { (seed, _) -> seed === foreground }?.let { (_, group) ->
                return confirmed(group.maxBy { it.score }, group, raw, still)
            }
            if (fraction == 0.12f) {
                // Select one intact hypothesis only; never splice four corners from different pages.
                // Independent corner support cannot prove a whole page, so confirmation is required.
                candidates.firstOrNull { seed ->
                    val tolerance = CornerSearchGeometry.consensusTolerance(seed.points)
                    seed.points.indices.all { i -> candidates.count { other ->
                        seed.points[i].distanceTo(other.points[i]) <= tolerance
                    } >= 2 }
                }?.let { return it.copy(needsReview = true) }
            }
            if (supported.any { (seed, _) -> CornerSearchGeometry.area(seed.points) > 0.70f }) {
                val largest = supported.maxOf { (seed, _) -> CornerSearchGeometry.area(seed.points) }
                val (seed, group) = supported.first { (seed, _) -> CornerSearchGeometry.area(seed.points) >= largest * 0.80f }
                return confirmed(seed, group, raw, still)
            }
        }

        // White paper on a white desk and saturated reflections can suppress the
        // colour edge evidence in every normal-scale pass.  Give the model one
        // contrast-normalised view as a recovery attempt.  This is deliberately
        // kept out of the normal selection path: a recovered quad is marked for
        // confirmation and can never silently replace a stable colour hypothesis.
        if (candidates.isEmpty() || supported(candidates).isEmpty()) {
            enhancedCandidate(bgr)?.let { enhanced ->
                val agreeing = candidates.filter {
                    CornerSearchGeometry.consensusAgrees(enhanced.points, it.points)
                }
                if (agreeing.isNotEmpty()) {
                    return enhanced.copy(
                        score = minOf(enhanced.score, agreeing.minOf { it.score }),
                        needsReview = true,
                    )
                }
                if (candidates.isEmpty()) return enhanced.copy(needsReview = true)
            }
        }
        if (candidates.isEmpty()) return null
        val supported = supported(candidates)
        // A conflicting outer page must not silently replace an interior foreground hypothesis.
        firstInterior(candidates)?.let { return it.copy(needsReview = true) }
        if (supported.isNotEmpty()) {
            // Completeness is relevant only for unresolved, frame-touching pages.
            val largest = supported.maxOf { (seed, _) -> CornerSearchGeometry.area(seed.points) }
            val (seed, group) = supported.first { (seed, _) -> CornerSearchGeometry.area(seed.points) >= largest * 0.80f }
            return confirmed(seed, group, raw, still)
        }
        return candidates.first().copy(needsReview = true)
    }

    /**
     * Produces a monochrome, locally contrast-normalised view for low-contrast
     * paper/background separation.  This is a detector fallback only; all final
     * geometry still comes from the original-resolution image.
     */
    private fun enhancedCandidate(bgr: Mat): Candidate? {
        val lab = Mat()
        val channels = mutableListOf<Mat>()
        val enhancedL = Mat()
        val enhanced = Mat()
        val clahe = Imgproc.createCLAHE(2.0, Size(8.0, 8.0))
        return try {
            Imgproc.cvtColor(bgr, lab, Imgproc.COLOR_BGR2Lab)
            Core.split(lab, channels)
            if (channels.size != 3) return null
            clahe.apply(channels[0], enhancedL)
            channels[0].release()
            channels[0] = enhancedL
            Core.merge(channels, lab)
            Imgproc.cvtColor(lab, enhanced, Imgproc.COLOR_Lab2BGR)
            candidate(enhanced)?.copy(needsReview = true)
        } finally {
            channels.forEach { it.release() }
            listOf(lab, enhanced).forEach { it.release() }
            clahe.collectGarbage()
        }
    }

    private fun supported(candidates: List<Candidate>) = candidates.map { seed ->
        seed to candidates.filter { CornerSearchGeometry.consensusAgrees(seed.points, it.points) }
    }.filter { (_, group) -> group.size >= 2 }

    private fun firstInterior(candidates: List<Candidate>) = candidates.firstOrNull {
        !CornerSearchGeometry.touchesFrame(it.points, 0.04f) && CornerSearchGeometry.area(it.points) >= 0.15f
    }

    private fun confirmed(seed: Candidate, group: List<Candidate>, raw: Candidate?, still: Boolean): Candidate {
        if (!still) {
            val other = group.first { it !== seed }
            previewContexts = seed.context to other.context
        }
        return seed.copy(score = group.minOf { it.score }, needsReview = still && raw == null)
    }

    private fun contextCandidate(bgr: Mat, fraction: Float) =
        if (fraction == 0f) candidate(bgr) else paddedCandidate(bgr, fraction)

    private fun candidate(bgr: Mat, window: CornerSearchWindow = CornerSearchWindow(), context: Float = 0f): Candidate? {
        val (heatmap, shape) = model.run(bgr, 256, 256, rgb = false)
        require(shape.size == 4 && shape[0] == 1L && shape[1] == 4L) { "边框模型输出定义不匹配" }
        val height = shape[2].toInt()
        val width = shape[3].toInt()
        require(width > 0 && height > 0 && heatmap.size == width * height * 4)
        val peaks = (0..3).map { c -> (0 until width * height).maxOf { heatmap[c * width * height + it] } }
        if (peaks.min() < 0.5f) return null
        val points = (0..3).mapNotNull { c -> centroid(heatmap, c * width * height, width, height) }
            .map(window::toSource)
        if (!CornerSearchGeometry.valid(points)) return null
        val bounded = points.map { it.clamped() }
        if (!CornerSearchGeometry.valid(bounded)) return null
        val uncertainty = (0..3).map { c ->
            var weight = 0.0
            var variance = 0.0
            for (i in 0 until width * height) {
                val value = heatmap[c * width * height + i]
                if (value < 0.3f) continue
                val p = window.toSource(NormalizedPoint((i % width + .5f) / width, (i / width + .5f) / height))
                val distance = p.distanceTo(points[c])
                variance += value * distance * distance
                weight += value
            }
            if (weight > 0) kotlin.math.sqrt(variance / weight).toFloat() else 1f
        }
        return Candidate(bounded, peaks.min().coerceIn(0f, 1f), CornerSearchGeometry.touchesFrame(points),
            context = context, uncertainty = uncertainty)
    }

    private fun paddedCandidate(bgr: Mat, fraction: Float): Candidate? {
        val working = Mat()
        val padded = Mat()
        try {
            val scale = minOf(1.0, 1024.0 / max(bgr.cols(), bgr.rows()))
            // Java OpenCV truncates fractional Size dimensions; round explicitly so input
            // pixels and coordinate transforms reproduce the desktop regression exactly.
            Imgproc.resize(bgr, working, Size(max(1, (bgr.cols() * scale).roundToInt()).toDouble(),
                max(1, (bgr.rows() * scale).roundToInt()).toDouble()))
            val px = max(1, (working.cols() * fraction).roundToInt())
            val py = max(1, (working.rows() * fraction).roundToInt())
            Core.copyMakeBorder(working, padded, py, py, px, px, Core.BORDER_CONSTANT, borderMedian(working))
            return candidate(padded, CornerSearchWindow(-px.toFloat() / working.cols(), -py.toFloat() / working.rows(),
                padded.cols().toFloat() / working.cols(), padded.rows().toFloat() / working.rows()), context = fraction)
        } finally { working.release(); padded.release() }
    }

    private fun borderMedian(image: Mat): Scalar {
        val samples = Array(3) { mutableListOf<Double>() }
        for (i in 0 until 32) {
            val x = i * (image.cols() - 1) / 31
            val y = i * (image.rows() - 1) / 31
            for (p in listOf(image.get(0, x), image.get(image.rows() - 1, x),
                image.get(y, 0), image.get(y, image.cols() - 1))) {
                for (c in 0..2) samples[c].add(p[c])
            }
        }
        return Scalar(DoubleArray(3) { c -> samples[c].sorted()[samples[c].size / 2] })
    }

    private fun cropCandidate(bgr: Mat, points: List<NormalizedPoint>): Candidate? {
        val x0 = points.minOf { it.x } * bgr.cols(); val x1 = points.maxOf { it.x } * bgr.cols()
        val y0 = points.minOf { it.y } * bgr.rows(); val y1 = points.maxOf { it.y } * bgr.rows()
        if (minOf(x1 - x0, y1 - y0) < 12f) return null
        val mx = max(20f, (x1 - x0) * 0.25f); val my = max(20f, (y1 - y0) * 0.25f)
        val left = floor(x0 - mx).toInt().coerceAtLeast(0)
        val top = floor(y0 - my).toInt().coerceAtLeast(0)
        val right = ceil(x1 + mx).toInt().coerceAtMost(bgr.cols())
        val bottom = ceil(y1 + my).toInt().coerceAtMost(bgr.rows())
        if (right <= left || bottom <= top) return null
        val crop = bgr.submat(Rect(left, top, right - left, bottom - top))
        return try {
            val result = candidate(crop, CornerSearchWindow(left.toFloat() / bgr.cols(), top.toFloat() / bgr.rows(),
                crop.cols().toFloat() / bgr.cols(), crop.rows().toFloat() / bgr.rows())) ?: return null
            // A prediction following an artificial crop boundary is not evidence of a document edge.
            if (result.points.any { (left > 0 && it.x * bgr.cols() < left + 2) ||
                    (right < bgr.cols() && it.x * bgr.cols() > right - 2) ||
                    (top > 0 && it.y * bgr.rows() < top + 2) ||
                    (bottom < bgr.rows() && it.y * bgr.rows() > bottom - 2) }) null else result
        } finally { crop.release() }
    }

    private fun centroid(values: FloatArray, offset: Int, width: Int, height: Int): NormalizedPoint? {
        val mask = Mat(height, width, CvType.CV_8UC1)
        val labels = Mat()
        val stats = Mat()
        val centers = Mat()
        try {
            mask.put(0, 0, ByteArray(width * height) { if (values[offset + it] >= 0.3f) 255.toByte() else 0 })
            val count = Imgproc.connectedComponentsWithStats(mask, labels, stats, centers)
            val component = (1 until count).maxByOrNull { stats.get(it, Imgproc.CC_STAT_AREA)[0] } ?: return null
            return NormalizedPoint(((centers.get(component, 0)[0] + 0.5) / width).toFloat(),
                ((centers.get(component, 1)[0] + 0.5) / height).toFloat())
        } finally {
            listOf(mask, labels, stats, centers).forEach { it.release() }
        }
    }

    /** Quality gate only. This intentionally never moves a model-predicted corner. */
    private fun boundaryEvidence(image: Mat, points: List<NormalizedPoint>): List<Float> {
        if (points.size != 4 || image.empty()) return List(4) { 0f }
        val gray = Mat()
        return try {
            Imgproc.cvtColor(image, gray, Imgproc.COLOR_BGR2GRAY)
            val scale = minOf(1.0, 512.0 / max(gray.cols(), gray.rows()))
            val working = Mat()
            try {
                if (scale < 1.0) Imgproc.resize(gray, working, Size(), scale, scale, Imgproc.INTER_AREA)
                else gray.copyTo(working)
                val width = working.cols().toFloat()
                val height = working.rows().toFloat()
                val radius = max(2.0f, minOf(width, height) * 0.012f)
                val samples = 20
                val edgeEvidence = MutableList(4) { 0f }
                for (edge in 0 until 4) {
                    val start = points[edge]
                    val end = points[(edge + 1) % 4]
                    val ax = start.x * width; val ay = start.y * height
                    val bx = end.x * width; val by = end.y * height
                    val dx = bx - ax; val dy = by - ay
                    val length = kotlin.math.hypot(dx, dy)
                    if (length < 12f) continue
                    val nx = -dy / length; val ny = dx / length
                    var supported = 0
                    for (index in 1..samples) {
                        val t = index.toFloat() / (samples + 1)
                        val x = ax + dx * t; val y = ay + dy * t
                        val outside = sampleGray(working, x - nx * radius, y - ny * radius)
                        val inside = sampleGray(working, x + nx * radius, y + ny * radius)
                        if (abs(inside - outside) >= 7f) supported++
                    }
                    edgeEvidence[edge] = supported.toFloat() / samples
                }
                edgeEvidence
            } finally { working.release() }
        } finally { gray.release() }
    }

    private fun sampleGray(image: Mat, x: Float, y: Float): Float {
        val px = x.roundToInt().coerceIn(0, image.cols() - 1)
        val py = y.roundToInt().coerceIn(0, image.rows() - 1)
        return image.get(py, px)[0].toFloat()
    }

    override fun close() = model.close()
}

data class DewarpGrid(val values: FloatArray, val columns: Int, val rows: Int) {
    init {
        require(columns >= 2 && rows >= 2 && values.size == columns * rows * 2)
    }

    /** UVDoc and torch grid_sample use align_corners=true, not resize's half-pixel rule. */
    fun sample(u: Float, v: Float): NormalizedPoint {
        val x = u.coerceIn(0f, 1f) * (columns - 1)
        val y = v.coerceIn(0f, 1f) * (rows - 1)
        val left = x.toInt().coerceAtMost(columns - 2)
        val top = y.toInt().coerceAtMost(rows - 2)
        val tx = x - left; val ty = y - top
        fun channel(c: Int): Float {
            val base = c * rows * columns + top * columns + left
            val a = values[base] * (1 - tx) + values[base + 1] * tx
            val b = values[base + columns] * (1 - tx) + values[base + columns + 1] * tx
            return ((a * (1 - ty) + b * ty) + 1) / 2
        }
        return NormalizedPoint(channel(0), channel(1))
    }

    fun validate() {
        require(values.all { it.isFinite() && it in -1.15f..1.15f }) { "展平网格越界，请关闭展平或重新拍摄" }
        for (row in 0 until rows - 1) for (column in 0 until columns - 1) {
            val a = sample(column.toFloat() / (columns - 1), row.toFloat() / (rows - 1))
            val b = sample((column + 1f) / (columns - 1), row.toFloat() / (rows - 1))
            val c = sample(column.toFloat() / (columns - 1), (row + 1f) / (rows - 1))
            val d = sample((column + 1f) / (columns - 1), (row + 1f) / (rows - 1))
            require((b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x) > 0 &&
                (d.x - c.x) * (d.y - b.y) - (d.y - c.y) * (d.x - b.x) > 0) {
                "展平模型产生折叠，请关闭展平或重新调整边框"
            }
        }
    }
}

class OnnxDewarper(private val model: OnnxImageModel) : AutoCloseable {
    fun predict(bgr: Mat): DewarpGrid {
        val (values, shape) = model.run(bgr, 488, 712, rgb = true)
        require(shape.contentEquals(longArrayOf(1, 2, 45, 31))) { "展平模型输出定义不匹配" }
        return DewarpGrid(values, columns = 31, rows = 45).also { it.validate() }
    }

    override fun close() = model.close()
}
