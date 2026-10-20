package com.huaan.sales.myapplication6.domain

import java.time.Instant
import java.time.ZoneId
import java.time.format.DateTimeFormatter
import kotlin.math.hypot

private val DocumentTitleFormatter = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm")

fun documentTitle(appName: String, createdAtMillis: Long): String {
    val time = Instant.ofEpochMilli(createdAtMillis)
        .atZone(ZoneId.systemDefault())
        .format(DocumentTitleFormatter)
    return "$appName $time"
}

data class NormalizedPoint(
    val x: Float,
    val y: Float,
) {
    fun clamped(): NormalizedPoint = NormalizedPoint(x.coerceIn(0f, 1f), y.coerceIn(0f, 1f))

    fun distanceTo(other: NormalizedPoint): Float = hypot(x - other.x, y - other.y)
}

enum class QualityIssue {
    NO_DOCUMENT,
    LOW_CONFIDENCE,
    TOO_BLURRY,
    TOO_DARK,
    TOO_BRIGHT,
    GLARE,
    EDGE_CLIPPED,
    TOO_SMALL,
}

enum class DetectionSource { OPENCV, LCNET, FASTVIT, SEGMENTATION }

data class DocumentDetection(
    val contour: List<NormalizedPoint>,
    val corners: List<NormalizedPoint>,
    val confidence: Float,
    val qualityIssues: Set<QualityIssue>,
    val rotationDegrees: Int,
    val sourceWidth: Int,
    val sourceHeight: Int,
    val inferenceMillis: Long,
    val edgeEvidence: List<Float> = emptyList(),
    // Normalized radius of the connected heatmap region, not a calibrated probability.
    val cornerUncertainty: List<Float> = emptyList(),
    val detectionSource: DetectionSource = DetectionSource.OPENCV,
    val requiresConfirmation: Boolean = false,
    val invisibleEdges: Set<Int> = emptySet(),
    val outOfFrame: Boolean = false,
) {
    val isUsable: Boolean
        get() = corners.size == 4 && confidence >= 0.55f && !requiresConfirmation && !outOfFrame &&
            QualityIssue.NO_DOCUMENT !in qualityIssues &&
            QualityIssue.LOW_CONFIDENCE !in qualityIssues &&
            QualityIssue.EDGE_CLIPPED !in qualityIssues
}

data class DocumentGeometry(
    val corners: List<NormalizedPoint>,
    val edgeMidpoints: List<NormalizedPoint>,
    val perspectiveMatrix: FloatArray? = null,
    val backwardGrid: List<NormalizedPoint>? = null,
    val manuallyAdjusted: Boolean = false,
    // Rotation relative to the EXIF-oriented original; the original pixels stay untouched.
    val sourceRotationDegrees: Int = 0,
) {
    init {
        require(corners.size == 4) { "DocumentGeometry requires four corners" }
        require(edgeMidpoints.size == 4) { "DocumentGeometry requires four edge midpoints" }
        require(sourceRotationDegrees in setOf(0, 90, 180, 270))
    }

    fun withControlPoint(index: Int, point: NormalizedPoint): DocumentGeometry {
        require(index in 0..7) { "Control point index must be between 0 and 7" }
        val bounded = point.clamped()
        return if (index < 4) {
            val deltaX = (bounded.x - corners[index].x) / 2f
            val deltaY = (bounded.y - corners[index].y) / 2f
            val adjacentEdges = setOf(index, (index + 3) % 4)
            copy(
                corners = corners.toMutableList().also { it[index] = bounded },
                edgeMidpoints = edgeMidpoints.mapIndexed { edgeIndex, midpoint ->
                    if (edgeIndex in adjacentEdges) {
                        NormalizedPoint(midpoint.x + deltaX, midpoint.y + deltaY).clamped()
                    } else {
                        midpoint
                    }
                },
                manuallyAdjusted = true,
            )
        } else {
            val edge = index - 4
            val next = (edge + 1) % 4
            val midpoint = edgeMidpoints[edge]
            val deltaX = bounded.x - midpoint.x
            val deltaY = bounded.y - midpoint.y
            val movedCorners = corners.toMutableList()
            movedCorners[edge] = NormalizedPoint(corners[edge].x + deltaX, corners[edge].y + deltaY).clamped()
            movedCorners[next] = NormalizedPoint(corners[next].x + deltaX, corners[next].y + deltaY).clamped()
            val movedMidpoints = edgeMidpoints.toMutableList()
            movedMidpoints[edge] = NormalizedPoint(
                (movedCorners[edge].x + movedCorners[next].x) / 2f,
                (movedCorners[edge].y + movedCorners[next].y) / 2f,
            )
            fun shiftAdjacentMidpoint(edgeIndex: Int, cornerIndex: Int) {
                val current = movedMidpoints[edgeIndex]
                movedMidpoints[edgeIndex] = NormalizedPoint(
                    current.x + (movedCorners[cornerIndex].x - corners[cornerIndex].x) / 2f,
                    current.y + (movedCorners[cornerIndex].y - corners[cornerIndex].y) / 2f,
                ).clamped()
            }
            shiftAdjacentMidpoint((edge + 3) % 4, edge)
            shiftAdjacentMidpoint(next, next)
            copy(corners = movedCorners, edgeMidpoints = movedMidpoints, manuallyAdjusted = true)
        }
    }

    companion object {
        fun fromCorners(corners: List<NormalizedPoint>): DocumentGeometry {
            require(corners.size == 4)
            val edges = corners.indices.map { index ->
                val next = corners[(index + 1) % corners.size]
                NormalizedPoint(
                    x = (corners[index].x + next.x) / 2f,
                    y = (corners[index].y + next.y) / 2f,
                )
            }
            return DocumentGeometry(corners = corners, edgeMidpoints = edges)
        }
    }

    fun rotatedLeft(): DocumentGeometry {
        fun NormalizedPoint.turnLeft() = NormalizedPoint(x = y, y = 1f - x)
        fun <T> List<T>.cycleForward() = listOf(this[1], this[2], this[3], this[0])
        return copy(
            corners = corners.map { it.turnLeft() }.cycleForward(),
            edgeMidpoints = edgeMidpoints.map { it.turnLeft() }.cycleForward(),
            perspectiveMatrix = null,
            backwardGrid = null,
            manuallyAdjusted = true,
            sourceRotationDegrees = (sourceRotationDegrees + 270) % 360,
        )
    }

    /**
     * Keep the corrected page in the same orientation as the captured photo.
     * Corner order decides which edge becomes the output width; a 90° relabel
     * swaps that choice without changing the detected quadrilateral.
     */
    fun matchedToCaptureOrientation(imageWidth: Int, imageHeight: Int): DocumentGeometry {
        if (imageWidth <= 1 || imageHeight <= 1) return this
        val captureLandscape = imageWidth >= imageHeight
        if (outputLandscape(corners, imageWidth, imageHeight) == captureLandscape) return this
        val turned = listOf(
            listOf(corners[3], corners[0], corners[1], corners[2]),
            listOf(corners[1], corners[2], corners[3], corners[0]),
        ).minBy { (it[0].y + it[1].y) / 2f }
        return fromCorners(turned)
    }

    private fun outputLandscape(corners: List<NormalizedPoint>, imageWidth: Int, imageHeight: Int): Boolean {
        fun length(a: NormalizedPoint, b: NormalizedPoint): Double {
            val dx = (a.x - b.x).toDouble() * (imageWidth - 1)
            val dy = (a.y - b.y).toDouble() * (imageHeight - 1)
            return hypot(dx, dy)
        }
        val width = maxOf(length(corners[0], corners[1]), length(corners[3], corners[2]))
        val height = maxOf(length(corners[0], corners[3]), length(corners[1], corners[2]))
        return width >= height
    }
}

enum class ColorMode { COLOR, GRAYSCALE, BLACK_AND_WHITE }

enum class PageEffect {
    ORIGINAL,
    BRIGHTEN,
    SHARPEN,
    REMOVE_SHADOW,
    MONOCHROME,
    GRAYSCALE,
    INVERT,
}

data class ProcessingOptions(
    val dewarpEnabled: Boolean = false,
    val dewarpStrength: Float = 0.5f,
    val removeShadows: Boolean = true,
    val colorMode: ColorMode = ColorMode.COLOR,
    val jpegQuality: Int = 94,
    val effect: PageEffect = PageEffect.ORIGINAL,
    // Applied only when the user completes an edit, never when importing/listing pages.
    val cropEnabled: Boolean = true,
    // False keeps a rectangular bounding crop; true rectifies the page perspective.
    val perspectiveEnabled: Boolean = true,
) {
    fun withEffect(effect: PageEffect): ProcessingOptions = when (effect) {
        PageEffect.ORIGINAL -> copy(effect = effect, colorMode = ColorMode.COLOR, removeShadows = false, dewarpEnabled = false)
        PageEffect.BRIGHTEN, PageEffect.SHARPEN, PageEffect.INVERT ->
            copy(effect = effect, colorMode = ColorMode.COLOR, removeShadows = false)
        PageEffect.REMOVE_SHADOW ->
            copy(effect = effect, colorMode = ColorMode.COLOR, removeShadows = true)
        PageEffect.MONOCHROME -> copy(effect = effect, colorMode = ColorMode.BLACK_AND_WHITE, removeShadows = false)
        PageEffect.GRAYSCALE -> copy(effect = effect, colorMode = ColorMode.GRAYSCALE, removeShadows = false)
    }

    fun displayEffect(): PageEffect = when {
        effect != PageEffect.ORIGINAL -> effect
        colorMode == ColorMode.GRAYSCALE -> PageEffect.GRAYSCALE
        colorMode == ColorMode.BLACK_AND_WHITE -> PageEffect.MONOCHROME
        removeShadows -> PageEffect.REMOVE_SHADOW
        else -> PageEffect.ORIGINAL
    }
}

data class CorrectionResult(
    val imagePath: String,
    val pageId: String,
    val width: Int,
    val height: Int,
    val processingMillis: Long,
    val modelVersion: String,
    val manuallyAdjusted: Boolean,
)

data class ScanDocument(
    val id: String,
    val createdAtMillis: Long,
    val pageCount: Int,
    val coverPath: String,
)

data class ScanPage(
    val id: String,
    val originalPath: String,
    val correctedPath: String,
    val width: Int,
    val height: Int,
    val createdAtMillis: Long,
    val processingMillis: Long,
    val modelVersion: String,
    val manuallyAdjusted: Boolean,
    val geometry: DocumentGeometry,
    val options: ProcessingOptions,
    val documentId: String = LEGACY_DOCUMENT_ID,
    val processed: Boolean = true,
)

const val LEGACY_DOCUMENT_ID = "legacy"

data class DraftPage(
    val id: String,
    val originalPath: String,
    val previewPath: String,
    val geometry: DocumentGeometry,
    val options: ProcessingOptions,
    val detectionConfidence: Float,
    val processingMillis: Long,
    val modelVersion: String,
    val hasProcessedPreview: Boolean,
    val detectionIssues: Set<QualityIssue> = emptySet(),
    val boundaryConfirmed: Boolean = false,
    // Full-frame effect image is distinct from the geometrically corrected save output.
    val effectPreviewPath: String? = null,
    // A small, disposable editor preview must never be persisted as the final image.
    val cropPreviewPath: String? = null,
    val boundaryDetectionPending: Boolean = false,
    val boundaryDetectionComplete: Boolean = false,
    val boundaryEditRevision: Int = 0,
    val boundaryEditedByUser: Boolean = false,
) {
    val requiresBoundaryConfirmation: Boolean
        get() = (detectionConfidence < 0.55f ||
            QualityIssue.NO_DOCUMENT in detectionIssues ||
            QualityIssue.LOW_CONFIDENCE in detectionIssues ||
            QualityIssue.EDGE_CLIPPED in detectionIssues ||
            QualityIssue.GLARE in detectionIssues) &&
            !boundaryEditedByUser && !boundaryConfirmed
}
