package com.huaan.sales.myapplication6.vision

import com.huaan.sales.myapplication6.domain.DocumentDetection
import com.huaan.sales.myapplication6.domain.QualityIssue
import org.opencv.core.Mat

fun interface StillDocumentDetector {
    /** EXIF-oriented BGR image. No pixels or user edits are retained by this interface. */
    fun detect(image: Mat): DocumentDetection
}

data class StillDetectionComparison(val baseline: DocumentDetection, val candidate: DocumentDetection?,
    val refinedBaseline: DocumentDetection, val refinedCandidate: DocumentDetection?, val candidateFailure: String?)

class StillDocumentPipeline(
    private val baseline: StillDocumentDetector,
    private val candidate: StillDocumentDetector? = null,
    private val candidateApproved: Boolean = false,
    private val refiner: BoundaryRefiner = BoundaryRefiner(),
) : StillDocumentDetector {
    override fun detect(image: Mat): DocumentDetection {
        val started = System.nanoTime()
        val selected = if (candidateApproved && candidate != null) {
            runCatching { candidate.detect(image) }.getOrElse {
                baseline.detect(image).let { result -> result.copy(requiresConfirmation = true,
                    qualityIssues = result.qualityIssues + QualityIssue.LOW_CONFIDENCE) }
            }
        } else baseline.detect(image)
        return refiner.refine(image, selected).copy(inferenceMillis = (System.nanoTime() - started) / 1_000_000)
    }

    /** Explicit benchmark path; never doubles inference cost during ordinary capture. */
    fun compare(image: Mat): StillDetectionComparison {
        val old = baseline.detect(image)
        val next = runCatching { candidate?.detect(image) }
        return StillDetectionComparison(old, next.getOrNull(), refiner.refine(image, old),
            next.getOrNull()?.let { refiner.refine(image, it) }, next.exceptionOrNull()?.message)
    }
}
