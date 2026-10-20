package com.huaan.sales.myapplication6

import com.huaan.sales.myapplication6.domain.*
import com.huaan.sales.myapplication6.ml.CornerSearchGeometry
import com.huaan.sales.myapplication6.ml.CornerSearchWindow
import org.junit.Assert.*
import org.junit.Test

class CornerSearchTest {
    private val small = listOf(NormalizedPoint(.4f, .4f), NormalizedPoint(.6f, .4f),
        NormalizedPoint(.6f, .55f), NormalizedPoint(.4f, .55f))

    @Test fun clearSmallDocumentIsNotRejectedBySixPercentRule() {
        assertTrue(CornerSearchGeometry.area(small) < .06f)
        assertTrue(CornerSearchGeometry.valid(small))
        assertFalse(CornerSearchGeometry.valid(listOf(small[0], small[2], small[1], small[3])))
        assertFalse(CornerSearchGeometry.valid(small.map { it.copy(x = Float.NaN) }))
    }

    @Test fun cropAndPaddingCoordinatesMapBackToOriginal() {
        val crop = CornerSearchWindow(.25f, .1f, .5f, .4f)
        val p = crop.toSource(NormalizedPoint(.5f, .5f))
        assertEquals(.5f, p.x, .00001f)
        assertEquals(.3f, p.y, .00001f)
        val padded = CornerSearchWindow(-.15f, -.15f, 1.3f, 1.3f)
        assertEquals(0f, padded.toSource(NormalizedPoint(.15f / 1.3f, .15f / 1.3f)).x, .00001f)
        assertTrue(CornerSearchGeometry.touchesFrame(listOf(NormalizedPoint(-.01f, .5f))))
    }

    @Test fun disagreementBetweenScalesIsNotSilentlyAccepted() {
        assertTrue(CornerSearchGeometry.agrees(small, small.map { it.copy(x = it.x + .01f) }))
        assertFalse(CornerSearchGeometry.agrees(small, small.map { it.copy(x = it.x + .25f) }))
    }

    @Test fun aSingleCornerOnAnotherPaperLayerIsNotWholePageConsensus() {
        val page = listOf(NormalizedPoint(.15f, .15f), NormalizedPoint(.8f, .16f),
            NormalizedPoint(.9f, .85f), NormalizedPoint(.2f, .9f))
        val mixed = page.toMutableList().apply { this[0] = NormalizedPoint(.25f, .16f) }
        assertTrue(CornerSearchGeometry.agrees(page, mixed))
        assertFalse(CornerSearchGeometry.consensusAgrees(page, mixed))
        assertTrue(CornerSearchGeometry.consensusAgrees(page, page.map { it.copy(x = it.x + .005f) }))
    }

    @Test fun highConfidenceClippedFrameStillRequiresExplicitConfirmation() {
        val draft = DraftPage("id", "original", "preview", DocumentGeometry.fromCorners(small),
            ProcessingOptions(), .9f, 0, "test", false, setOf(QualityIssue.EDGE_CLIPPED))
        assertTrue(draft.requiresBoundaryConfirmation)
        assertFalse(draft.copy(boundaryConfirmed = true).requiresBoundaryConfirmation)
        assertFalse(draft.copy(geometry = draft.geometry.withControlPoint(0, small[0]),
            boundaryEditedByUser = true).requiresBoundaryConfirmation)
        assertTrue(draft.copy(detectionConfidence = .3f, detectionIssues = emptySet()).requiresBoundaryConfirmation)
        assertTrue(draft.copy(geometry = draft.geometry.rotatedLeft()).requiresBoundaryConfirmation)
    }

    @Test fun glareAndOverexposureRequireBoundaryConfirmation() {
        val geometry = DocumentGeometry.fromCorners(small)
        val glare = DraftPage("glare", "original", "preview", geometry,
            ProcessingOptions(), .9f, 0, "test", false, setOf(QualityIssue.GLARE))
        val bright = glare.copy(id = "bright", detectionIssues = setOf(QualityIssue.TOO_BRIGHT))
        assertTrue(glare.requiresBoundaryConfirmation)
        assertFalse(bright.requiresBoundaryConfirmation)
        assertFalse(glare.copy(boundaryConfirmed = true).requiresBoundaryConfirmation)
    }

    @Test fun weakBoundaryEvidenceRequiresConfirmationEvenWithHighModelScore() {
        val draft = DraftPage("weak", "original", "preview", DocumentGeometry.fromCorners(small),
            ProcessingOptions(), .9f, 0, "test", false, setOf(QualityIssue.LOW_CONFIDENCE))
        assertTrue(draft.requiresBoundaryConfirmation)
        assertFalse(draft.copy(boundaryConfirmed = true).requiresBoundaryConfirmation)
    }
}
