package com.huaan.sales.myapplication6

import com.huaan.sales.myapplication6.domain.*
import com.huaan.sales.myapplication6.vision.DetectionSmoother
import org.junit.Assert.*
import org.junit.Test

class DetectionSmootherTest {
    private fun page(x: Float = .10f) = DocumentDetection(
        listOf(NormalizedPoint(x, .1f), NormalizedPoint(x + .5f, .1f),
            NormalizedPoint(x + .5f, .8f), NormalizedPoint(x, .8f)),
        listOf(NormalizedPoint(x, .1f), NormalizedPoint(x + .5f, .1f),
            NormalizedPoint(x + .5f, .8f), NormalizedPoint(x, .8f)),
        .9f, emptySet(), 0, 800, 600, 20)

    @Test fun firstFrameNeedsConfirmationAndSmallJitterStaysStill() {
        val filter = DetectionSmoother()
        assertFalse(filter.update(page(), 0).isUsable)
        val stable = filter.update(page(), 500)
        assertTrue(stable.isUsable)
        assertEquals(stable.corners, filter.update(page(.102f), 1000).corners)
    }

    @Test fun isolatedWrongRectangleDoesNotMoveTheFrame() {
        val filter = DetectionSmoother()
        filter.update(page(), 0)
        filter.update(page(), 500)
        val wrong = filter.update(page(.4f), 1000)
        assertEquals(page().corners, wrong.corners)
        assertFalse(wrong.isUsable)
        assertTrue(filter.update(page(), 1500).isUsable)
    }

    @Test fun moderateMovementTracksWithinTwoThreeHundredMillisecondUpdates() {
        val filter = DetectionSmoother()
        filter.update(page(), 0)
        filter.update(page(), 300)
        val moving = page(.13f)
        val first = filter.update(moving, 600)
        assertTrue(first.corners[0].distanceTo(moving.corners[0]) < .01f)
        val second = filter.update(moving, 900)
        assertTrue(second.corners[0].distanceTo(moving.corners[0]) < .006f)
        assertTrue(second.isUsable)
    }

    @Test fun persistentLargeMotionSwitchesWithoutInventingIntermediateCorners() {
        val filter = DetectionSmoother()
        filter.update(page(), 0)
        filter.update(page(), 500)
        filter.update(page(.4f), 1000)
        val moved = filter.update(page(.4f), 1500)
        assertEquals(page(.4f).corners, moved.corners)
        assertTrue(moved.isUsable)
    }

    @Test fun missedFrameExpiresAndNeverRetainsUsableConfidence() {
        val filter = DetectionSmoother()
        filter.update(page(), 0)
        filter.update(page(), 500)
        val empty = page().copy(corners = emptyList(), contour = emptyList(), confidence = 0f,
            qualityIssues = setOf(QualityIssue.NO_DOCUMENT))
        val held = filter.update(empty, 1000)
        assertEquals(4, held.corners.size)
        assertFalse(held.isUsable)
        assertTrue(filter.update(empty, 1800).corners.isEmpty())
    }

    @Test fun rotationResolutionAndResetCannotReuseOldGeometry() {
        val filter = DetectionSmoother()
        filter.update(page(), 0)
        filter.update(page(), 500)
        assertFalse(filter.update(page().copy(sourceWidth = 600, sourceHeight = 800), 1000).isUsable)
        filter.reset()
        assertFalse(filter.update(page(.4f), 1500).isUsable)
    }
}
