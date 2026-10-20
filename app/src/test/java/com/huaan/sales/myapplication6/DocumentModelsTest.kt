package com.huaan.sales.myapplication6

import com.huaan.sales.myapplication6.domain.DocumentDetection
import com.huaan.sales.myapplication6.domain.DocumentGeometry
import com.huaan.sales.myapplication6.domain.NormalizedPoint
import com.huaan.sales.myapplication6.domain.QualityIssue
import com.huaan.sales.myapplication6.vision.DetectionSmoother
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class DocumentModelsTest {
    private val corners = listOf(
        NormalizedPoint(0.1f, 0.2f),
        NormalizedPoint(0.9f, 0.2f),
        NormalizedPoint(0.9f, 0.8f),
        NormalizedPoint(0.1f, 0.8f),
    )

    @Test
    fun geometryBuildsClockwiseEdgeMidpoints() {
        val geometry = DocumentGeometry.fromCorners(corners)

        assertEquals(NormalizedPoint(0.5f, 0.2f), geometry.edgeMidpoints[0])
        assertEquals(NormalizedPoint(0.9f, 0.5f), geometry.edgeMidpoints[1])
        assertEquals(NormalizedPoint(0.5f, 0.8f), geometry.edgeMidpoints[2])
        assertEquals(NormalizedPoint(0.1f, 0.5f), geometry.edgeMidpoints[3])
    }

    @Test
    fun editingControlPointClampsCoordinatesAndMarksManualAdjustment() {
        val geometry = DocumentGeometry.fromCorners(corners)
            .withControlPoint(5, NormalizedPoint(1.4f, -0.2f))

        assertEquals(NormalizedPoint(1f, 0f), geometry.corners[1])
        assertEquals(NormalizedPoint(1f, 0.3f), geometry.corners[2])
        assertEquals(1f, geometry.edgeMidpoints[1].x, 0.0001f)
        assertEquals(0.15f, geometry.edgeMidpoints[1].y, 0.0001f)
        assertTrue(geometry.manuallyAdjusted)
    }

    @Test
    fun draggingCornerKeepsAdjacentStraightEdgesStraight() {
        val geometry = DocumentGeometry.fromCorners(corners)
            .withControlPoint(0, NormalizedPoint(0.3f, 0.4f))

        assertEquals(0.6f, geometry.edgeMidpoints[0].x, 0.0001f)
        assertEquals(0.3f, geometry.edgeMidpoints[0].y, 0.0001f)
        assertEquals(0.2f, geometry.edgeMidpoints[3].x, 0.0001f)
        assertEquals(0.6f, geometry.edgeMidpoints[3].y, 0.0001f)
        assertEquals(NormalizedPoint(0.9f, 0.5f), geometry.edgeMidpoints[1])
        assertEquals(NormalizedPoint(0.5f, 0.8f), geometry.edgeMidpoints[2])
        assertTrue(geometry.manuallyAdjusted)
    }

    @Test
    fun draggingEdgeMovesBothCornersAndStaysStraight() {
        val geometry = DocumentGeometry.fromCorners(corners)
            .withControlPoint(4, NormalizedPoint(0.5f, 0.12f))

        assertEquals(NormalizedPoint(0.1f, 0.12f), geometry.corners[0])
        assertEquals(NormalizedPoint(0.9f, 0.12f), geometry.corners[1])
        val straightMidpointY = (geometry.corners[0].y + geometry.corners[1].y) / 2f
        assertEquals(0.5f, geometry.edgeMidpoints[0].x, 0.0001f)
        assertEquals(0f, geometry.edgeMidpoints[0].y - straightMidpointY, 0.0001f)
    }

    @Test
    fun rotatingLeftUpdatesTheSourceRotationAndCornerCoordinates() {
        val rotated = DocumentGeometry.fromCorners(corners).rotatedLeft()

        assertEquals(270, rotated.sourceRotationDegrees)
        assertPointsEqual(listOf(NormalizedPoint(0.2f, 0.1f), NormalizedPoint(0.8f, 0.1f),
            NormalizedPoint(0.8f, 0.9f), NormalizedPoint(0.2f, 0.9f)), rotated.corners)
    }

    @Test
    fun fourLeftRotationsRestoreCornerCoordinatesButPreserveManualEdit() {
        val original = DocumentGeometry.fromCorners(corners)
        val rotated = original.rotatedLeft().rotatedLeft().rotatedLeft().rotatedLeft()

        assertPointsEqual(original.corners, rotated.corners)
        assertPointsEqual(original.edgeMidpoints, rotated.edgeMidpoints)
        assertEquals(0, rotated.sourceRotationDegrees)
        assertTrue(rotated.manuallyAdjusted)
    }

    @Test
    fun detectionRequiresCornersAndConfidence() {
        val usable = detection(corners = corners, confidence = 0.8f)
        val lowConfidence = detection(corners = corners, confidence = 0.4f)
        val weakBoundary = detection(corners = corners, confidence = 0.9f,
            issues = setOf(QualityIssue.LOW_CONFIDENCE))
        val missing = detection(corners = emptyList(), confidence = 0.9f, issues = setOf(QualityIssue.NO_DOCUMENT))

        assertTrue(usable.isUsable)
        assertFalse(lowConfidence.isUsable)
        assertFalse(weakBoundary.isUsable)
        assertFalse(missing.isUsable)
    }

    @Test
    fun smootherBlendsStableFourCornerDetections() {
        val smoother = DetectionSmoother(alpha = 0.5f)
        smoother.update(detection(corners = corners, confidence = 0.8f), 0)
        smoother.update(detection(corners = corners, confidence = 0.8f), 500)
        val shifted = corners.map { it.copy(x = it.x + 0.02f) }

        val result = smoother.update(detection(corners = shifted, confidence = 1f), 1000)

        assertTrue(result.corners[0].x > 0.11f && result.corners[0].x < shifted[0].x)
        assertEquals(1f, result.confidence, 0.0001f)
    }

    private fun detection(
        corners: List<NormalizedPoint>,
        confidence: Float,
        issues: Set<QualityIssue> = emptySet(),
    ) = DocumentDetection(
        contour = corners,
        corners = corners,
        confidence = confidence,
        qualityIssues = issues,
        rotationDegrees = 0,
        sourceWidth = 1000,
        sourceHeight = 800,
        inferenceMillis = 10,
    )

    private fun assertPointsEqual(expected: List<NormalizedPoint>, actual: List<NormalizedPoint>) {
        assertEquals(expected.size, actual.size)
        expected.zip(actual).forEach { (a, b) ->
            assertEquals(a.x, b.x, 0.000001f)
            assertEquals(a.y, b.y, 0.000001f)
        }
    }
}
