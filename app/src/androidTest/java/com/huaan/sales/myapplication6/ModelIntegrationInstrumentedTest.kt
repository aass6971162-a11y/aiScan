package com.huaan.sales.myapplication6

import android.graphics.BitmapFactory
import android.content.Context
import androidx.test.core.app.ApplicationProvider
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.huaan.sales.myapplication6.ml.*
import com.huaan.sales.myapplication6.vision.*
import com.huaan.sales.myapplication6.domain.ProcessingOptions
import com.huaan.sales.myapplication6.domain.NormalizedPoint
import com.huaan.sales.myapplication6.domain.QualityIssue
import org.junit.Assert.*
import org.junit.BeforeClass
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.abs

@RunWith(AndroidJUnit4::class)
class ModelIntegrationInstrumentedTest {
    private val context: Context = ApplicationProvider.getApplicationContext()
    private val registry = ModelAssetRegistry(context)
    private val testAssets = InstrumentationRegistry.getInstrumentation().context.assets

    @Test
    fun packagedModelsAreVerifiedAndTamperingIsRejected() {
        val status = registry.verify()
        assertTrue(status.missingModelIds.isEmpty())
        assertTrue(status.readyModelIds.containsAll(setOf("document_corners", "document_dewarp")))
        val descriptor = registry.descriptors().first().copy(sha256 = "0".repeat(64))
        assertThrows(IllegalArgumentException::class.java) { registry.copyVerifiedModelToCache(descriptor) }
        assertTrue(registry.verify().missingModelIds.isEmpty())
    }

    @Test
    fun actualCornerGraphMatchesDesktopAndRejectsEmptyFrames() {
        val bgr = fixture("corners_input.png")
        OnnxImageModel(registry.openOnnxSession("document_corners")).use { model ->
            val (actual, shape) = model.run(bgr, 256, 256, rgb = false)
            assertArrayEquals(longArrayOf(1, 4, 128, 128), shape)
            assertClose(expected("corners_expected.f32"), actual, 1e-4f)
            val detector = OnnxCornerDetector(model, OpenCvDocumentDetector())
            val detection = detector.detectColor(bgr)
            assertTrue(detection.isUsable)
            assertEquals(0.094f, detection.corners[0].x, 0.015f)
            assertEquals(0.388f, detection.corners[1].y, 0.015f)
            assertTrue(detection.corners[2].x > detection.corners[3].x)
            for (level in listOf(0.0, 127.0, 255.0)) {
                val blank = Mat(256, 256, CvType.CV_8UC3, Scalar(level, level, level))
                try { assertFalse(detector.detectColor(blank).isUsable) } finally { blank.release() }
            }
            val original = fixture("photo_original.jpg")
            val tableOnly = original.submat(Rect(0, 0, original.cols(), 140))
            try {
                val empty = detector.detectColor(tableOnly)
                assertFalse("Table background must not be accepted as a document", empty.isUsable)
            } finally { tableOnly.release(); original.release() }
        }
        bgr.release()
    }

    @Test
    fun actualDewarpGraphMatchesDesktopNativeGrid() {
        val bgr = fixture("uvdoc_input.png")
        OnnxImageModel(registry.openOnnxSession("document_dewarp")).use { model ->
            val (actual, shape) = model.run(bgr, 488, 712, rgb = true)
            assertArrayEquals(longArrayOf(1, 2, 45, 31), shape)
            assertClose(expected("uvdoc_expected.f32"), actual, 1e-4f)
            DewarpGrid(actual, 31, 45).validate()
        }
        bgr.release()
    }

    @Test
    fun photoEntersOriginalEditorThenUsesAiForCorrection() {
        val root = File(context.cacheDir, "ai-pipeline-${System.nanoTime()}").apply { mkdirs() }
        val photo = File(root, "photo.jpg")
        testAssets.open("model-golden/photo_original.jpg").use { input -> photo.outputStream().use { input.copyTo(it) } }
        OnnxCornerDetector(OnnxImageModel(registry.openOnnxSession("document_corners")), OpenCvDocumentDetector()).use { detector ->
            OnnxDewarper(OnnxImageModel(registry.openOnnxSession("document_dewarp"))).use { dewarper ->
                val processor = DocumentProcessor(root, detector, registry.verify(), dewarper)
                val draft = processor.createDraft(photo)
                assertTrue(draft.detectionConfidence > 0.55f)
                assertFalse(draft.hasProcessedPreview)
                assertFalse(File(draft.previewPath).exists())
                val result = processor.processDraft(draft.copy(options = ProcessingOptions(dewarpEnabled = true)))
                assertTrue(result.hasProcessedPreview)
                val bitmap = BitmapFactory.decodeFile(result.previewPath)
                // The detector sees 256px; correction must retain the 512px photo's detail.
                assertTrue(bitmap.width > 300)
                assertTrue(bitmap.height > 100)
                bitmap.recycle()
            }
        }
        root.deleteRecursively()
    }

    @Test
    fun smallNearAndSlightlyClippedDocumentsRetainMappedCorners() {
        val cases = listOf(
            Triple("distance-small.png", listOf(320f to 240f, 480f to 235f, 485f to 335f, 315f to 340f), false),
            Triple("distance-near.png", listOf(12f to 20f, 785f to 9f, 788f to 580f, 8f to 590f), true),
            Triple("distance-clipped.png", listOf(0f to 20f, 785f to 9f, 788f to 580f, 0f to 590f), true),
        )
        OnnxCornerDetector(OnnxImageModel(registry.openOnnxSession("document_corners")), OpenCvDocumentDetector()).use { detector ->
            cases.forEach { (name, expected, clipped) ->
                val input = fixture(name)
                try {
                    val result = detector.detectColor(input)
                    assertEquals("$name must offer four corners", 4, result.corners.size)
                    expected.forEachIndexed { i, p ->
                        val point = NormalizedPoint(p.first / input.cols(), p.second / input.rows())
                        assertTrue("$name corner $i is inaccurate: ${result.corners[i]}", point.distanceTo(result.corners[i]) < .025f)
                    }
                    assertEquals(clipped, QualityIssue.EDGE_CLIPPED in result.qualityIssues)
                    if (clipped) assertFalse(result.isUsable) else {
                        assertTrue(result.isUsable)
                        assertTrue(QualityIssue.TOO_SMALL in result.qualityIssues)
                    }
                    android.util.Log.i("DistanceRegression", "$name: score=${result.confidence}, corners=${result.corners}, ms=${result.inferenceMillis}")
                } finally { input.release() }
            }
        }
    }

    @Test
    fun interruptedVisibleEdgeKeepsTheSameDocumentCorners() {
        val original = fixture("photo_original.jpg")
        val interrupted = fixture("distance-edge-gap.png")
        try {
            OnnxCornerDetector(OnnxImageModel(registry.openOnnxSession("document_corners")), OpenCvDocumentDetector()).use { detector ->
                val reference = detector.detectColor(original)
                val result = detector.detectColor(interrupted)
                assertTrue(reference.isUsable)
                assertTrue("A small interruption in one edge must not discard the page", result.isUsable)
                assertEquals(4, result.corners.size)
                reference.corners.indices.forEach { i ->
                    assertTrue("Interrupted edge moved corner $i", reference.corners[i].distanceTo(result.corners[i]) < .015f)
                }
            }
        } finally { original.release(); interrupted.release() }
    }

    @Test
    fun stackedLowContrastPagesUseColorContextFallback() {
        val input = fixture("complex-stacked-pages.jpg")
        try {
            assertFalse("OpenCV fast path should reproduce the reported failure",
                OpenCvDocumentDetector().detectColor(input).isUsable)
            OnnxCornerDetector(OnnxImageModel(registry.openOnnxSession("document_corners")), OpenCvDocumentDetector()).use { detector ->
                val preview = detector.detectPreviewColor(input)
                assertTrue("Color AI fallback must recover a live candidate", preview.isUsable)
                assertComplexCorners(preview.corners)

                val still = detector.detectColor(input)
                assertEquals(4, still.corners.size)
                assertComplexCorners(still.corners)
                assertTrue(QualityIssue.LOW_CONFIDENCE in still.qualityIssues)
                assertFalse(QualityIssue.NO_DOCUMENT in still.qualityIssues)
                assertFalse("A context-only recovery must be confirmed before export", still.isUsable)
            }
        } finally { input.release() }
    }

    private fun assertComplexCorners(actual: List<NormalizedPoint>) {
        val expected = listOf(NormalizedPoint(.064f, .132f), NormalizedPoint(.815f, .166f),
            NormalizedPoint(.920f, .838f), NormalizedPoint(.205f, .944f))
        assertEquals(4, actual.size)
        expected.indices.forEach { index ->
            assertTrue("Stacked-page corner $index is inaccurate: ${actual[index]}",
                expected[index].distanceTo(actual[index]) < .035f)
        }
    }

    private fun fixture(name: String): Mat {
        val bitmap = testAssets.open("model-golden/$name").use { BitmapFactory.decodeStream(it) }
        val rgba = Mat(); val bgr = Mat()
        Utils.bitmapToMat(bitmap, rgba)
        Imgproc.cvtColor(rgba, bgr, Imgproc.COLOR_RGBA2BGR)
        bitmap.recycle(); rgba.release()
        return bgr
    }

    private fun expected(name: String): FloatArray {
        val bytes = testAssets.open("model-golden/$name").use { it.readBytes() }
        val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
        return FloatArray(buffer.remaining()).also { buffer.get(it) }
    }

    private fun assertClose(expected: FloatArray, actual: FloatArray, tolerance: Float) {
        assertEquals(expected.size, actual.size)
        val error = actual.indices.maxOf { abs(expected[it] - actual[it]) }
        assertTrue("maximum absolute error $error exceeds $tolerance", error <= tolerance)
        android.util.Log.i("ModelNumerics", "maxAbsError=$error, elements=${actual.size}")
    }

    companion object {
        @JvmStatic @BeforeClass fun initialize() { check(OpenCVLoader.initLocal()) }
    }
}
