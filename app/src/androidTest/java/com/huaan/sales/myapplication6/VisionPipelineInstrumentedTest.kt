package com.huaan.sales.myapplication6

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.exifinterface.media.ExifInterface
import com.huaan.sales.myapplication6.data.PdfExporter
import com.huaan.sales.myapplication6.domain.ColorMode
import com.huaan.sales.myapplication6.domain.DocumentGeometry
import com.huaan.sales.myapplication6.domain.NormalizedPoint
import com.huaan.sales.myapplication6.domain.ProcessingOptions
import com.huaan.sales.myapplication6.domain.ScanPage
import com.huaan.sales.myapplication6.ml.InferenceBackend
import com.huaan.sales.myapplication6.ml.ModelRuntimeStatus
import com.huaan.sales.myapplication6.vision.DocumentProcessor
import com.huaan.sales.myapplication6.vision.OpenCvDocumentDetector
import com.huaan.sales.myapplication6.vision.decodeOrientedBitmap
import com.huaan.sales.myapplication6.vision.decodeOrientedMat
import org.junit.Assert.assertEquals
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.BeforeClass
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.android.OpenCVLoader
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.imgproc.Imgproc
import java.io.File
import java.nio.ByteBuffer
import java.io.FileOutputStream

@RunWith(AndroidJUnit4::class)
class VisionPipelineInstrumentedTest {

    @Test
    fun cameraRgbaPlaneConvertsToOrientedBgrWithoutChannelSwap() {
        val rgba = byteArrayOf(
            10, 20, 30, 255.toByte(),
            40, 50, 60, 255.toByte(),
        )
        val result = com.huaan.sales.myapplication6.camera.rgbaPlaneToOrientedBgr(
            ByteBuffer.wrap(rgba), width = 2, height = 1, rowStride = 8,
            pixelStride = 4, rotationDegrees = 90,
        )
        try {
            assertEquals(2, result.rows())
            assertEquals(1, result.cols())
            assertArrayEquals(doubleArrayOf(30.0, 20.0, 10.0), result.get(0, 0), 0.0)
            assertArrayEquals(doubleArrayOf(60.0, 50.0, 40.0), result.get(1, 0), 0.0)
        } finally { result.release() }
    }

    @Test
    fun rgbaRowsRespectPaddingAndIgnoreAlphaIncludingShortFinalRow() {
        val rgba = byteArrayOf(10, 20, 30, 7, 99, 99, 99, 99, 40, 50, 60, 0)
        for (rotation in listOf(0, 90, 180, 270)) {
            val result = com.huaan.sales.myapplication6.camera.rgbaPlaneToOrientedBgr(
                ByteBuffer.wrap(rgba), 1, 2, 8, 4, rotation)
            try {
                val first = if (rotation == 90 || rotation == 180) 60.0 else 30.0
                assertEquals(first, result.get(0, 0)[0], 0.0)
                assertEquals(if (rotation % 180 == 0) 2 else 1, result.rows())
                assertEquals(if (rotation % 180 == 0) 1 else 2, result.cols())
            } finally { result.release() }
        }
    }
    private val context: Context = ApplicationProvider.getApplicationContext()
    private val detector = OpenCvDocumentDetector()

    @Test
    fun detectsLargeDocumentOnContrastingBackground() {
        val image = Mat.zeros(600, 800, CvType.CV_8UC1)
        Imgproc.rectangle(image, Point(100.0, 80.0), Point(700.0, 520.0), Scalar(245.0), -1)
        Imgproc.rectangle(image, Point(100.0, 80.0), Point(700.0, 520.0), Scalar(255.0), 5)

        val detection = detector.detect(image, 0)

        assertEquals(4, detection.corners.size)
        assertTrue(detection.confidence >= 0.55f)
        image.release()
    }

    @Test
    fun previewRetainsSmallCandidateAndShowsDistanceAdvice() {
        val image = Mat.zeros(600, 800, CvType.CV_8UC1)
        try {
            Imgproc.rectangle(image, Point(310.0, 230.0), Point(490.0, 350.0), Scalar(240.0), -1)
            val result = detector.detect(image, 0)
            assertEquals(4, result.corners.size)
            assertTrue(com.huaan.sales.myapplication6.domain.QualityIssue.TOO_SMALL in result.qualityIssues)
            assertFalse(result.isUsable)
        } finally { image.release() }
    }

    @Test
    fun correctsImageAndExportsReadablePdf() {
        val root = File(context.cacheDir, "vision-test-${System.nanoTime()}").apply { mkdirs() }
        val original = File(root, "original.jpg")
        val corrected = File(root, "corrected.jpg")
        createSyntheticDocument(original)
        val geometry = DocumentGeometry.fromCorners(
            listOf(
                NormalizedPoint(0.12f, 0.10f),
                NormalizedPoint(0.88f, 0.14f),
                NormalizedPoint(0.91f, 0.90f),
                NormalizedPoint(0.09f, 0.86f),
            ),
        )
        val status = ModelRuntimeStatus(
            backend = InferenceBackend.OPENCV_FALLBACK,
            modelVersion = "test-fallback",
            readyModelIds = emptySet(),
            missingModelIds = setOf("document_preview"),
            message = "test",
        )
        val processor = DocumentProcessor(root, detector, status)

        val result = processor.process(
            pageId = "test-page",
            original = original,
            output = corrected,
            geometry = geometry,
            options = ProcessingOptions(
                dewarpEnabled = false,
                dewarpStrength = 0.7f,
                removeShadows = true,
                colorMode = ColorMode.COLOR,
            ),
        )

        assertTrue(corrected.length() > 1_000)
        assertTrue(result.width > 500)
        assertTrue(result.height > 400)
        val page = ScanPage(
            id = result.pageId,
            originalPath = original.absolutePath,
            correctedPath = corrected.absolutePath,
            width = result.width,
            height = result.height,
            createdAtMillis = System.currentTimeMillis(),
            processingMillis = result.processingMillis,
            modelVersion = result.modelVersion,
            manuallyAdjusted = false,
            geometry = geometry,
            options = ProcessingOptions(),
        )
        val pdf = PdfExporter(File(root, "exports")).export(listOf(page))
        assertTrue(pdf.length() > 1_000)
        assertTrue(pdf.inputStream().use { input ->
            String(input.readNBytes(4), Charsets.US_ASCII) == "%PDF"
        })
        root.deleteRecursively()
    }

    @Test
    fun draftDetectionDoesNotCorrectBeforeUserConfirmation() {
        val root = File(context.cacheDir, "draft-test-${System.nanoTime()}").apply { mkdirs() }
        val original = File(root, "original.jpg")
        createSyntheticDocument(original)
        val status = ModelRuntimeStatus(
            backend = InferenceBackend.OPENCV_FALLBACK,
            modelVersion = "test-fallback",
            readyModelIds = emptySet(),
            missingModelIds = setOf("document_preview"),
            message = "test",
        )
        val processor = DocumentProcessor(root, detector, status)

        val draft = processor.createDraft(original)

        assertFalse(draft.hasProcessedPreview)
        assertFalse(File(draft.previewPath).exists())

        val processed = processor.processDraft(draft)

        assertTrue(processed.hasProcessedPreview)
        assertTrue(File(processed.previewPath).length() > 1_000)
        root.deleteRecursively()
    }

    @Test
    fun decodesCameraExifOrientationForEditorAndProcessor() {
        val target = File(context.cacheDir, "rotation-${System.nanoTime()}.jpg")
        val bitmap = Bitmap.createBitmap(120, 80, Bitmap.Config.ARGB_8888)
        FileOutputStream(target).use { bitmap.compress(Bitmap.CompressFormat.JPEG, 95, it) }
        bitmap.recycle()
        ExifInterface(target).apply {
            setAttribute(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_ROTATE_90.toString())
            saveAttributes()
        }

        val oriented = decodeOrientedBitmap(target)

        assertEquals(80, oriented.width)
        assertEquals(120, oriented.height)
        oriented.recycle()
        target.delete()
    }

    @Test
    fun nativeDecodeAppliesExifAndUserRotationExactlyOnce() {
        val target = File(context.cacheDir, "native-rotation-${System.nanoTime()}.jpg")
        val bitmap = Bitmap.createBitmap(120, 80, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(bitmap)
        canvas.drawColor(Color.BLACK)
        val paint = Paint().apply { color = Color.RED }
        canvas.drawRect(0f, 0f, 60f, 40f, paint)
        paint.color = Color.GREEN
        canvas.drawRect(60f, 0f, 120f, 40f, paint)
        paint.color = Color.BLUE
        canvas.drawRect(0f, 40f, 60f, 80f, paint)
        paint.color = Color.YELLOW
        canvas.drawRect(60f, 40f, 120f, 80f, paint)
        FileOutputStream(target).use { bitmap.compress(Bitmap.CompressFormat.JPEG, 98, it) }
        bitmap.recycle()
        ExifInterface(target).apply {
            setAttribute(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_ROTATE_90.toString())
            saveAttributes()
        }

        for (userRotation in listOf(0, 90, 180, 270)) {
            val expected = decodeOrientedBitmap(target, additionalRotationDegrees = userRotation)
            val actual = decodeOrientedMat(target, additionalRotationDegrees = userRotation)
            try {
                assertEquals(expected.width, actual.cols())
                assertEquals(expected.height, actual.rows())
                val pixel = actual.get(5, 5)
                val color = expected.getPixel(5, 5)
                assertEquals(android.graphics.Color.blue(color).toDouble(), pixel[0], 12.0)
                assertEquals(android.graphics.Color.green(color).toDouble(), pixel[1], 12.0)
                assertEquals(android.graphics.Color.red(color).toDouble(), pixel[2], 12.0)
            } finally {
                expected.recycle()
                actual.release()
            }
        }
        target.delete()
    }

    @Test
    fun missingDewarpRequiresUserToDisableItInsteadOfExportingApproximation() {
        val root = File(context.cacheDir, "unavailable-test-${System.nanoTime()}").apply { mkdirs() }
        val original = File(root, "original.jpg")
        createSyntheticDocument(original)
        val status = ModelRuntimeStatus(InferenceBackend.OPENCV_FALLBACK, "test",
            emptySet(), setOf("document_dewarp"), "test")
        val processor = DocumentProcessor(root, detector, status)
        val draft = processor.createDraft(original)
        org.junit.Assert.assertThrows(IllegalStateException::class.java) {
            processor.processDraft(draft.copy(options = ProcessingOptions(dewarpEnabled = true)))
        }
        assertFalse(File(draft.previewPath).exists())
        assertTrue(processor.processDraft(draft.copy(options = ProcessingOptions(dewarpEnabled = false))).hasProcessedPreview)
        root.deleteRecursively()
    }

    private fun createSyntheticDocument(target: File) {
        val bitmap = Bitmap.createBitmap(1000, 800, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(bitmap)
        canvas.drawColor(Color.rgb(35, 44, 42))
        val paper = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.rgb(246, 244, 237) }
        canvas.drawRect(90f, 70f, 910f, 730f, paper)
        val ink = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.rgb(35, 35, 35)
            strokeWidth = 7f
        }
        repeat(10) { row -> canvas.drawLine(170f, 165f + row * 45f, 820f, 165f + row * 45f, ink) }
        FileOutputStream(target).use { bitmap.compress(Bitmap.CompressFormat.JPEG, 96, it) }
        bitmap.recycle()
    }

    companion object {
        @JvmStatic
        @BeforeClass
        fun initializeOpenCv() {
            check(OpenCVLoader.initLocal())
        }
    }
}
