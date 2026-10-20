package com.huaan.sales.myapplication6

import android.content.Context
import android.graphics.BitmapFactory
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.huaan.sales.myapplication6.domain.DocumentDetection
import com.huaan.sales.myapplication6.domain.NormalizedPoint
import com.huaan.sales.myapplication6.ml.*
import com.huaan.sales.myapplication6.vision.OpenCvDocumentDetector
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.BeforeClass
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import java.io.File
import kotlin.math.hypot

/** One local real photograph plus nine controlled transforms, not ten independent real scenes. */
@RunWith(AndroidJUnit4::class)
class MultilayerDocumentInstrumentedTest {
    private val context: Context = ApplicationProvider.getApplicationContext()
    private val assets = InstrumentationRegistry.getInstrumentation().context.assets
    private val registry = ModelAssetRegistry(context)

    @Test fun layeredForegroundSurvivesExposureDistanceRotationAndReframing() {
        val cases = JSONArray(assets.open("multilayer/expected.json").bufferedReader().use { it.readText() })
        val measured = JSONArray()
        var totalError = 0.0
        var oldError = 0.0
        var withinOnePercent = 0
        detector().use { detector ->
            for (i in 0 until cases.length()) {
                val item = cases.getJSONObject(i)
                val id = item.getString("id")
                val image = fixture(id)
                try {
                    val result = detector.detectColor(image)
                    assertEquals("$id retains a whole-page hypothesis", 4, result.corners.size)
                    val expected = points(item.getJSONArray("expected_corners"))
                    val desktop = points(item.getJSONObject("new").getJSONArray("corners"))
                    val difference = expected.indices.maxOf { desktop[it].distanceTo(result.corners[it]) }
                    assertTrue("$id Android/desktop disagree: $difference", difference < .002f)
                    val error = error(result, expected)
                    assertTrue("$id foreground error=$error", error < 2.0)
                    totalError += error
                    oldError += item.getJSONObject("old").getDouble("corner_error_pct")
                    if (error <= 1.0) withinOnePercent++
                    measured.put(JSONObject().put("id", id).put("corner_error_pct", error)
                        .put("old_corner_error_pct", item.getJSONObject("old").getDouble("corner_error_pct"))
                        .put("desktop_max_corner_difference", difference).put("inference_ms", result.inferenceMillis)
                        .put("needs_confirmation", !result.isUsable)
                        .put("corners", JSONArray(result.corners.map { listOf(it.x, it.y) })))
                } finally { image.release() }
            }
        }
        File(context.filesDir, "multilayer-android.json").writeText(measured.toString(2))
        assertEquals(10, cases.length())
        assertTrue("Layered regression must substantially improve on v4", totalError < oldError * .35)
        assertTrue("At least nine controlled cases stay within 1%", withinOnePercent >= 9)
    }

    @Test fun liveTrackingRecomputesCurrentPixelsAndRecoversAfterSceneChanges() {
        val cases = JSONArray(assets.open("multilayer/expected.json").bufferedReader().use { it.readText() })
        val measured = JSONArray()
        detector().use { detector ->
            for (i in listOf(0, 0, 0, 7, 7, 5, 5, 0, 0)) {
                val item = cases.getJSONObject(i)
                val image = fixture(item.getString("id"))
                try {
                    val result = detector.detectPreviewColor(image)
                    assertEquals(4, result.corners.size)
                    val error = error(result, points(item.getJSONArray("expected_corners")))
                    assertTrue("Scale tracking reused stale corners after a scene change: $error", error < 1.0)
                    measured.put(JSONObject().put("id", item.getString("id")).put("error_pct", error)
                        .put("inference_ms", result.inferenceMillis))
                } finally { image.release() }
            }
            val blank = Mat(640, 480, CvType.CV_8UC3, Scalar(127.0, 127.0, 127.0))
            try { assertFalse("Scale cache must not retain the old page", detector.detectPreviewColor(blank).isUsable) }
            finally { blank.release() }
            val image = fixture("stack-original")
            try { assertTrue(error(detector.detectPreviewColor(image), points(cases.getJSONObject(0)
                .getJSONArray("expected_corners"))) < 1.0) } finally { image.release() }
        }
        File(context.filesDir, "multilayer-preview-android.json").writeText(measured.toString(2))
    }

    private fun detector() = OnnxCornerDetector(OnnxImageModel(registry.openOnnxSession("document_corners")),
        OpenCvDocumentDetector())

    private fun fixture(id: String): Mat {
        val bitmap = assets.open("multilayer/$id.png").use { BitmapFactory.decodeStream(it) }
        val rgba = Mat(); val bgr = Mat()
        Utils.bitmapToMat(bitmap, rgba)
        Imgproc.cvtColor(rgba, bgr, Imgproc.COLOR_RGBA2BGR)
        bitmap.recycle(); rgba.release()
        return bgr
    }

    private fun points(array: JSONArray) = (0 until array.length()).map {
        val p = array.getJSONArray(it)
        NormalizedPoint(p.getDouble(0).toFloat(), p.getDouble(1).toFloat())
    }

    private fun error(result: DocumentDetection, expected: List<NormalizedPoint>) = expected.indices.sumOf {
        hypot((result.corners[it].x - expected[it].x) * result.sourceWidth.toDouble(),
            (result.corners[it].y - expected[it].y) * result.sourceHeight.toDouble())
    } / 4 / hypot(result.sourceWidth.toDouble(), result.sourceHeight.toDouble()) * 100

    companion object {
        @JvmStatic @BeforeClass fun initialize() { check(OpenCVLoader.initLocal()) }
    }
}
