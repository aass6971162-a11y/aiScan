package com.huaan.sales.myapplication6

import android.content.Context
import android.graphics.BitmapFactory
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.huaan.sales.myapplication6.domain.NormalizedPoint
import com.huaan.sales.myapplication6.ml.ModelAssetRegistry
import com.huaan.sales.myapplication6.ml.OnnxCornerDetector
import com.huaan.sales.myapplication6.ml.OnnxImageModel
import com.huaan.sales.myapplication6.vision.OpenCvDocumentDetector
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.BeforeClass
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc
import java.io.File
import kotlin.math.hypot

@RunWith(AndroidJUnit4::class)
class ExtremeDocumentInstrumentedTest {
    @Test fun twentyPublicExtremePagesMatchDesktopAndOfficialCorners() {
        val context: Context = ApplicationProvider.getApplicationContext()
        val assets = InstrumentationRegistry.getInstrumentation().context.assets
        val cases = JSONArray(assets.open("extreme20/expected.json").bufferedReader().use { it.readText() })
        val registry = ModelAssetRegistry(context)
        val measured = JSONArray()
        var totalError = 0.0
        var withinOnePercent = 0
        var maxDesktopDifference = 0f
        OnnxCornerDetector(OnnxImageModel(registry.openOnnxSession("document_corners")), OpenCvDocumentDetector()).use { detector ->
            for (i in 0 until cases.length()) {
                val item = cases.getJSONObject(i)
                val id = item.getString("id")
                val bitmap = assets.open("extreme20/$id.png").use { BitmapFactory.decodeStream(it) }
                val rgba = Mat()
                val bgr = Mat()
                val result = try {
                    Utils.bitmapToMat(bitmap, rgba)
                    Imgproc.cvtColor(rgba, bgr, Imgproc.COLOR_RGBA2BGR)
                    detector.detectColor(bgr)
                } finally { bitmap.recycle(); rgba.release(); bgr.release() }
                assertEquals("$id must retain a four-corner candidate", 4, result.corners.size)
                val desktop = points(item.getJSONObject("new").getJSONArray("corners"))
                val difference = result.corners.indices.maxOf { corner ->
                    desktop[corner].distanceTo(result.corners[corner])
                }
                maxDesktopDifference = maxOf(maxDesktopDifference,difference)
                val expected = points(item.getJSONArray("expected_corners"))
                val width = result.sourceWidth.toDouble()
                val height = result.sourceHeight.toDouble()
                val error = expected.indices.sumOf { corner ->
                    hypot((expected[corner].x-result.corners[corner].x)*width,
                        (expected[corner].y-result.corners[corner].y)*height)
                } / 4 / hypot(width,height) * 100
                totalError += error
                if (error <= 1) withinOnePercent++
                measured.put(JSONObject().put("id",id).put("corner_error_pct",error)
                    .put("inference_ms",result.inferenceMillis).put("confidence",result.confidence)
                    .put("needs_confirmation",!result.isUsable)
                    .put("desktop_max_corner_difference",difference)
                    .put("corners",JSONArray(result.corners.map { listOf(it.x,it.y) })))
                android.util.Log.i("ExtremeRegression","$id error=$error ms=${result.inferenceMillis} desktopDifference=$difference corners=${result.corners}")
            }
        }
        File(context.filesDir,"extreme20-android.json").writeText(measured.toString(2))
        assertTrue("Android/desktop coordinates differ: $maxDesktopDifference",maxDesktopDifference < .002f)
        assertEquals(20,cases.length())
        assertTrue("Mean corner error must retain the v5 improvement",totalError/20 < .85)
        assertTrue("At least 17 fixed samples must stay below 1%",withinOnePercent >= 17)
    }

    private fun points(array: JSONArray) = (0 until array.length()).map { i ->
        val point = array.getJSONArray(i)
        NormalizedPoint(point.getDouble(0).toFloat(),point.getDouble(1).toFloat())
    }

    companion object {
        @JvmStatic @BeforeClass fun initialize() { check(OpenCVLoader.initLocal()) }
    }
}
