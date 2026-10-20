package com.huaan.sales.myapplication6

import android.content.Context
import android.graphics.Bitmap
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.huaan.sales.myapplication6.data.ScanRepository
import com.huaan.sales.myapplication6.domain.ColorMode
import com.huaan.sales.myapplication6.domain.DocumentGeometry
import com.huaan.sales.myapplication6.domain.DraftPage
import com.huaan.sales.myapplication6.domain.NormalizedPoint
import com.huaan.sales.myapplication6.domain.PageEffect
import com.huaan.sales.myapplication6.domain.ProcessingOptions
import com.huaan.sales.myapplication6.ml.InferenceBackend
import com.huaan.sales.myapplication6.ml.ModelRuntimeStatus
import com.huaan.sales.myapplication6.vision.DocumentEffects
import com.huaan.sales.myapplication6.vision.DocumentProcessor
import com.huaan.sales.myapplication6.vision.OpenCvDocumentDetector
import com.huaan.sales.myapplication6.vision.decodeOrientedBitmap
import org.junit.Assert.*
import org.junit.BeforeClass
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Point
import org.opencv.core.Rect
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc
import java.io.File
import kotlin.math.exp
import kotlin.math.roundToInt

@RunWith(AndroidJUnit4::class)
class DocumentEffectsInstrumentedTest {
    private val context: Context = ApplicationProvider.getApplicationContext()

    @Test fun brightenIsMonotoneAndProtectsHighlights() {
        val ramp = Mat(8, 256, CvType.CV_8UC3)
        for (x in 0..255) for (y in 0..7) ramp.put(y, x, byteArrayOf(x.toByte(), x.toByte(), x.toByte()))
        val output = DocumentEffects.apply(ramp, PageEffect.BRIGHTEN)
        try {
            var previous = -1.0
            for (x in 0..255) {
                val level = output.get(4, x).average()
                assertTrue("ramp reverses at $x", level >= previous - 1.0)
                previous = level
            }
            assertTrue(output.get(4, 100).average() > 110)
            assertTrue(output.get(4, 240).average() in 240.0..252.0)
            assertEquals(0.0, output.get(4, 0).average(), 1.0)
            assertEquals(255.0, output.get(4, 255).average(), 1.0)
        } finally { ramp.release(); output.release() }
    }

    @Test fun sharpenSuppressesFlatNoiseAndStrengthensRealEdgesWithoutLargeHalos() {
        val source = Mat(128, 256, CvType.CV_8UC3)
        for (y in 0 until 128) for (x in 0 until 256) {
            val v = if (x < 128) 128 + (x * 17 + y * 13) % 5 - 2 else if (x < 192) 90 else 200
            source.put(y, x, byteArrayOf(v.toByte(), v.toByte(), v.toByte()))
        }
        val output = DocumentEffects.apply(source, PageEffect.SHARPEN)
        try {
            val noise = output.submat(Rect(8, 8, 100, 100))
            val original = source.submat(Rect(8, 8, 100, 100))
            try { assertTrue(Core.norm(noise, original, Core.NORM_INF) <= 2.0) }
            finally { noise.release(); original.release() }
            assertTrue(output.get(64, 191).average() < source.get(64, 191).average())
            assertTrue(output.get(64, 192).average() > source.get(64, 192).average())
            assertTrue(Core.norm(source, output, Core.NORM_INF) <= 14.0)
        } finally { source.release(); output.release() }
    }

    @Test fun shadowCorrectionKeepsCleanPaperAndChannelRatios() {
        for (value in listOf(238.0, 155.0, 45.0)) {
            val flat = Mat(96, 128, CvType.CV_8UC3, Scalar(value, value, value))
            val output = DocumentEffects.apply(flat, PageEffect.REMOVE_SHADOW)
            try { assertEquals(0.0, Core.norm(flat, output, Core.NORM_INF), 0.0) }
            finally { flat.release(); output.release() }
        }
        val color = Mat(96, 128, CvType.CV_8UC3, Scalar(80.0, 120.0, 200.0))
        val output = DocumentEffects.apply(color, PageEffect.REMOVE_SHADOW)
        try { assertEquals(0.0, Core.norm(color, output, Core.NORM_INF), 0.0) }
        finally { color.release(); output.release() }
    }

    @Test fun shadowIsReducedAndFaintStrokesSurviveBinarization() {
        val clean = fixture()
        val photo = shadowed(clean)
        val corrected = DocumentEffects.apply(photo, PageEffect.REMOVE_SHADOW)
        val binary = DocumentEffects.apply(photo, PageEffect.MONOCHROME)
        try {
            val cleanBytes = bytes(clean)
            val photoBytes = bytes(photo)
            val correctedBytes = bytes(corrected)
            val binaryBytes = bytes(binary)
            val before = mutableListOf<Double>()
            val after = mutableListOf<Double>()
            var faint = 0; var retained = 0; var paper = 0; var dirtyPaper = 0
            for (i in binaryBytes.indices) {
                val c = cleanBytes[i * 3].toInt() and 255
                if (c == 238) {
                    before += (photoBytes[i * 3].toInt() and 255).toDouble()
                    after += (correctedBytes[i * 3].toInt() and 255).toDouble()
                    paper++
                    if (binaryBytes[i].toInt() == 0) dirtyPaper++
                }
                if (c in 185..205) { faint++; if (binaryBytes[i].toInt() == 0) retained++ }
                assertTrue(binaryBytes[i].toInt() == 0 || binaryBytes[i].toInt() == -1)
            }
            val beforeSd = sd(before); val afterSd = sd(after)
            val recall = retained.toDouble() / faint
            assertTrue("shadow $beforeSd -> $afterSd", afterSd < beforeSd * .4)
            assertTrue("paper ${after.average()}", after.average() > 220)
            assertTrue("faint stroke recall=$recall", recall > .95)
            assertTrue("background speckle", dirtyPaper.toDouble() / paper < .01)
            File(context.filesDir, "effects-metrics.txt").writeText(
                "paper_before_mean=${before.average()}\npaper_after_mean=${after.average()}\n" +
                    "paper_before_std=$beforeSd\npaper_after_std=$afterSd\nfaint_recall=$recall\n" +
                    "paper_black_fraction=${dirtyPaper.toDouble()/paper}\n")
            saveSheet(photo, "effects-synthetic.png")
        } finally { clean.release(); photo.release(); corrected.release(); binary.release() }
    }

    @Test fun grayscaleIsPlainLuminanceAndLegacyModesUseSameImplementation() {
        val source = fixture()
        val gray = DocumentEffects.apply(source, PageEffect.GRAYSCALE)
        val legacy = DocumentEffects.apply(source, ProcessingOptions(removeShadows = false, colorMode = ColorMode.GRAYSCALE))
        val expected = Mat()
        val bw = DocumentEffects.apply(source, PageEffect.MONOCHROME)
        val oldBw = DocumentEffects.apply(source, ProcessingOptions(removeShadows = false, colorMode = ColorMode.BLACK_AND_WHITE))
        try {
            Imgproc.cvtColor(source, expected, Imgproc.COLOR_BGR2GRAY)
            assertEquals(1, gray.channels())
            assertEquals(0.0, Core.norm(expected, gray, Core.NORM_INF), 0.0)
            assertEquals(0.0, Core.norm(gray, legacy, Core.NORM_INF), 0.0)
            assertEquals(0.0, Core.norm(bw, oldBw, Core.NORM_INF), 0.0)
        } finally { listOf(source, gray, legacy, expected, bw, oldBw).forEach { it.release() } }
    }

    @Test fun invertIsExactReversibleAndAllEffectsLeaveSourceUntouched() {
        val source = fixture()
        val original = source.clone()
        try {
            for (effect in PageEffect.entries) {
                val output = DocumentEffects.apply(source, effect)
                try { assertEquals(0.0, Core.norm(source, original, Core.NORM_INF), 0.0) }
                finally { output.release() }
            }
            val inverted = DocumentEffects.apply(source, PageEffect.INVERT)
            val twice = DocumentEffects.apply(inverted, PageEffect.INVERT)
            try {
                assertEquals(0.0, Core.norm(source, twice, Core.NORM_INF), 0.0)
                assertArrayEquals(doubleArrayOf(17.0, 17.0, 17.0), inverted.get(0, 0), 0.0)
            } finally { inverted.release(); twice.release() }
        } finally { source.release(); original.release() }
    }

    @Test fun allEffectsSupportTinyImagesAndSolidDarkMarksRemainFilled() {
        for (size in listOf(1, 2, 8, 16, 128)) {
            val source = Mat(size, size, CvType.CV_8UC3, Scalar(180.0, 190.0, 200.0))
            try {
                for (effect in PageEffect.entries) {
                    val output = DocumentEffects.apply(source, effect)
                    try { assertEquals(source.size(), output.size()) } finally { output.release() }
                }
            } finally { source.release() }
        }
        val marks = Mat(300, 300, CvType.CV_8UC3, Scalar(240.0, 240.0, 240.0))
        Imgproc.rectangle(marks, Point(80.0, 80.0), Point(220.0, 220.0), Scalar(15.0, 15.0, 15.0), -1)
        val binary = DocumentEffects.apply(marks, PageEffect.MONOCHROME)
        try { assertEquals(0.0, binary.get(150, 150)[0], 0.0) }
        finally { marks.release(); binary.release() }
    }

    @Test fun highResolutionPreviewIsBoundedButSavedOutputRetainsOriginalResolution() {
        val root=File(context.cacheDir,"effects-resolution-${System.nanoTime()}").apply { mkdirs() }
        val original=File(root,"source.png")
        val input=Mat(4000,3000,CvType.CV_8UC3,Scalar(230.0,230.0,230.0))
        Imgproc.putText(input,"12 MP ORIGINAL",Point(200.0,2100.0),Imgproc.FONT_HERSHEY_SIMPLEX,4.0,Scalar(45.0,45.0,45.0),3)
        Imgcodecs.imwrite(original.absolutePath,input)
        input.release()
        val processor=DocumentProcessor(root,OpenCvDocumentDetector(),
            ModelRuntimeStatus(InferenceBackend.OPENCV_FALLBACK,"test",emptySet(),emptySet(),"test"))
        val geometry=DocumentGeometry.fromCorners(listOf(NormalizedPoint(0f,0f),NormalizedPoint(1f,0f),
            NormalizedPoint(1f,1f),NormalizedPoint(0f,1f)))
        val draft=DraftPage("large",original.absolutePath,File(root,"result.jpg").absolutePath,geometry,
            ProcessingOptions(removeShadows=false).withEffect(PageEffect.SHARPEN),1f,0L,"test",false)
        try {
            val preview=processor.applyFullFrameEffect(draft)
            val bitmap=decodeOrientedBitmap(File(preview.effectPreviewPath!!))
            try { assertTrue(maxOf(bitmap.width,bitmap.height)<=1600) } finally { bitmap.recycle() }
            val result=processor.process(draft.id,original,File(draft.previewPath),geometry,draft.options,warp=false)
            assertEquals(3000,result.width)
            assertEquals(4000,result.height)
            assertTrue(result.modelVersion.contains(DocumentEffects.VERSION))
        } finally { root.deleteRecursively() }
    }

    @Test fun previewSwitchingSavingAndReeditingAlwaysUseOriginalPixels() {
        val root = File(context.cacheDir, "effects-pipeline-${System.nanoTime()}").apply { mkdirs() }
        val source = fixture()
        val original = File(root, "source.png")
        Imgcodecs.imwrite(original.absolutePath, source)
        source.release()
        val originalHash = original.readBytes().contentHashCode()
        val processor = DocumentProcessor(root, OpenCvDocumentDetector(),
            ModelRuntimeStatus(InferenceBackend.OPENCV_FALLBACK, "test", emptySet(), emptySet(), "test"))
        val repository = ScanRepository(context)
        val geometry = DocumentGeometry.fromCorners(listOf(NormalizedPoint(.1f,.1f), NormalizedPoint(.9f,.1f),
            NormalizedPoint(.9f,.9f), NormalizedPoint(.1f,.9f)))
        var draft = DraftPage("effects-test-${System.nanoTime()}", original.absolutePath,
            File(root,"result.jpg").absolutePath, geometry, ProcessingOptions(removeShadows=false), 1f, 0L, "test", false)
        try {
            draft = processor.applyFullFrameEffect(draft.copy(options=draft.options.withEffect(PageEffect.BRIGHTEN)))
            val first = File(draft.effectPreviewPath!!).readBytes()
            val preview = decodeOrientedBitmap(File(draft.effectPreviewPath!!))
            try { assertEquals(720, preview.width); assertEquals(1000, preview.height) } finally { preview.recycle() }
            assertFalse(File(draft.previewPath).exists())
            draft = processor.applyFullFrameEffect(draft.copy(options=draft.options.withEffect(PageEffect.INVERT)))
            draft = processor.applyFullFrameEffect(draft.copy(options=draft.options.withEffect(PageEffect.BRIGHTEN)))
            assertArrayEquals(first, File(draft.effectPreviewPath!!).readBytes())
            draft = processor.processDraft(draft)
            assertTrue(draft.hasProcessedPreview)
            val page = repository.saveDraft(draft)
            assertEquals(draft.options, page.options)
            assertTrue(page.width < 720 && page.height < 1000)
            val reopened = repository.toDraft(page, root)
            val regenerated = processor.applyFullFrameEffect(reopened)
            assertArrayEquals(first, File(regenerated.effectPreviewPath!!).readBytes())
            assertEquals(originalHash, original.readBytes().contentHashCode())
            val savedAgain = processor.processDraft(regenerated)
            assertArrayEquals(File(page.correctedPath).readBytes(), File(savedAgain.previewPath).readBytes())
        } finally { repository.delete(draft.id); root.deleteRecursively() }
    }

    @Test fun realPhotosRenderAllEffectsAndRepeatedProcessingStaysValid() {
        val assets = InstrumentationRegistry.getInstrumentation().context.assets
        for ((index, asset) in listOf("model-golden/photo_original.jpg", "model-golden/complex-stacked-pages.jpg",
            "extreme20/00000.png", "extreme20/00028.png").withIndex()) {
            val temp = File(context.cacheDir, "effects-real-$index.jpg")
            assets.open(asset).use { input -> temp.outputStream().use { input.copyTo(it) } }
            val bitmap = decodeOrientedBitmap(temp, maxEdge=1200)
            val rgba = Mat(); val bgr = Mat()
            try {
                Utils.bitmapToMat(bitmap, rgba)
                Imgproc.cvtColor(rgba,bgr,Imgproc.COLOR_RGBA2BGR)
                saveSheet(bgr, "effects-real-$index.png")
                repeat(8) {
                    val out=DocumentEffects.apply(bgr,PageEffect.REMOVE_SHADOW)
                    try { assertEquals(bgr.size(),out.size()) } finally { out.release() }
                }
            } finally { bitmap.recycle(); rgba.release(); bgr.release(); temp.delete() }
        }
    }

    private fun fixture(): Mat {
        val clean=Mat(1000,720,CvType.CV_8UC3,Scalar(238.0,238.0,238.0))
        repeat(13) { i ->
            val v=if(i%3==0)192.0 else 45.0
            Imgproc.putText(clean,"Page %02d  fine text 12345".format(i+1), Point(32.0,110.0+i*53),
                Imgproc.FONT_HERSHEY_SIMPLEX,.64,Scalar(v,v,v),1,Imgproc.LINE_AA)
        }
        return clean
    }
    private fun shadowed(clean: Mat): Mat {
        val data=bytes(clean)
        for(y in 0 until clean.rows()) for(x in 0 until clean.cols()) {
            val gx=.45+.55/(1+exp(-(x/(clean.cols()-1.0)-.45)*16))
            val gy=1-.22*exp(-Math.pow((y/(clean.rows()-1.0)-.62)/.08,2.0))
            for(c in 0..2) { val i=(y*clean.cols()+x)*3+c; data[i]=((data[i].toInt() and 255)*gx*gy).toInt().toByte() }
        }
        return Mat(clean.rows(),clean.cols(),clean.type()).also { it.put(0,0,data) }
    }
    private fun bytes(mat: Mat)=ByteArray((mat.total()*mat.channels()).toInt()).also { mat.get(0,0,it) }
    private fun sd(values: List<Double>): Double { val mean=values.average();return kotlin.math.sqrt(values.sumOf { (it-mean)*(it-mean) }/values.size) }
    private fun saveSheet(source: Mat, name: String) {
        val tiles=mutableListOf<Mat>()
        val labels=listOf("Original","Brighten","Sharpen","Shadow v2","B&W","Gray","Invert")
        try {
            for((index,effect) in PageEffect.entries.withIndex()) {
                val result=DocumentEffects.apply(source,effect)
                val color=Mat();val tile=Mat()
                try {
                    if(result.channels()==1)Imgproc.cvtColor(result,color,Imgproc.COLOR_GRAY2BGR) else result.copyTo(color)
                    Imgproc.resize(color,tile,Size(252.0,350.0))
                    Core.copyMakeBorder(tile,tile,32,0,0,0,Core.BORDER_CONSTANT,Scalar(32.0,32.0,32.0))
                    Imgproc.putText(tile,labels[index],Point(8.0,23.0),Imgproc.FONT_HERSHEY_SIMPLEX,.6,Scalar(255.0,255.0,255.0),1,Imgproc.LINE_AA)
                    tiles+=tile
                } finally { result.release();color.release() }
            }
            val sheet=Mat()
            try { Core.hconcat(tiles,sheet);assertTrue(Imgcodecs.imwrite(File(context.filesDir,name).absolutePath,sheet)) }
            finally { sheet.release() }
        } finally { tiles.forEach { it.release() } }
    }
    companion object { @JvmStatic @BeforeClass fun loadOpenCv() { assertTrue(OpenCVLoader.initLocal()) } }
}
