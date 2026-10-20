package com.huaan.sales.myapplication6.vision

import android.graphics.Bitmap
import com.huaan.sales.myapplication6.domain.ColorMode
import com.huaan.sales.myapplication6.domain.PageEffect
import com.huaan.sales.myapplication6.domain.ProcessingOptions
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/** All entry points borrow the source and return a separately owned result. */
object DocumentEffects {
    const val VERSION = "document-effects-2"

    fun apply(source: Mat, options: ProcessingOptions): Mat {
        require(!source.empty() && source.type() == CvType.CV_8UC3) { "效果需要 BGR 8 位彩色图像" }
        if (options.effect != PageEffect.ORIGINAL) return apply(source, options.effect)
        val color = if (options.removeShadows) removeShadows(source) else source.clone()
        if (options.colorMode == ColorMode.COLOR) return color
        return try {
            when (options.colorMode) {
                ColorMode.COLOR -> error("Handled above")
                ColorMode.GRAYSCALE -> grayscale(color)
                ColorMode.BLACK_AND_WHITE -> monochrome(color)
            }
        } finally { color.release() }
    }

    fun apply(source: Mat, effect: PageEffect): Mat {
        require(!source.empty() && source.type() == CvType.CV_8UC3) { "效果需要 BGR 8 位彩色图像" }
        return when (effect) {
            PageEffect.ORIGINAL -> source.clone()
            PageEffect.BRIGHTEN -> mapLightness(source) { light, pool ->
                val lut = pool.mat(1, 256, CvType.CV_8U)
                // Raise midtones, with zero shift at black/white and a monotone highlight shoulder.
                lut.put(0, 0, ByteArray(256) { x ->
                    (x + 0.38 * x * (1 - x / 255.0)).roundToInt().coerceIn(0, 255).toByte()
                })
                Core.LUT(light, lut, light)
            }
            PageEffect.SHARPEN -> mapLightness(source) { light, pool ->
                val blurred = pool.mat()
                Imgproc.GaussianBlur(light, blurred, Size(0.0, 0.0), 0.85)
                // A row buffer avoids several full-resolution float maps on 12–48 MP photos.
                val row = ByteArray(light.cols())
                val smooth = ByteArray(light.cols())
                for (y in 0 until light.rows()) {
                    light.get(y, 0, row)
                    blurred.get(y, 0, smooth)
                    for (x in row.indices) {
                        val value = row[x].toInt() and 255
                        val detail = value - (smooth[x].toInt() and 255)
                        val boost = min(8.0, max(0, abs(detail) - 5) * 0.55).roundToInt()
                        row[x] = (value + if (detail < 0) -boost else boost).coerceIn(0, 255).toByte()
                    }
                    light.put(y, 0, row)
                }
            }
            PageEffect.REMOVE_SHADOW -> removeShadows(source)
            PageEffect.GRAYSCALE -> grayscale(source)
            PageEffect.MONOCHROME -> monochrome(source)
            PageEffect.INVERT -> Mat().also { output ->
                try { Core.bitwise_not(source, output) } catch (error: Throwable) {
                    output.release(); throw error
                }
            }
        }
    }

    /** Thumbnails run the same algorithms off the UI thread, on a bounded oriented bitmap. */
    fun thumbnail(source: Bitmap, effect: PageEffect): Bitmap = Mats().use { pool ->
        val rgba = pool.mat()
        val bgr = pool.mat()
        Utils.bitmapToMat(source, rgba)
        Imgproc.cvtColor(rgba, bgr, Imgproc.COLOR_RGBA2BGR)
        val result = pool.own(apply(bgr, effect))
        Imgproc.cvtColor(result, rgba, if (result.channels() == 1) Imgproc.COLOR_GRAY2RGBA else Imgproc.COLOR_BGR2RGBA)
        val bitmap = Bitmap.createBitmap(rgba.cols(), rgba.rows(), Bitmap.Config.ARGB_8888)
        try { Utils.matToBitmap(rgba, bitmap); bitmap } catch (error: Throwable) {
            bitmap.recycle(); throw error
        }
    }

    private fun mapLightness(source: Mat, transform: (Mat, Mats) -> Unit): Mat = Mats().use { pool ->
        val lab = pool.mat()
        val light = pool.mat()
        Imgproc.cvtColor(source, lab, Imgproc.COLOR_BGR2Lab)
        Core.extractChannel(lab, light, 0)
        transform(light, pool)
        Core.insertChannel(light, lab, 0)
        val output = Mat()
        try { Imgproc.cvtColor(lab, output, Imgproc.COLOR_Lab2BGR); output } catch (error: Throwable) {
            output.release(); throw error
        }
    }

    private fun grayscale(source: Mat): Mat = Mat().also { output ->
        try { Imgproc.cvtColor(source, output, Imgproc.COLOR_BGR2GRAY) } catch (error: Throwable) {
            output.release(); throw error
        }
    }

    private fun removeShadows(source: Mat): Mat = Mats().use { pool ->
        // Estimate achromatic shading from the brightest channel. A common gain on B/G/R
        // preserves channel ratios (unlike changing Lab L around saturated coloured ink).
        val maximum = pool.mat()
        val channel = pool.mat()
        Core.extractChannel(source, maximum, 0)
        for (index in 1..2) {
            Core.extractChannel(source, channel, index)
            Core.max(maximum, channel, maximum)
        }
        val background = pool.own(illumination(maximum))
        val target = percentile(background, 0.90)
        val gain = pool.mat()
        background.convertTo(gain, CvType.CV_32F)
        Core.add(gain, Scalar(6.0), gain)
        Core.divide(target + 6.0, gain, gain)
        // Never darken already lit paper or amplify deep-shadow noise without bound.
        Core.max(gain, Scalar(1.0), gain)
        Core.min(gain, Scalar(2.2), gain)
        maximum.convertTo(channel, CvType.CV_32F)
        Core.max(channel, Scalar(1.0), channel)
        Core.divide(255.0, channel, channel)
        Core.min(gain, channel, gain)
        val output = Mat(source.rows(), source.cols(), source.type())
        try {
            for (index in 0..2) {
                Core.extractChannel(source, channel, index)
                channel.convertTo(channel, CvType.CV_32F)
                Core.multiply(channel, gain, channel)
                channel.convertTo(channel, CvType.CV_8U)
                Core.insertChannel(channel, output, index)
            }
            output
        } catch (error: Throwable) { output.release(); throw error }
    }

    private fun monochrome(source: Mat): Mat = Mats().use { pool ->
        val gray = pool.own(grayscale(source))
        val background = pool.own(illumination(gray))
        val normalized = pool.mat()
        val gain = pool.mat()
        background.convertTo(gain, CvType.CV_32F)
        Core.add(gain, Scalar(1.0), gain)
        Core.divide(245.0, gain, gain)
        Core.max(gain, Scalar(1.0), gain)
        Core.min(gain, Scalar(2.8), gain)
        gray.convertTo(normalized, CvType.CV_32F)
        Core.multiply(normalized, gain, normalized)
        normalized.convertTo(normalized, CvType.CV_8U)
        val block = (min(source.cols(), source.rows()) / 14.0).roundToInt().coerceIn(15, 101) or 1
        val output = Mat()
        try {
            // No pre-blur: thin and faint strokes must not be averaged into the paper.
            Imgproc.adaptiveThreshold(normalized, output, 255.0, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C,
                Imgproc.THRESH_BINARY, block, 5.0)
            val dark = pool.mat()
            Core.compare(normalized, Scalar(110.0), dark, Core.CMP_LT)
            // Keep the interior of solid dark marks, which local thresholding alone hollows out.
            output.setTo(Scalar(0.0), dark)
            output
        } catch (error: Throwable) { output.release(); throw error }
    }

    private fun illumination(light: Mat): Mat = Mats().use { pool ->
        val scale = min(1.0, 768.0 / max(light.cols(), light.rows()))
        val small = pool.mat()
        Imgproc.resize(light, small, Size(max(1, (light.cols() * scale).roundToInt()).toDouble(),
            max(1, (light.rows() * scale).roundToInt()).toDouble()), 0.0, 0.0, Imgproc.INTER_AREA)
        val size = (min(small.cols(), small.rows()) / 20.0).roundToInt().coerceIn(3, 61) or 1
        val kernel = pool.own(Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(size.toDouble(), size.toDouble())))
        // Closing removes dark character strokes before estimating low-frequency light.
        Imgproc.morphologyEx(small, small, Imgproc.MORPH_CLOSE, kernel,
            org.opencv.core.Point(-1.0, -1.0), 1, Core.BORDER_REPLICATE)
        Imgproc.GaussianBlur(small, small, Size(0.0, 0.0), max(1.0, size / 6.0))
        val output = Mat()
        try { Imgproc.resize(small, output, light.size(), 0.0, 0.0, Imgproc.INTER_LINEAR); output }
        catch (error: Throwable) { output.release(); throw error }
    }

    private fun percentile(gray: Mat, fraction: Double): Double = Mats().use { pool ->
        val small = pool.mat()
        Imgproc.resize(gray, small, Size(128.0, 128.0), 0.0, 0.0, Imgproc.INTER_AREA)
        val values = ByteArray(small.total().toInt())
        small.get(0, 0, values)
        val bins = IntArray(256)
        values.forEach { bins[it.toInt() and 255]++ }
        val target = (values.size * fraction).roundToInt()
        var count = 0
        for (index in bins.indices) {
            count += bins[index]
            if (count >= target) return@use index.toDouble()
        }
        255.0
    }

    private class Mats : AutoCloseable {
        private val owned = mutableListOf<Mat>()
        fun mat(): Mat = own(Mat())
        fun mat(rows: Int, cols: Int, type: Int): Mat = own(Mat(rows, cols, type))
        fun own(mat: Mat): Mat = mat.also { owned += it }
        override fun close() { owned.asReversed().forEach { it.release() } }
    }
}
