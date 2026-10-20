package com.huaan.sales.myapplication6.vision

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import androidx.exifinterface.media.ExifInterface
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.Mat
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc
import java.io.File
import java.io.FileOutputStream

/** Pixel dimensions after EXIF orientation, without allocating the image. */
fun orientedImageSize(file: File): Pair<Int, Int> {
    val bounds = BitmapFactory.Options().also { it.inJustDecodeBounds = true }
    BitmapFactory.decodeFile(file.absolutePath, bounds)
    require(bounds.outWidth > 0 && bounds.outHeight > 0) { "无法读取图像尺寸" }
    return if (ExifInterface(file).rotationDegrees % 180 == 0) {
        bounds.outWidth to bounds.outHeight
    } else {
        bounds.outHeight to bounds.outWidth
    }
}

/** Decodes directly to native BGR storage to avoid a full-resolution Android ARGB bitmap. */
fun decodeOrientedMat(
    file: File,
    maxEdge: Int = Int.MAX_VALUE,
    additionalRotationDegrees: Int = 0,
): Mat {
    require(maxEdge > 0)
    require(additionalRotationDegrees % 90 == 0)
    if (maxEdge != Int.MAX_VALUE) return decodeBitmapMat(file, maxEdge, additionalRotationDegrees)
    var image = Imgcodecs.imread(file.absolutePath,
        Imgcodecs.IMREAD_COLOR or Imgcodecs.IMREAD_IGNORE_ORIENTATION)
    if (image.empty()) {
        image.release()
        return decodeBitmapMat(file, maxEdge, additionalRotationDegrees)
    }
    try {
        val exif = ExifInterface(file)
        if (exif.isFlipped) {
            val flipped = Mat()
            Core.flip(image, flipped, 1)
            image.release()
            image = flipped
        }
        val degrees = ((exif.rotationDegrees + additionalRotationDegrees) % 360 + 360) % 360
        val rotation = when (degrees) {
            90 -> Core.ROTATE_90_CLOCKWISE
            180 -> Core.ROTATE_180
            270 -> Core.ROTATE_90_COUNTERCLOCKWISE
            else -> null
        }
        if (rotation != null) {
            val rotated = Mat()
            Core.rotate(image, rotated, rotation)
            image.release()
            image = rotated
        }
        return image
    } catch (error: Throwable) {
        image.release()
        throw error
    }
}

private fun decodeBitmapMat(file: File, maxEdge: Int, rotationDegrees: Int): Mat {
    val bitmap = decodeOrientedBitmap(file, maxEdge, rotationDegrees)
    val rgba = Mat()
    val bgr = Mat()
    try {
        Utils.bitmapToMat(bitmap, rgba)
        Imgproc.cvtColor(rgba, bgr, Imgproc.COLOR_RGBA2BGR)
        return bgr
    } catch (error: Throwable) {
        bgr.release()
        throw error
    } finally {
        rgba.release()
        bitmap.recycle()
    }
}

fun decodeOrientedBitmap(
    file: File,
    maxEdge: Int = Int.MAX_VALUE,
    additionalRotationDegrees: Int = 0,
    applyOrientation: Boolean = true,
): Bitmap {
    require(maxEdge > 0)
    require(additionalRotationDegrees % 90 == 0)
    val options = BitmapFactory.Options()
    if (maxEdge != Int.MAX_VALUE) {
        options.inJustDecodeBounds = true
        BitmapFactory.decodeFile(file.absolutePath, options)
        require(options.outWidth > 0 && options.outHeight > 0) { "无法读取拍摄图像" }
        var sample = 1
        // Sample before allocating, then fit exactly. Sampling to the next smaller
        // power of two alone can throw away almost half the required PDF detail.
        while (maxOf(options.outWidth, options.outHeight) / (sample * 2L) >= maxEdge) sample *= 2
        options.inSampleSize = sample
        options.inJustDecodeBounds = false
    }
    var bitmap = checkNotNull(BitmapFactory.decodeFile(file.absolutePath, options)) { "无法读取拍摄图像" }
    try {
        val longest = maxOf(bitmap.width, bitmap.height)
        if (longest > maxEdge) {
            val scale = maxEdge.toDouble() / longest
            val scaled = Bitmap.createScaledBitmap(bitmap,
                (bitmap.width * scale).toInt().coerceAtLeast(1),
                (bitmap.height * scale).toInt().coerceAtLeast(1), true)
            if (scaled !== bitmap) bitmap.recycle()
            bitmap = scaled
        }
        if (!applyOrientation) return bitmap
        val exif = ExifInterface(file)
        val degrees = ((exif.rotationDegrees + additionalRotationDegrees) % 360 + 360) % 360
        val flipped = exif.isFlipped
        if (degrees == 0 && !flipped) return bitmap
        val matrix = Matrix().apply {
            if (flipped) postScale(-1f, 1f)
            if (degrees != 0) postRotate(degrees.toFloat())
        }
        val oriented = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, false)
        if (oriented !== bitmap) bitmap.recycle()
        return oriented
    } catch (error: Throwable) {
        bitmap.recycle()
        throw error
    }
}

/** Small JPEG for lists. Pixels stay as stored: no rotation and no filter. */
fun writeCachedThumbnail(source: File, target: File, maxEdge: Int = 240) {
    val bitmap = decodeOrientedBitmap(source, maxEdge, applyOrientation = false)
    val temporary = File(target.parentFile, "${target.name}.tmp")
    try {
        target.parentFile?.mkdirs()
        FileOutputStream(temporary).use { output ->
            check(bitmap.compress(Bitmap.CompressFormat.JPEG, 80, output)) { "无法保存缩略图" }
            output.fd.sync()
        }
        if (target.exists()) target.delete()
        check(temporary.renameTo(target)) { "无法保存缩略图" }
    } finally {
        temporary.delete()
        bitmap.recycle()
    }
}

fun decodeCachedThumbnail(file: File): Bitmap? {
    if (!file.isFile) return null
    return BitmapFactory.decodeFile(file.absolutePath)
}
