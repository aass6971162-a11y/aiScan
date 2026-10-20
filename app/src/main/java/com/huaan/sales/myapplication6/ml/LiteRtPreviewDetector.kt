package com.huaan.sales.myapplication6.ml

import com.huaan.sales.myapplication6.domain.DocumentDetection
import com.huaan.sales.myapplication6.domain.NormalizedPoint
import com.huaan.sales.myapplication6.domain.QualityIssue
import com.huaan.sales.myapplication6.vision.PreviewDetector
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.abs

class LiteRtPreviewDetector(
    private val interpreter: Interpreter,
    private val fallback: PreviewDetector,
) : PreviewDetector, AutoCloseable {
    override fun detect(gray: Mat, rotationDegrees: Int): DocumentDetection {
        val fallbackDetection = fallback.detect(gray, rotationDegrees)
        return runCatching { infer(gray, rotationDegrees, fallbackDetection) }
            .getOrDefault(fallbackDetection)
    }

    private fun infer(
        gray: Mat,
        rotationDegrees: Int,
        fallbackDetection: DocumentDetection,
    ): DocumentDetection {
        val started = System.nanoTime()
        val inputShape = interpreter.getInputTensor(0).shape()
        val inputHeight = inputShape[1]
        val inputWidth = inputShape[2]
        val resized = Mat()
        Imgproc.resize(gray, resized, Size(inputWidth.toDouble(), inputHeight.toDouble()))
        val pixels = ByteArray(inputWidth * inputHeight)
        resized.get(0, 0, pixels)
        resized.release()
        val input = ByteBuffer.allocateDirect(inputWidth * inputHeight * 3 * Float.SIZE_BYTES)
            .order(ByteOrder.nativeOrder())
        pixels.forEach { value ->
            val normalized = (value.toInt() and 0xff) / 255f
            repeat(3) { input.putFloat(normalized) }
        }
        input.rewind()

        val outputs = mutableMapOf<Int, Any>()
        repeat(interpreter.outputTensorCount) { index ->
            val tensor = interpreter.getOutputTensor(index)
            outputs[index] = ByteBuffer.allocateDirect(tensor.numBytes()).order(ByteOrder.nativeOrder())
        }
        interpreter.runForMultipleInputsOutputs(arrayOf(input), outputs)
        val maskIndex = (0 until interpreter.outputTensorCount).firstOrNull {
            interpreter.getOutputTensor(it).name().contains("mask", ignoreCase = true)
        } ?: 0
        val tensor = interpreter.getOutputTensor(maskIndex)
        val shape = tensor.shape()
        val channelsLast = shape.size == 4 && shape[3] <= 4
        val maskHeight = when {
            shape.size != 4 -> inputHeight
            channelsLast -> shape[1]
            else -> shape[2]
        }
        val maskWidth = when {
            shape.size != 4 -> inputWidth
            channelsLast -> shape[2]
            else -> shape[3]
        }
        val outputBuffer = outputs.getValue(maskIndex) as ByteBuffer
        outputBuffer.rewind()
        val maskBytes = ByteArray(maskWidth * maskHeight)
        var probabilityTotal = 0f
        for (index in maskBytes.indices) {
            val probability = outputBuffer.float
            probabilityTotal += probability
            maskBytes[index] = if (probability >= 0.5f) 255.toByte() else 0
        }
        val mask = Mat(maskHeight, maskWidth, CvType.CV_8UC1)
        mask.put(0, 0, maskBytes)
        val heatmapIndex = (0 until interpreter.outputTensorCount).firstOrNull {
            val name = interpreter.getOutputTensor(it).name()
            name.contains("corner", ignoreCase = true) || name.contains("heat", ignoreCase = true)
        }
        val heatmapCorners = heatmapIndex?.let { index ->
            extractHeatmapCorners(interpreter.getOutputTensor(index).shape(), outputs.getValue(index) as ByteBuffer)
        }.orEmpty()
        val corners = if (heatmapCorners.size == 4) heatmapCorners else extractCorners(mask)
        mask.release()
        if (corners.size != 4) return fallbackDetection

        val confidence = ((probabilityTotal / maskBytes.size) * 2f).coerceIn(0f, 1f)
        val issues = fallbackDetection.qualityIssues.toMutableSet().apply {
            remove(QualityIssue.NO_DOCUMENT)
            if (confidence >= 0.55f) remove(QualityIssue.LOW_CONFIDENCE) else add(QualityIssue.LOW_CONFIDENCE)
        }
        return fallbackDetection.copy(
            contour = corners,
            corners = corners,
            confidence = confidence,
            qualityIssues = issues,
            inferenceMillis = (System.nanoTime() - started) / 1_000_000,
        )
    }

    private fun extractCorners(mask: Mat): List<NormalizedPoint> {
        val contours = mutableListOf<MatOfPoint>()
        Imgproc.findContours(mask, contours, Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)
        val largest = contours.maxByOrNull { abs(Imgproc.contourArea(it)) } ?: return emptyList()
        val curve = MatOfPoint2f(*largest.toArray())
        val approximation = MatOfPoint2f()
        Imgproc.approxPolyDP(curve, approximation, Imgproc.arcLength(curve, true) * 0.02, true)
        val raw = approximation.toArray().toList()
        val points = if (raw.size == 4) raw else {
            val rectangle = Imgproc.minAreaRect(curve)
            Array(4) { Point() }.also(rectangle::points).toList()
        }
        curve.release()
        approximation.release()
        contours.forEach { it.release() }
        val ordered = listOf(
            points.minBy { it.x + it.y },
            points.maxBy { it.x - it.y },
            points.maxBy { it.x + it.y },
            points.minBy { it.x - it.y },
        )
        return ordered.map {
            NormalizedPoint((it.x / mask.cols()).toFloat(), (it.y / mask.rows()).toFloat()).clamped()
        }
    }

    private fun extractHeatmapCorners(shape: IntArray, values: ByteBuffer): List<NormalizedPoint> {
        if (shape.size != 4) return emptyList()
        val channelsLast = shape[3] == 4
        val channelsFirst = shape[1] == 4
        if (!channelsLast && !channelsFirst) return emptyList()
        val height = if (channelsLast) shape[1] else shape[2]
        val width = if (channelsLast) shape[2] else shape[3]
        if (height <= 1 || width <= 1) return emptyList()
        values.rewind()
        val floats = FloatArray(height * width * 4)
        for (index in floats.indices) floats[index] = values.float
        return (0 until 4).map { channel ->
            var bestIndex = 0
            var bestValue = Float.NEGATIVE_INFINITY
            for (y in 0 until height) {
                for (x in 0 until width) {
                    val index = if (channelsLast) {
                        (y * width + x) * 4 + channel
                    } else {
                        channel * height * width + y * width + x
                    }
                    if (floats[index] > bestValue) {
                        bestValue = floats[index]
                        bestIndex = y * width + x
                    }
                }
            }
            NormalizedPoint(
                x = (bestIndex % width).toFloat() / (width - 1),
                y = (bestIndex / width).toFloat() / (height - 1),
            )
        }
    }

    override fun close() = interpreter.close()
}
