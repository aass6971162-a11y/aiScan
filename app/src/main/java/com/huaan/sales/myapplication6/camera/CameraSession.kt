package com.huaan.sales.myapplication6.camera

import android.content.Context
import android.os.SystemClock
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.core.UseCaseGroup
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import com.huaan.sales.myapplication6.domain.DocumentDetection
import com.huaan.sales.myapplication6.domain.QualityIssue
import com.huaan.sales.myapplication6.vision.DetectionSmoother
import com.huaan.sales.myapplication6.vision.PreviewDetector
import org.opencv.core.CvType
import org.opencv.core.Core
import org.opencv.core.Mat
import java.io.File
import java.nio.ByteBuffer
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

class CameraSession(
    private val context: Context,
    private val lifecycleOwner: LifecycleOwner,
    private val previewView: PreviewView,
    private val detector: PreviewDetector,
    private val onDetection: (DocumentDetection) -> Unit,
    private val onError: (String) -> Unit,
) : AutoCloseable {
    private val cameraExecutor: ExecutorService = Executors.newSingleThreadExecutor()
    // Saving/resizing a JPEG must not starve the latest-frame analyzer.
    private val captureExecutor: ExecutorService = Executors.newSingleThreadExecutor()
    private val isAnalyzing = AtomicBoolean(false)
    private val smoother = DetectionSmoother()
    private var cameraProvider: ProcessCameraProvider? = null
    private var camera: Camera? = null
    private var imageCapture: ImageCapture? = null
    private var flashSetting = FlashSetting.OFF
    private var captureSize = CaptureSize.Default
    private var lastAnalysisMillis = 0L
    private var lastRotation = -1
    private var lastAnalyzedGeneration = -1
    @Volatile private var bindingGeneration = 0
    @Volatile private var closed = false
    @Volatile private var borderScanEnabled = false

    fun setBorderScanEnabled(enabled: Boolean) {
        if (borderScanEnabled == enabled) {
            if (!enabled) smoother.reset()
            return
        }
        borderScanEnabled = enabled
        if (!enabled) smoother.reset()
        // ImageAnalysis keeps a third camera stream open. Leaving it bound makes
        // still capture wait, even when every frame is discarded.
        if (cameraProvider == null) return
        runCatching { bindUseCases() }
            .onFailure { error -> onError(error.message ?: "无法切换边框扫描") }
    }

    fun start(flash: FlashSetting = FlashSetting.OFF, size: CaptureSize = CaptureSize.Default) {
        flashSetting = flash
        captureSize = size
        val future = ProcessCameraProvider.getInstance(context)
        future.addListener(
            {
                if (closed) return@addListener
                runCatching {
                    val provider = future.get()
                    cameraProvider = provider
                    bindUseCases()
                }.onFailure { error ->
                    onError(error.message ?: "无法启动相机")
                }
            },
            ContextCompat.getMainExecutor(context),
        )
    }

    fun updateFlash(setting: FlashSetting) {
        flashSetting = setting
        applyFlash()
    }

    fun updateResolution(size: CaptureSize) {
        if (size == captureSize) return
        captureSize = size
        if (cameraProvider == null) return
        runCatching { bindUseCases() }
            .onFailure { error -> onError(error.message ?: "无法切换分辨率") }
    }

    @androidx.annotation.OptIn(androidx.camera.core.ExperimentalZeroShutterLag::class)
    private fun bindUseCases() {
        if (closed) return
        val provider = cameraProvider ?: return
        val generation = ++bindingGeneration
        previewView.scaleType = PreviewView.ScaleType.FIT_CENTER
        val displayRotation = previewView.display.rotation
        val preview = Preview.Builder()
            .setTargetRotation(displayRotation)
            .build()
            .also { it.surfaceProvider = previewView.surfaceProvider }
        val captureBuilder = ImageCapture.Builder()
            .setTargetRotation(displayRotation)
            .setCaptureMode(ImageCapture.CAPTURE_MODE_ZERO_SHUTTER_LAG)
        captureBuilder.setResolutionSelector(captureSize.toResolutionSelector())
        imageCapture = captureBuilder.build()
        provider.unbindAll()
        val group = UseCaseGroup.Builder()
            .addUseCase(preview)
            .addUseCase(checkNotNull(imageCapture))
            .apply {
                if (borderScanEnabled) {
                    val analysis = ImageAnalysis.Builder()
                        .setTargetRotation(displayRotation)
                        .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .build()
                    analysis.setAnalyzer(cameraExecutor) { analyze(it, generation) }
                    addUseCase(analysis)
                }
            }
            .build()
        camera = provider.bindToLifecycle(lifecycleOwner, CameraSelector.DEFAULT_BACK_CAMERA, group)
        applyFlash()
    }

    private fun applyFlash() {
        val boundCamera = camera ?: return
        val capture = imageCapture ?: return
        if (!boundCamera.cameraInfo.hasFlashUnit()) {
            capture.flashMode = ImageCapture.FLASH_MODE_OFF
            if (flashSetting != FlashSetting.OFF) onError("当前设备没有闪光灯")
            return
        }
        val torchOn = flashSetting == FlashSetting.ON
        capture.flashMode = if (torchOn) ImageCapture.FLASH_MODE_OFF else flashSetting.cameraXMode
        boundCamera.cameraControl.enableTorch(torchOn)
    }

    fun capture(output: File, onSaved: (File) -> Unit, onFailed: () -> Unit = {}) {
        val capture = imageCapture ?: return run {
            onError("相机尚未准备好")
            onFailed()
        }
        val options = ImageCapture.OutputFileOptions.Builder(output).build()
        capture.takePicture(
            options,
            captureExecutor,
            object : ImageCapture.OnImageSavedCallback {
                override fun onImageSaved(outputFileResults: ImageCapture.OutputFileResults) {
                    ContextCompat.getMainExecutor(context).execute { onSaved(output) }
                }

                override fun onError(exception: ImageCaptureException) {
                    ContextCompat.getMainExecutor(context).execute {
                        onFailed()
                        onError(exception.message ?: "拍摄失败")
                    }
                }
            },
        )
    }

    private fun analyze(image: ImageProxy, generation: Int) {
        val now = SystemClock.elapsedRealtime()
        if (!borderScanEnabled || closed || generation != bindingGeneration || now - lastAnalysisMillis < 300L) {
            image.close()
            return
        }
        if (!isAnalyzing.compareAndSet(false, true)) {
            image.close()
            return
        }
        try {
            lastAnalysisMillis = now
            if (lastRotation != image.imageInfo.rotationDegrees || lastAnalyzedGeneration != generation) {
                smoother.reset()
                lastRotation = image.imageInfo.rotationDegrees
                lastAnalyzedGeneration = generation
            }
            val bgr = image.toOrientedBgrMat()
            val detection = try {
                smoother.update(detector.detectColor(bgr), SystemClock.elapsedRealtime())
            } finally { bgr.release() }
            ContextCompat.getMainExecutor(context).execute {
                if (!closed && borderScanEnabled && generation == bindingGeneration) onDetection(detection)
            }
        } catch (error: Throwable) {
            smoother.reset()
            val sourceWidth = image.width
            val sourceHeight = image.height
            ContextCompat.getMainExecutor(context).execute {
                if (!closed && borderScanEnabled && generation == bindingGeneration) {
                    onDetection(DocumentDetection(emptyList(), emptyList(), 0f,
                        setOf(QualityIssue.NO_DOCUMENT, QualityIssue.LOW_CONFIDENCE),
                        0, sourceWidth, sourceHeight, 0))
                    onError(error.message ?: "实时检测失败")
                }
            }
        } finally {
            isAnalyzing.set(false)
            image.close()
        }
    }

    override fun close() {
        closed = true
        bindingGeneration++
        camera?.cameraControl?.enableTorch(false)
        cameraProvider?.unbindAll()
        camera = null
        cameraExecutor.shutdown()
        captureExecutor.shutdown()
    }
}

private fun ImageProxy.toOrientedBgrMat(): Mat {
    val plane = planes.first()
    return rgbaPlaneToOrientedBgr(plane.buffer, width, height, plane.rowStride,
        plane.pixelStride, imageInfo.rotationDegrees)
}

internal fun rgbaPlaneToOrientedBgr(
    planeBuffer: ByteBuffer,
    width: Int,
    height: Int,
    rowStride: Int,
    pixelStride: Int,
    rotationDegrees: Int,
): Mat {
    val buffer = planeBuffer.duplicate().apply { rewind() }
    require(width > 0 && height > 0 && pixelStride >= 4 && rowStride >= width * pixelStride) {
        "相机没有返回有效的 RGBA 分析帧"
    }
    val row = ByteArray(rowStride)
    val bgrBytes = ByteArray(width * height * 3)
    for (y in 0 until height) {
        buffer.position(y * rowStride)
        val available = minOf(rowStride, buffer.remaining())
        require(available >= (width - 1) * pixelStride + 4) { "相机分析帧数据不完整" }
        buffer.get(row, 0, available)
        for (x in 0 until width) {
            val source = x * pixelStride
            val target = (y * width + x) * 3
            // CameraX 1.6.2 ImageAnalysis API specifies increasing byte indices R, G, B, A.
            bgrBytes[target] = row[source + 2]
            bgrBytes[target + 1] = row[source + 1]
            bgrBytes[target + 2] = row[source]
        }
    }
    val raw = Mat(height, width, CvType.CV_8UC3)
    raw.put(0, 0, bgrBytes)
    val oriented = Mat()
    when ((rotationDegrees % 360 + 360) % 360) {
        90 -> Core.rotate(raw, oriented, Core.ROTATE_90_CLOCKWISE)
        180 -> Core.rotate(raw, oriented, Core.ROTATE_180)
        270 -> Core.rotate(raw, oriented, Core.ROTATE_90_COUNTERCLOCKWISE)
        else -> raw.copyTo(oriented)
    }
    raw.release()
    return oriented
}
