package com.huaan.sales.myapplication6.vision

import com.huaan.sales.myapplication6.domain.CorrectionResult
import com.huaan.sales.myapplication6.domain.DocumentGeometry
import com.huaan.sales.myapplication6.domain.DraftPage
import com.huaan.sales.myapplication6.domain.NormalizedPoint
import com.huaan.sales.myapplication6.domain.ProcessingOptions
import com.huaan.sales.myapplication6.ml.ModelRuntimeStatus
import com.huaan.sales.myapplication6.ml.OnnxDewarper
import org.opencv.android.Utils
import org.opencv.core.Mat
import org.opencv.core.MatOfByte
import org.opencv.core.MatOfInt
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc
import java.io.File
import java.io.FileOutputStream
import java.util.UUID

class DocumentProcessor(
    private val workingDirectory: File,
    private val detector: PreviewDetector,
    private val modelStatus: ModelRuntimeStatus,
    private val dewarper: OnnxDewarper? = null,
    private val stillDetector: StillDocumentDetector = StillDocumentDetector { detector.detectColor(it) },
) {
    init {
        workingDirectory.mkdirs()
    }

    /** Open an unmodified photo immediately; detection is only needed for an enabled crop. */
    fun createFullFrameDraft(original: File): DraftPage {
        orientedImageSize(original)
        val id = UUID.randomUUID().toString()
        return DraftPage(
            id = id,
            originalPath = original.absolutePath,
            previewPath = File(workingDirectory, "draft-$id.jpg").absolutePath,
            geometry = DocumentGeometry.fromCorners(listOf(
                NormalizedPoint(0f, 0f), NormalizedPoint(1f, 0f),
                NormalizedPoint(1f, 1f), NormalizedPoint(0f, 1f),
            )),
            options = ProcessingOptions(removeShadows = false, cropEnabled = false),
            detectionConfidence = 0f,
            processingMillis = 0L,
            modelVersion = "original",
            hasProcessedPreview = false,
        )
    }

    fun createDraft(original: File): DraftPage {
        val started = System.nanoTime()
        // Model inference and edge refinement already use smaller working images.
        // Keep normalized geometry; only final correction decodes the original pixels.
        val bitmap = decodeOrientedBitmap(original, maxEdge = 2048)
        val source = Mat()
        val bgr = Mat()
        val detection = try {
            Utils.bitmapToMat(bitmap, source)
            Imgproc.cvtColor(source, bgr, Imgproc.COLOR_RGBA2BGR)
            stillDetector.detect(bgr)
        } finally {
            bgr.release()
            source.release()
            bitmap.recycle()
        }

        val corners = if (detection.corners.size == 4) {
            detection.corners
        } else {
            listOf(
                NormalizedPoint(0.06f, 0.06f),
                NormalizedPoint(0.94f, 0.06f),
                NormalizedPoint(0.94f, 0.94f),
                NormalizedPoint(0.06f, 0.94f),
            )
        }
        val id = UUID.randomUUID().toString()
        val preview = File(workingDirectory, "draft-$id.jpg")
        val geometry = DocumentGeometry.fromCorners(corners)
        return DraftPage(
            id = id,
            originalPath = original.absolutePath,
            previewPath = preview.absolutePath,
            geometry = geometry,
            options = ProcessingOptions(removeShadows = false, cropEnabled = false),
            detectionConfidence = detection.confidence,
            processingMillis = (System.nanoTime() - started) / 1_000_000,
            modelVersion = "${detection.detectionSource.name.lowercase()}-still-boundary-v1",
            hasProcessedPreview = false,
            detectionIssues = detection.qualityIssues + if (detection.requiresConfirmation) {
                setOf(com.huaan.sales.myapplication6.domain.QualityIssue.LOW_CONFIDENCE)
            } else emptySet(),
            boundaryDetectionComplete = true,
        )
    }

    fun processDraft(draft: DraftPage, enforceBoundary: Boolean = true): DraftPage {
        if (enforceBoundary && draft.options.cropEnabled) {
            check(!draft.boundaryDetectionPending) { "正在检测边框，请稍候或手动调整" }
            check(!draft.requiresBoundaryConfirmation) { "边框需要确认，请调整边框或确认当前裁剪范围" }
        }
        val result = process(
            pageId = draft.id,
            original = File(draft.originalPath),
            output = File(draft.previewPath),
            geometry = draft.geometry,
            options = draft.options,
            warp = draft.options.cropEnabled,
        )
        return draft.copy(
            processingMillis = result.processingMillis,
            modelVersion = result.modelVersion,
            hasProcessedPreview = true,
        )
    }

    /** Bounded full-frame preview; the final save still reads the full-resolution original. */
    fun applyFullFrameEffect(draft: DraftPage): DraftPage {
        val effectFile = File(workingDirectory, "effect-${draft.id}.jpg")
        val result = process(
            pageId = draft.id,
            original = File(draft.originalPath),
            output = effectFile,
            geometry = draft.geometry,
            options = draft.options,
            warp = false,
            decodeMaxEdge = 1600,
            sourceRotationDegrees = 0,
        )
        return draft.copy(
            processingMillis = result.processingMillis,
            modelVersion = result.modelVersion,
            hasProcessedPreview = false,
            effectPreviewPath = effectFile.absolutePath,
        )
    }

    /** Only the editor reads this bounded preview; final output always reads the original. */
    fun createEditPreview(draft: DraftPage): DraftPage {
        if (draft.options.cropEnabled) {
            check(!draft.boundaryDetectionPending && !draft.requiresBoundaryConfirmation) {
                "边框需要确认，请调整边框或确认当前裁剪范围"
            }
        }
        val output = File(workingDirectory, "crop-preview-${draft.id}.jpg")
        process(draft.id, File(draft.originalPath), output, draft.geometry, draft.options,
            warp = draft.options.cropEnabled, decodeMaxEdge = 1600)
        return draft.copy(cropPreviewPath = output.absolutePath, hasProcessedPreview = false)
    }

    fun process(
        pageId: String,
        original: File,
        output: File,
        geometry: DocumentGeometry,
        options: ProcessingOptions,
        warp: Boolean = true,
        decodeMaxEdge: Int = Int.MAX_VALUE,
        sourceRotationDegrees: Int = geometry.sourceRotationDegrees,
    ): CorrectionResult {
        val started = System.nanoTime()
        // An untouched page is a byte-for-byte copy, including its camera EXIF.
        // Do not send a full-frame rectangle through warpPerspective: it loses edge pixels.
        if (!warp && decodeMaxEdge == Int.MAX_VALUE && sourceRotationDegrees == 0 &&
            options.effect == com.huaan.sales.myapplication6.domain.PageEffect.ORIGINAL &&
            !options.removeShadows && options.colorMode == com.huaan.sales.myapplication6.domain.ColorMode.COLOR) {
            val (width, height) = orientedImageSize(original)
            output.parentFile?.mkdirs()
            val temporary = File(output.parentFile, "${output.name}.tmp")
            try {
                original.inputStream().use { input ->
                    FileOutputStream(temporary).use { destination ->
                        input.copyTo(destination)
                        destination.fd.sync()
                    }
                }
                check(temporary.renameTo(output)) { "无法保存原始图像" }
            } finally { temporary.delete() }
            return CorrectionResult(output.absolutePath, pageId, width, height,
                (System.nanoTime() - started) / 1_000_000, "original", geometry.manuallyAdjusted)
        }
        val source = decodeOrientedMat(original, maxEdge = decodeMaxEdge,
            additionalRotationDegrees = sourceRotationDegrees)
        val corrected = if (!warp) {
            source
        } else {
            try {
                if (options.dewarpEnabled && options.dewarpStrength > 0f) {
                    check(dewarper != null) { "AI 展平当前不可用，请关闭展平后重试" }
                }
                check(!options.dewarpEnabled || options.perspectiveEnabled) { "请先开启透视校正再使用褶皱展平" }
                OriginalResolutionWarp(source, geometry, options.perspectiveEnabled).use {
                    it.render(if (options.dewarpEnabled) dewarper else null, options.dewarpStrength)
                }
            } finally {
                source.release()
            }
        }
        val enhanced = if (options.effect == com.huaan.sales.myapplication6.domain.PageEffect.ORIGINAL &&
            !options.removeShadows && options.colorMode == com.huaan.sales.myapplication6.domain.ColorMode.COLOR) {
            corrected
        } else {
            try { DocumentEffects.apply(corrected, options) } finally { corrected.release() }
        }
        val width = enhanced.cols()
        val height = enhanced.rows()
        try {
            writeMatAtomically(enhanced, output, options.jpegQuality)
        } finally {
            enhanced.release()
        }
        return CorrectionResult(
            imagePath = output.absolutePath,
            pageId = pageId,
            width = width,
            height = height,
            processingMillis = (System.nanoTime() - started) / 1_000_000,
            modelVersion = listOfNotNull(cornerVersion(),
                if (warp && options.dewarpEnabled && options.dewarpStrength > 0) "uvdoc-026782d-grid-fp32" else null,
                DocumentEffects.VERSION).joinToString("+"),
            manuallyAdjusted = geometry.manuallyAdjusted,
        )
    }

    private fun writeMatAtomically(image: Mat, target: File, quality: Int) {
        target.parentFile?.mkdirs()
        val temporary = File(target.parentFile, "${target.name}.tmp")
        val encoded = MatOfByte()
        val parameters = MatOfInt(Imgcodecs.IMWRITE_JPEG_QUALITY, quality.coerceIn(80, 100))
        try {
            check(Imgcodecs.imencode(".jpg", image, encoded, parameters)) { "无法编码校正图像" }
            FileOutputStream(temporary).use { output ->
                output.write(encoded.toArray())
                output.fd.sync()
            }
        } finally {
            encoded.release()
            parameters.release()
        }
        check(temporary.renameTo(target)) { "无法写入校正图像" }
    }

    private fun cornerVersion() = if (modelStatus.hasTrainedModels) "docaligner-1.1.1-lcnet100-fp32-search-v5" else "opencv-fallback-2"
}
