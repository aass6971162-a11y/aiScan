package com.huaan.sales.myapplication6

import android.app.Application
import android.content.Context
import android.net.Uri
import android.os.SystemClock
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.huaan.sales.myapplication6.camera.CaptureSize
import com.huaan.sales.myapplication6.camera.FlashSetting
import com.huaan.sales.myapplication6.data.PdfExporter
import com.huaan.sales.myapplication6.data.ScanRepository
import com.huaan.sales.myapplication6.data.cachedThumbnailFile
import com.huaan.sales.myapplication6.domain.ColorMode
import com.huaan.sales.myapplication6.domain.PageEffect
import com.huaan.sales.myapplication6.domain.DocumentDetection
import com.huaan.sales.myapplication6.domain.DraftPage
import com.huaan.sales.myapplication6.domain.ProcessingOptions
import com.huaan.sales.myapplication6.domain.ScanDocument
import com.huaan.sales.myapplication6.domain.ScanPage
import com.huaan.sales.myapplication6.ml.InferenceBackend
import com.huaan.sales.myapplication6.ml.OnnxImageModel
import com.huaan.sales.myapplication6.ml.OnnxCornerDetector
import com.huaan.sales.myapplication6.ml.OnnxDewarper
import com.huaan.sales.myapplication6.ml.ModelAssetRegistry
import com.huaan.sales.myapplication6.ml.ModelRuntimeStatus
import com.huaan.sales.myapplication6.vision.DocumentProcessor
import com.huaan.sales.myapplication6.vision.OpenCvDocumentDetector
import com.huaan.sales.myapplication6.vision.PreviewDetector
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.CoroutineStart
import kotlinx.coroutines.Deferred
import kotlinx.coroutines.async
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import java.io.File
import java.io.FileOutputStream
import java.util.UUID
import org.opencv.core.Mat
import org.opencv.core.CvType
import org.opencv.core.Scalar

enum class ScannerScreen { HOME, CAMERA, PROCESSING, EDITOR, PAGES }

enum class CaptureMode { SINGLE, BURST }

data class BurstItem(
    val path: String,
    val pageId: String? = null,
    val preview: android.graphics.Bitmap? = null,
)

data class ScannerUiState(
    val screen: ScannerScreen = ScannerScreen.HOME,
    val detection: DocumentDetection? = null,
    val draft: DraftPage? = null,
    val pages: List<ScanPage> = emptyList(),
    val documents: List<ScanDocument> = emptyList(),
    val activeDocumentId: String? = null,
    val documentCreatedAtMillis: Long? = null,
    val modelStatus: ModelRuntimeStatus,
    val errorMessage: String? = null,
    val sharePdfPath: String? = null,
    val isApplying: Boolean = false,
    val flashSetting: FlashSetting = FlashSetting.OFF,
    val captureSize: CaptureSize = CaptureSize.Default,
    val autoCapture: Boolean = false,
    val autoCrop: Boolean = false,
    val documentCorrection: Boolean = false,
    val editorCropActive: Boolean = false,
    val captureMode: CaptureMode = CaptureMode.SINGLE,
    val burstShots: List<BurstItem> = emptyList(),
    val pendingAutoCapture: Boolean = false,
    val retakePageId: String? = null,
    val retakeBurstPath: String? = null,
    val returnToPages: Boolean = false,
    val processingMessage: String = "正在处理图片",
)

class ScannerViewModel(application: Application) : AndroidViewModel(application) {
    private val registry = ModelAssetRegistry(application)
    private val initialModelStatus = ModelRuntimeStatus(InferenceBackend.OPENCV_FALLBACK,
        "loading", emptySet(), emptySet(), "正在加载离线 AI 模型…")
    private val fallbackDetector = OpenCvDocumentDetector()
    // The camera confirms model predictions across scales; OpenCV supplies image quality hints.
    val previewDetector: PreviewDetector = object : PreviewDetector {
        override fun detect(gray: Mat, rotationDegrees: Int) = fallbackDetector.detect(gray, rotationDegrees)
        override fun detectColor(bgr: Mat) = cornerModel?.detectPreviewColor(bgr)
            ?: fallbackDetector.detectColor(bgr)
    }
    private val cameraSettings = application.getSharedPreferences("scanner_camera", Context.MODE_PRIVATE)
    private val workingDirectory = File(application.cacheDir, "scanner-drafts")
    private val repository = ScanRepository(application)
    private val exporter = PdfExporter(File(application.filesDir, "document_scanner/exports"))
    @Volatile private var processor = DocumentProcessor(workingDirectory, previewDetector, initialModelStatus)
    @Volatile private var cornerModel: OnnxCornerDetector? = null
    private var dewarpModel: OnnxDewarper? = null
    private val modelLock = Any()
    private var cleared = false
    private val processingMutex = Mutex()
    private val boundaryMutex = Mutex()
    private val boundaryDetections = mutableMapOf<String, Deferred<Result<DraftPage>>>()
    private var stableAccurateFrames = 0
    private var autoCaptureArmed = true
    private var captureModeBeforeRetake: CaptureMode? = null
    private var pendingBurstPath: String? = null
    private val _state = MutableStateFlow(
        run {
            val all = repository.loadPages()
            ScannerUiState(
                documents = all.toScanDocuments(),
                modelStatus = initialModelStatus,
                autoCapture = cameraSettings.getBoolean(KEY_AUTO_CAPTURE, false),
                autoCrop = cameraSettings.getBoolean(KEY_AUTO_CROP, false),
                documentCorrection = cameraSettings.getBoolean(KEY_DOCUMENT_CORRECTION, false),
            )
        },
    )
    val state: StateFlow<ScannerUiState> = _state.asStateFlow()

    private val modelInitialization = viewModelScope.launch(Dispatchers.IO) {
        val ready = mutableSetOf<String>()
        val failed = mutableSetOf<String>()
        val errors = mutableListOf<String>()
        val probe = Mat(256, 256, CvType.CV_8UC3, Scalar(127.0, 127.0, 127.0))
        var corners: OnnxCornerDetector? = null
        var dewarp: OnnxDewarper? = null
        try {
            runCatching {
                val model = OnnxImageModel(registry.openOnnxSession("document_corners"))
                try {
                    model.run(probe, 256, 256, rgb = false)
                    OnnxCornerDetector(model, OpenCvDocumentDetector())
                } catch (error: Throwable) { model.close(); throw error }
            }.onSuccess { corners = it; ready += "document_corners" }
                .onFailure {
                    android.util.Log.e("ScannerModels", "Corner model initialization failed", it)
                    failed += "document_corners"; errors += "AI 边框加载失败，可手动调整"
                }
            runCatching {
                val model = OnnxImageModel(registry.openOnnxSession("document_dewarp"))
                try {
                    // Run the actual graph, checking operators without accepting a blank-image grid.
                    model.run(probe, 488, 712, rgb = true)
                    OnnxDewarper(model)
                } catch (error: Throwable) { model.close(); throw error }
            }.onSuccess { dewarp = it; ready += "document_dewarp" }
                .onFailure {
                    android.util.Log.e("ScannerModels", "Dewarp model initialization failed", it)
                    failed += "document_dewarp"; errors += "AI 展平加载失败"
                }
        } finally { probe.release() }
        val status = ModelRuntimeStatus(
            backend = if (ready.isEmpty()) InferenceBackend.OPENCV_FALLBACK else InferenceBackend.ONNX_CPU,
            modelVersion = listOfNotNull(if (corners != null) "docaligner-1.1.1-lcnet100-search-v5" else "opencv-fallback-2",
                if (dewarp != null) "uvdoc-026782d-fp32" else null).joinToString("+"),
            readyModelIds = ready.toSet(), missingModelIds = failed.toSet(),
            message = listOf(if (corners != null) "AI 实时辅助与拍照边框已就绪" else "手动裁剪可用",
                if (dewarp != null) "AI 展平已就绪" else "AI 展平不可用", "基础去阴影").joinToString(" · "),
        )
        synchronized(modelLock) {
            if (cleared) { corners?.close(); dewarp?.close() } else {
                cornerModel = corners
                dewarpModel = dewarp
                processor = DocumentProcessor(workingDirectory, corners ?: fallbackDetector, status, dewarp)
                _state.update { it.copy(modelStatus = status, errorMessage = errors.takeIf { it.isNotEmpty() }?.joinToString("；")) }
            }
        }
    }

    fun newCaptureFile(): File {
        workingDirectory.mkdirs()
        return File(workingDirectory, "capture-${UUID.randomUUID()}.jpg")
    }

    fun onDetection(detection: DocumentDetection) {
        val accurate = detection.isUsable && detection.qualityIssues.isEmpty()
        _state.update { state ->
            if (!state.autoCrop) {
                stableAccurateFrames = 0
                return@update state.copy(detection = null, pendingAutoCapture = false)
            }
            if (state.screen != ScannerScreen.CAMERA || !state.autoCapture || state.pendingAutoCapture) {
                stableAccurateFrames = 0
                return@update state.copy(detection = detection)
            }
            if (!accurate) {
                stableAccurateFrames = 0
                autoCaptureArmed = true
                return@update state.copy(detection = detection)
            }
            if (!autoCaptureArmed) return@update state.copy(detection = detection)
            stableAccurateFrames += 1
            if (stableAccurateFrames < AUTO_CAPTURE_STABLE_FRAMES) return@update state.copy(detection = detection)
            stableAccurateFrames = 0
            autoCaptureArmed = false
            state.copy(detection = detection, pendingAutoCapture = true)
        }
    }

    fun consumeAutoCapture() = _state.update { it.copy(pendingAutoCapture = false) }

    fun rearmAutoCapture() {
        autoCaptureArmed = true
        stableAccurateFrames = 0
    }

    fun disarmAutoCapture() {
        autoCaptureArmed = false
        stableAccurateFrames = 0
    }

    fun onCameraError(message: String) {
        _state.update { it.copy(errorMessage = message) }
    }

    /** Deliver the untouched camera JPEG first; neither switch may overwrite it. */
    fun acceptCapturedPhoto(file: File, burst: Boolean, replacingBurst: Boolean) {
        deliverCapturedPhoto(file, burst, replacingBurst)
    }

    private fun deliverCapturedPhoto(file: File, burst: Boolean, replacingBurst: Boolean) {
        when {
            replacingBurst -> replaceRetakenBurstShot(file)
            burst -> commitBurstShot(file)
            else -> processCapture(file)
        }
    }

    fun processCapture(file: File) {
        val started = SystemClock.elapsedRealtime()
        val replaceId = _state.value.retakePageId
        runCatching {
            val created = processor.createFullFrameDraft(file).let {
                it.copy(options = it.options.copy(
                    cropEnabled = _state.value.autoCrop,
                    perspectiveEnabled = _state.value.documentCorrection,
                ))
            }
            if (replaceId == null) created else created.copy(id = replaceId)
        }.onSuccess { draft ->
            Log.d("ScannerPerf", "capture-to-editor=${SystemClock.elapsedRealtime() - started}ms")
            _state.update { current ->
                current.copy(
                    screen = ScannerScreen.EDITOR,
                    draft = draft,
                    isApplying = false,
                    editorCropActive = draft.options.cropEnabled && draft.cropPreviewPath == null,
                    retakePageId = null,
                    retakeBurstPath = null,
                    documentCreatedAtMillis = current.documentCreatedAtMillis ?: System.currentTimeMillis(),
                )
            }
            if (draft.options.cropEnabled && !draft.boundaryDetectionComplete) {
                detectEditorBoundary(draft)
            }
        }.onFailure { error ->
            file.delete()
            _state.update { current ->
                current.copy(
                    screen = ScannerScreen.CAMERA,
                    isApplying = false,
                    errorMessage = error.message ?: "文档处理失败",
                    burstShots = current.burstShots.filterNot { shot -> shot.path == file.absolutePath },
                )
            }
        }
    }

    fun updateControlPoint(index: Int, x: Float, y: Float) {
        _state.update { current ->
            if (current.isApplying) return@update current
            val draft = current.draft ?: return@update current
            draft.cropPreviewPath?.let { File(it).delete() }
            current.copy(
                draft = draft.copy(
                    geometry = draft.geometry.withControlPoint(
                        index,
                        com.huaan.sales.myapplication6.domain.NormalizedPoint(x, y),
                    ),
                    hasProcessedPreview = false,
                    cropPreviewPath = null,
                    boundaryEditRevision = draft.boundaryEditRevision + 1,
                    boundaryEditedByUser = true,
                ),
            )
        }
    }

    fun updateOptions(transform: (ProcessingOptions) -> ProcessingOptions) {
        _state.update { current ->
            if (current.isApplying) return@update current
            val draft = current.draft ?: return@update current
            val options = transform(draft.options)
            if (options == draft.options) return@update current
            draft.effectPreviewPath?.let { File(it).delete() }
            draft.cropPreviewPath?.let { File(it).delete() }
            current.copy(draft = draft.copy(options = options, hasProcessedPreview = false,
                effectPreviewPath = null, cropPreviewPath = null))
        }
    }

    fun setColorMode(mode: ColorMode) = updateOptions { it.copy(colorMode = mode) }

    fun setPageEffect(effect: PageEffect) {
        if (_state.value.isApplying) return
        val draft = _state.value.draft ?: return
        if (draft.boundaryDetectionPending) return
        draft.cropPreviewPath?.let { File(it).delete() }
        val updated = draft.copy(options = draft.options.withEffect(effect), hasProcessedPreview = false,
            effectPreviewPath = null, cropPreviewPath = null)
        if (effect == PageEffect.ORIGINAL) {
            draft.effectPreviewPath?.let { File(it).delete() }
            _state.update { it.copy(draft = updated, errorMessage = null) }
            if (updated.options.cropEnabled && !_state.value.editorCropActive) refreshEditorPreview()
            return
        }
        _state.update { it.copy(draft = updated, isApplying = true, errorMessage = null) }
        viewModelScope.launch(Dispatchers.Default) {
            processingMutex.withLock {
                runCatching {
                    if (updated.options.cropEnabled && !_state.value.editorCropActive) processor.createEditPreview(updated)
                    else processor.applyFullFrameEffect(updated)
                }
                    .onSuccess { processed ->
                        _state.update { it.copy(draft = processed, isApplying = false) }
                    }
                    .onFailure { error ->
                        _state.update {
                            it.copy(draft = draft, isApplying = false, errorMessage = error.message ?: "效果处理失败")
                        }
                    }
            }
        }
    }

    fun rotateLeft() {
        _state.update { current ->
            if (current.isApplying) return@update current
            val draft = current.draft ?: return@update current
            draft.cropPreviewPath?.let { File(it).delete() }
            // Effects use EXIF-oriented source coordinates; the editor rotates the
            // cached image. Turning never reruns filters or recompresses the original.
            current.copy(
                draft = draft.copy(
                    geometry = draft.geometry.rotatedLeft(),
                    hasProcessedPreview = false,
                    cropPreviewPath = null,
                ),
                errorMessage = null,
            )
        }
        if (_state.value.draft?.options?.cropEnabled == true && !_state.value.editorCropActive) refreshEditorPreview()
    }

    fun confirmBoundary() {
        val draft = _state.value.draft ?: return
        if (_state.value.isApplying || draft.boundaryDetectionPending) return
        val confirmed = draft.copy(boundaryConfirmed = true)
        _state.update { it.copy(draft = confirmed, errorMessage = null) }
    }

    /** Confirm parameters and make a small preview. Only Complete writes original-resolution output. */
    fun confirmEditorCrop() {
        if (_state.value.isApplying || !_state.value.editorCropActive) return
        val draft = _state.value.draft?.copy(boundaryConfirmed = true) ?: return
        if (!allowProcessing(draft)) return
        _state.update { it.copy(isApplying = true, errorMessage = null) }
        viewModelScope.launch(Dispatchers.IO) {
            processingMutex.withLock {
                runCatching {
                    processor.createEditPreview(draft.copy(boundaryConfirmed = true))
                }.onSuccess { processed ->
                    _state.update {
                        it.copy(
                            screen = ScannerScreen.EDITOR,
                            draft = processed,
                            isApplying = false,
                            editorCropActive = false,
                        )
                    }
                }.onFailure { error ->
                    _state.update {
                        it.copy(isApplying = false, errorMessage = error.message ?: "裁剪失败")
                    }
                }
            }
        }
    }

    /** Editing a border never warps or encodes the full photograph. */
    fun startEditorCrop() {
        if (_state.value.isApplying) return
        val draft = _state.value.draft ?: return
        draft.cropPreviewPath?.let { File(it).delete() }
        val ready = draft.copy(options = draft.options.copy(cropEnabled = true),
            hasProcessedPreview = false, cropPreviewPath = null)
        _state.update { it.copy(draft = ready, editorCropActive = true, errorMessage = null) }
        if (!ready.boundaryDetectionComplete) {
            detectEditorBoundary(ready)
        }
    }

    fun keepFullPhoto() {
        if (_state.value.isApplying) return
        _state.update { current ->
            current.copy(editorCropActive = false, errorMessage = null, draft = current.draft?.let {
                it.cropPreviewPath?.let { path -> File(path).delete() }
                it.copy(options = it.options.copy(cropEnabled = false, dewarpEnabled = false),
                    boundaryDetectionPending = false, hasProcessedPreview = false, cropPreviewPath = null)
            })
        }
    }

    fun setDewarpEnabled(enabled: Boolean) {
        if (enabled && (_state.value.draft?.options?.cropEnabled != true ||
                "document_dewarp" !in _state.value.modelStatus.readyModelIds)) return
        updateOptions { it.copy(dewarpEnabled = enabled, perspectiveEnabled = it.perspectiveEnabled || enabled) }
        if (!_state.value.editorCropActive) refreshEditorPreview()
    }

    fun setPerspectiveEnabled(enabled: Boolean) {
        updateOptions { it.copy(perspectiveEnabled = enabled,
            dewarpEnabled = it.dewarpEnabled && enabled) }
        if (!_state.value.editorCropActive) refreshEditorPreview()
    }

    fun refreshEditorPreview() {
        if (_state.value.isApplying) return
        val draft = _state.value.draft ?: return
        if (!allowProcessing(draft)) return
        _state.update { it.copy(isApplying = true, errorMessage = null) }
        viewModelScope.launch(Dispatchers.Default) {
            processingMutex.withLock {
                runCatching { processor.createEditPreview(draft) }
                    .onSuccess { updated -> _state.update { it.copy(draft = updated, isApplying = false) } }
                    .onFailure { error -> _state.update { it.copy(isApplying = false,
                        errorMessage = error.message ?: "预览生成失败") } }
            }
        }
    }

    fun applyDraft() {
        if (_state.value.isApplying) return
        val draft = _state.value.draft ?: return
        if (!allowProcessing(draft)) return
        if (draft.hasProcessedPreview && File(draft.previewPath).isFile) return
        _state.update { it.copy(isApplying = true, errorMessage = null) }
        viewModelScope.launch(Dispatchers.Default) {
            processingMutex.withLock {
                runCatching { processor.processDraft(draft) }
                    .onSuccess { updated ->
                        _state.update { it.copy(draft = updated, isApplying = false) }
                    }
                    .onFailure { error ->
                        _state.update {
                            it.copy(isApplying = false, errorMessage = error.message ?: "重新处理失败")
                        }
                    }
            }
        }
    }

    fun saveDraft() {
        if (_state.value.isApplying) return
        val draft = _state.value.draft ?: return
        val singleShot = _state.value.captureMode == CaptureMode.SINGLE
        if (!allowProcessing(draft)) return
        val started = SystemClock.elapsedRealtime()
        _state.update { it.copy(isApplying = true, errorMessage = null) }
        viewModelScope.launch(Dispatchers.IO) {
            processingMutex.withLock {
                runCatching {
                    val unchangedPixels = !draft.options.cropEnabled &&
                        draft.options.displayEffect() == PageEffect.ORIGINAL &&
                        draft.geometry.sourceRotationDegrees == 0
                    val known = repository.loadPages().firstOrNull { it.id == draft.id }
                    val unchangedEdit = known != null && draft.originalPath == known.originalPath &&
                        draft.geometry == known.geometry && draft.options == known.options &&
                        !draft.hasProcessedPreview && File(known.correctedPath).isFile
                    val processed = if (unchangedPixels || unchangedEdit) draft
                        else if (draft.hasProcessedPreview && File(draft.previewPath).isFile) draft
                        else processor.processDraft(draft)
                    val processingFinished = SystemClock.elapsedRealtime()
                    val documentId = known?.documentId
                        ?: _state.value.activeDocumentId
                        ?: UUID.randomUUID().toString()
                    val page = when {
                        unchangedEdit -> known!!
                        unchangedPixels -> repository.saveOriginalDraft(processed, documentId,
                            known?.createdAtMillis ?: createdAtForNewPage(documentId))
                        known != null -> repository.updateDraft(processed)
                        else -> repository.saveDraft(processed, documentId, createdAtForNewPage(documentId))
                    }
                    val all = repository.loadPages()
                    Log.d("ScannerPerf", "save-process=${processingFinished - started}ms save-persist=${SystemClock.elapsedRealtime() - processingFinished}ms crop=${draft.options.cropEnabled}")
                    (all to page.id) to documentId
                }.onSuccess { (result, documentId) ->
                    val (pages, pageId) = result
                    val restoredMode = captureModeBeforeRetake
                    if (singleShot) captureModeBeforeRetake = null
                    val capturedPath = draft.originalPath
                    draft.effectPreviewPath?.let { File(it).delete() }
                    draft.cropPreviewPath?.let { File(it).delete() }
                    File(draft.previewPath).delete()
                    // The persisted original (or its hard link) is safe now; keep no
                    // second camera-size copy in cache after a completed single shot.
                    val capture = File(capturedPath)
                    if (capturedPath != pages.first { it.id == pageId }.originalPath &&
                        _state.value.burstShots.none { it.path == capturedPath } &&
                        capture.canonicalPath.startsWith(workingDirectory.canonicalPath + File.separator)) {
                        capture.delete()
                    }
                    val openedFromDocument = _state.value.returnToPages
                    _state.update {
                        it.withDocuments(pages, documentId).copy(
                            screen = if (singleShot || openedFromDocument) ScannerScreen.PAGES else ScannerScreen.CAMERA,
                            draft = null,
                            isApplying = false,
                            returnToPages = false,
                            retakePageId = if (singleShot || openedFromDocument) null else it.retakePageId,
                            captureMode = if (singleShot) restoredMode ?: it.captureMode else it.captureMode,
                            sharePdfPath = null,
                            burstShots = if (singleShot) {
                                it.burstShots.filterNot { shot -> shot.path == capturedPath }
                            } else {
                                it.burstShots.map { shot ->
                                    if (shot.path == capturedPath) shot.copy(pageId = pageId) else shot
                                }
                            },
                        )
                    }
                }.onFailure { error ->
                    _state.update {
                        it.copy(isApplying = false, errorMessage = error.message ?: "保存页面失败")
                    }
                }
            }
        }
    }

    fun showBurstShot(index: Int) {
        if (_state.value.isApplying) return
        val shots = _state.value.burstShots
        val shot = shots.getOrNull(index) ?: return
        val current = _state.value.draft
        if (current != null && (current.id == shot.pageId || current.originalPath == shot.path)) return
        _state.update { it.copy(isApplying = true, editorCropActive = false, errorMessage = null) }
        viewModelScope.launch(Dispatchers.IO) {
            current?.let {
                File(it.previewPath).delete()
                it.effectPreviewPath?.let { path -> File(path).delete() }
                it.cropPreviewPath?.let { path -> File(path).delete() }
            }
            runCatching {
                val page = shot.pageId?.let { id -> repository.loadPages().firstOrNull { it.id == id } }
                if (page != null) {
                    val loaded = repository.toDraft(page, workingDirectory)
                    File(loaded.previewPath).delete()
                    if (loaded.options.displayEffect() == PageEffect.ORIGINAL) {
                        loaded.copy(hasProcessedPreview = false)
                    } else {
                        processor.applyFullFrameEffect(loaded.copy(hasProcessedPreview = false))
                    }
                } else {
                    processor.createFullFrameDraft(File(shot.path))
                }
            }.onSuccess { draft ->
                _state.update {
                    it.copy(
                        screen = ScannerScreen.EDITOR,
                        draft = draft,
                        isApplying = false,
                        editorCropActive = false,
                    )
                }
            }.onFailure { error ->
                _state.update { it.copy(isApplying = false, errorMessage = error.message ?: "无法切换图片") }
            }
        }
    }

    fun editPage(page: ScanPage, fromDocument: Boolean = false) {
        if (_state.value.isApplying) return
        _state.update { it.copy(isApplying = true, errorMessage = null, editorCropActive = false) }
        viewModelScope.launch(Dispatchers.IO) {
            processingMutex.withLock {
                runCatching {
                    val loaded = repository.toDraft(page, workingDirectory)
                    File(loaded.previewPath).delete()
                    val draft = loaded.copy(
                        options = loaded.options.copy(
                            cropEnabled = page.options.cropEnabled || (!page.processed && _state.value.autoCrop),
                            perspectiveEnabled = if (!page.processed) _state.value.documentCorrection
                                else page.options.perspectiveEnabled,
                        ),
                        hasProcessedPreview = false,
                    )
                    if (draft.options.displayEffect() == PageEffect.ORIGINAL) draft else {
                        // Regenerate from the original using the current effect version on save.
                        processor.applyFullFrameEffect(draft)
                    }
                }.onSuccess { draft ->
                    _state.update {
                        it.copy(
                            screen = ScannerScreen.EDITOR,
                            draft = draft,
                            isApplying = false,
                            editorCropActive = draft.options.cropEnabled && !page.processed,
                            returnToPages = fromDocument,
                        )
                    }
                    if (draft.options.cropEnabled && !draft.boundaryDetectionComplete) {
                        detectEditorBoundary(draft)
                    }
                }.onFailure { error ->
                    _state.update { it.copy(isApplying = false, errorMessage = error.message ?: "无法打开页面") }
                }
            }
        }
    }

    fun deleteDocument(documentId: String) {
        viewModelScope.launch(Dispatchers.IO) {
            val all = repository.deleteDocument(documentId)
            _state.update { state ->
                val active = if (state.activeDocumentId == documentId) null else state.activeDocumentId
                state.withDocuments(all, active)
            }
        }
    }

    fun deletePage(pageId: String) {
        viewModelScope.launch(Dispatchers.IO) {
            val all = repository.delete(pageId)
            _state.update { it.withDocuments(all, it.activeDocumentId) }
        }
    }

    fun savePageOrder(orderedIds: List<String>) {
        val documentId = _state.value.activeDocumentId ?: return
        viewModelScope.launch(Dispatchers.IO) {
            val all = repository.reorder(documentId, orderedIds)
            _state.update { it.withDocuments(all, documentId) }
        }
    }

    fun movePage(pageId: String, offset: Int) {
        viewModelScope.launch(Dispatchers.IO) {
            val all = repository.move(pageId, offset)
            _state.update { it.withDocuments(all, it.activeDocumentId) }
        }
    }

    fun showHome() {
        val shots = _state.value.burstShots
        val pages = _state.value.pages
        captureModeBeforeRetake = null
        shots.forEach { shot ->
            val savedOriginal = shot.pageId?.let { id -> pages.firstOrNull { it.id == id }?.originalPath }
            if (shot.path != savedOriginal) File(shot.path).delete()
        }
        val all = repository.loadPages()
        _state.update {
            it.copy(
                screen = ScannerScreen.HOME,
                draft = null,
                detection = null,
                pages = emptyList(),
                documents = all.toScanDocuments(),
                activeDocumentId = null,
                documentCreatedAtMillis = null,
                burstShots = emptyList(),
                retakePageId = null,
                returnToPages = false,
            )
        }
    }

    fun startNewScan() {
        _state.update {
            it.copy(
                screen = ScannerScreen.CAMERA,
                draft = null,
                detection = null,
                pages = emptyList(),
                activeDocumentId = null,
                documentCreatedAtMillis = System.currentTimeMillis(),
            )
        }
    }

    fun openDocument(id: String) {
        val all = repository.loadPages()
        val createdAt = all.filter { it.documentId == id }.minOfOrNull { it.createdAtMillis }
        _state.update {
            it.withDocuments(all, id).copy(
                screen = ScannerScreen.PAGES,
                draft = null,
                detection = null,
                retakePageId = null,
                returnToPages = false,
                documentCreatedAtMillis = createdAt,
            )
        }
    }

    fun showCamera() = _state.update { it.copy(screen = ScannerScreen.CAMERA, draft = null, detection = null) }

    fun importImages(uris: List<Uri>) {
        if (uris.isEmpty() || _state.value.isApplying) return
        val documentId = _state.value.activeDocumentId ?: UUID.randomUUID().toString()
        val createdAt = _state.value.documentCreatedAtMillis ?: System.currentTimeMillis()
        _state.update {
            it.copy(
                isApplying = true,
                errorMessage = null,
                documentCreatedAtMillis = createdAt,
            )
        }
        viewModelScope.launch(Dispatchers.IO) {
            runCatching {
                val resolver = getApplication<Application>().contentResolver
                var saved = 0
                for (uri in uris) {
                    val file = newCaptureFile()
                    resolver.openInputStream(uri)?.use { input ->
                        FileOutputStream(file).use { output -> input.copyTo(output) }
                    } ?: error("无法读取图片")
                    repository.importOriginal(file, documentId, createdAtForNewPage(documentId))
                    file.delete()
                    saved += 1
                }
                if (saved == 0) error("没有可导入的图片")
            }.onSuccess {
                val pendingShots = _state.value.burstShots
                pendingShots.forEach { shot ->
                    if (shot.pageId == null) File(shot.path).delete()
                }
                val all = repository.loadPages()
                _state.update { state ->
                    state.withDocuments(all, documentId).copy(
                        screen = ScannerScreen.PAGES,
                        draft = null,
                        burstShots = emptyList(),
                        retakePageId = null,
                        returnToPages = false,
                        isApplying = false,
                    )
                }
            }.onFailure { error ->
                val all = repository.loadPages()
                _state.update { state ->
                    state.withDocuments(all, state.activeDocumentId).copy(
                        isApplying = false,
                        errorMessage = error.message ?: "导入图片失败",
                    )
                }
            }
        }
    }

    fun setFlashSetting(setting: FlashSetting) = _state.update { it.copy(flashSetting = setting) }

    fun setCaptureSize(size: CaptureSize) = _state.update { it.copy(captureSize = size) }

    fun setAutoCapture(enabled: Boolean) {
        cameraSettings.edit().putBoolean(KEY_AUTO_CAPTURE, enabled).apply()
        if (enabled) rearmAutoCapture() else stableAccurateFrames = 0
        _state.update { it.copy(autoCapture = enabled, pendingAutoCapture = false) }
    }

    fun setDocumentCorrection(enabled: Boolean) {
        cameraSettings.edit().putBoolean(KEY_DOCUMENT_CORRECTION, enabled).apply()
        _state.update { it.copy(documentCorrection = enabled) }
    }

    fun setAutoCrop(enabled: Boolean) {
        cameraSettings.edit().putBoolean(KEY_AUTO_CROP, enabled).apply()
        if (!enabled) {
            stableAccurateFrames = 0
            autoCaptureArmed = true
        }
        _state.update {
            it.copy(
                autoCrop = enabled,
                detection = if (enabled) it.detection else null,
                pendingAutoCapture = if (enabled) it.pendingAutoCapture else false,
            )
        }
    }

    fun setCaptureMode(mode: CaptureMode) {
        if (_state.value.retakePageId != null) return
        _state.update { it.copy(captureMode = mode) }
    }

    fun retakeCurrentPage() {
        if (_state.value.isApplying) return
        val draft = _state.value.draft ?: return
        val burstShot = _state.value.burstShots.firstOrNull {
            it.pageId == draft.id || it.path == draft.originalPath
        }
        val savedPage = _state.value.pages.any { it.id == draft.id }
        if (!savedPage && burstShot == null) {
            discardDraft()
            return
        }
        File(draft.previewPath).delete()
        draft.effectPreviewPath?.let { File(it).delete() }
        draft.cropPreviewPath?.let { File(it).delete() }
        if (captureModeBeforeRetake == null) captureModeBeforeRetake = _state.value.captureMode
        _state.update {
            it.copy(
                screen = ScannerScreen.CAMERA,
                draft = null,
                detection = null,
                captureMode = if (burstShot != null) it.captureMode else CaptureMode.SINGLE,
                retakePageId = if (savedPage) draft.id else null,
                retakeBurstPath = burstShot?.path,
            )
        }
    }

    fun replaceRetakenBurstShot(saved: File) {
        val oldPath = _state.value.retakeBurstPath ?: return
        _state.update { state ->
            state.copy(
                burstShots = state.burstShots.map { shot ->
                    if (shot.path == oldPath) shot.copy(path = saved.absolutePath) else shot
                },
                retakeBurstPath = null,
            )
        }
        if (oldPath != saved.absolutePath) {
            val stillUsed = _state.value.pages.any { it.originalPath == oldPath || it.correctedPath == oldPath } ||
                _state.value.burstShots.any { it.path == oldPath }
            if (!stillUsed) File(oldPath).delete()
        }
        processCapture(saved)
    }

    fun cancelRetake() {
        val mode = captureModeBeforeRetake ?: _state.value.captureMode
        val returnToBurst = _state.value.retakeBurstPath != null || mode == CaptureMode.BURST
        captureModeBeforeRetake = null
        _state.update {
            it.copy(
                screen = if (returnToBurst) ScannerScreen.CAMERA else ScannerScreen.PAGES,
                draft = null,
                detection = null,
                captureMode = mode,
                retakePageId = null,
                retakeBurstPath = null,
                returnToPages = false,
            )
        }
    }

    fun finishBurst() {
        if (_state.value.isApplying || _state.value.captureMode != CaptureMode.BURST) return
        val shots = _state.value.burstShots.filterNot { it.path.startsWith("pending-") }
        if (shots.isEmpty()) return
        val createdAt = _state.value.documentCreatedAtMillis ?: System.currentTimeMillis()
        val started = SystemClock.elapsedRealtime()
        _state.update { it.copy(isApplying = true, errorMessage = null, documentCreatedAtMillis = createdAt) }
        viewModelScope.launch(Dispatchers.IO) {
            val remaining = shots.toMutableList()
            val documentId = _state.value.activeDocumentId ?: UUID.randomUUID().toString()
            runCatching {
                val saved = repository.loadPages().associateBy { it.id }
                val pending = shots.filter { it.pageId?.let(saved::containsKey) != true }
                val firstDate = if (saved.values.none { it.documentId == documentId }) createdAt
                    else System.currentTimeMillis()
                repository.importOriginalBatch(pending.mapIndexed { index, shot ->
                    File(shot.path) to if (index == 0) firstDate else System.currentTimeMillis()
                }, documentId)
                shots.forEach { shot ->
                    if (shot.path != shot.pageId?.let { saved[it]?.originalPath }) File(shot.path).delete()
                    remaining.removeAll { it.path == shot.path }
                }
                Log.d("ScannerPerf", "burst-to-pages=${SystemClock.elapsedRealtime() - started}ms count=${shots.size}")
            }.onSuccess {
                val all = repository.loadPages()
                _state.update {
                    it.withDocuments(all, documentId).copy(
                        screen = ScannerScreen.PAGES,
                        burstShots = emptyList(),
                        draft = null,
                        isApplying = false,
                        errorMessage = null,
                    )
                }
            }.onFailure { error ->
                val all = repository.loadPages()
                _state.update {
                    it.withDocuments(all, it.activeDocumentId).copy(
                        burstShots = remaining,
                        isApplying = false,
                        errorMessage = error.message ?: "连拍保存失败",
                    )
                }
            }
        }
    }

    fun addPendingBurstShot(preview: android.graphics.Bitmap?) {
        val path = "pending-${UUID.randomUUID()}"
        pendingBurstPath = path
        _state.update {
            it.copy(
                burstShots = it.burstShots + BurstItem(path, preview = preview),
                documentCreatedAtMillis = it.documentCreatedAtMillis ?: System.currentTimeMillis(),
            )
        }
    }

    /** Test/import compatibility entry: publish an already-saved burst JPEG immediately. */
    fun addBurstShot(file: File) {
        if (!file.isFile || _state.value.captureMode != CaptureMode.BURST) return
        pendingBurstPath = null
        _state.update { state ->
            state.copy(
                burstShots = state.burstShots + BurstItem(file.absolutePath),
                documentCreatedAtMillis = state.documentCreatedAtMillis ?: System.currentTimeMillis(),
            )
        }
    }

    fun dropPendingBurstShot() {
        val pending = pendingBurstPath ?: return
        pendingBurstPath = null
        _state.update { it.copy(burstShots = it.burstShots.filterNot { shot -> shot.path == pending }) }
    }

    private fun commitBurstShot(file: File) {
        val pending = pendingBurstPath
        pendingBurstPath = null
        if (pending != null && _state.value.burstShots.none { it.path == pending }) {
            file.delete()
            _state.update { it.copy(isApplying = false, screen = ScannerScreen.CAMERA) }
            return
        }
        _state.update { state ->
            val shots = if (pending == null) {
                state.burstShots + BurstItem(file.absolutePath)
            } else {
                state.burstShots.map { shot ->
                    if (shot.path == pending) BurstItem(file.absolutePath, preview = shot.preview) else shot
                }
            }
            state.copy(burstShots = shots, isApplying = false, screen = ScannerScreen.CAMERA)
        }
    }

    fun removeBurstShot(path: String) {
        val shot = _state.value.burstShots.firstOrNull { it.path == path } ?: return
        if (path == pendingBurstPath) pendingBurstPath = null
        if (shot.pageId == null && !path.startsWith("pending-")) File(path).delete()
        _state.update { it.copy(burstShots = it.burstShots.filterNot { item -> item.path == path }) }
    }

    fun discardDraft() {
        if (_state.value.isApplying) return
        val draft = _state.value.draft
        val retakeId = _state.value.retakePageId
        draft?.let {
            val savedOriginal = _state.value.pages.firstOrNull { page -> page.id == it.id }?.originalPath
            val keepOriginal = savedOriginal == it.originalPath ||
                _state.value.burstShots.any { shot -> shot.path == it.originalPath }
            if (!keepOriginal) File(it.originalPath).delete()
            File(it.previewPath).delete()
            it.effectPreviewPath?.let { path -> File(path).delete() }
            it.cropPreviewPath?.let { path -> File(path).delete() }
        }
        if (retakeId != null && draft?.id == retakeId) {
            cancelRetake()
        } else if (_state.value.returnToPages) {
            _state.update { it.copy(screen = ScannerScreen.PAGES, draft = null, detection = null, returnToPages = false) }
        } else {
            showCamera()
        }
    }

    fun exportPdf() {
        if (_state.value.isApplying) return
        val pages = _state.value.pages
        if (pages.isEmpty()) return
        _state.update { it.copy(isApplying = true, errorMessage = null) }
        viewModelScope.launch(Dispatchers.IO) {
            runCatching { processingMutex.withLock { exporter.export(pages) } }
                .onSuccess { file ->
                    _state.update { it.copy(sharePdfPath = file.absolutePath, isApplying = false) }
                }
                .onFailure { error ->
                    _state.update {
                        it.copy(isApplying = false, errorMessage = error.message ?: "PDF 导出失败")
                    }
                }
        }
    }

    fun clearShareRequest() = _state.update { it.copy(sharePdfPath = null) }

    fun clearError() = _state.update { it.copy(errorMessage = null) }

    private fun allowProcessing(draft: DraftPage): Boolean {
        if (draft.options.cropEnabled && draft.boundaryDetectionPending) {
            onCameraError("正在检测边框，请稍候，或手动调整边框")
            return false
        }
        if (draft.options.cropEnabled && draft.requiresBoundaryConfirmation) {
            onCameraError("边框需要确认，请先调整边框或点击“确认当前边框”")
            return false
        }
        return true
    }

    override fun onCleared() {
        synchronized(modelLock) {
            cleared = true
            cornerModel?.close()
            dewarpModel?.close()
        }
    }

    /** Deduplicate still-image inference; do not queue it ahead of the next burst capture. */
    private fun boundaryResult(path: String): Deferred<Result<DraftPage>> = synchronized(boundaryDetections) {
        boundaryDetections[path]?.let { return@synchronized it }
        // Keep only a few completed results. No image pixels are retained here.
        if (boundaryDetections.size >= 8) {
            boundaryDetections.entries.removeAll { it.value.isCompleted }
        }
        viewModelScope.async(Dispatchers.Default, start = CoroutineStart.LAZY) {
            runCatching {
                modelInitialization.join()
                boundaryMutex.withLock { processor.createDraft(File(path)) }
            }
        }.also { boundaryDetections[path] = it; it.start() }
    }

    private fun detectEditorBoundary(draft: DraftPage) {
        if (draft.boundaryDetectionPending) return
        _state.update { current ->
            if (current.draft?.id != draft.id) current else
                current.copy(draft = current.draft.copy(boundaryDetectionPending = true))
        }
        viewModelScope.launch(Dispatchers.Default) {
            val result = boundaryResult(draft.originalPath).await()
            _state.update { current ->
                val active = current.draft ?: return@update current
                // A drag, a different photo, or Keep Full Photo supersedes detection.
                if (active.id != draft.id || active.originalPath != draft.originalPath ||
                    !active.options.cropEnabled) return@update current
                result.fold(onSuccess = { detected ->
                    var geometry = detected.geometry
                    val leftTurns = ((360 - active.geometry.sourceRotationDegrees) % 360) / 90
                    repeat(leftTurns) { geometry = geometry.rotatedLeft() }
                    current.copy(draft = active.copy(
                        geometry = if (active.boundaryEditRevision == draft.boundaryEditRevision) {
                            geometry.copy(manuallyAdjusted = active.geometry.manuallyAdjusted)
                        } else active.geometry,
                        detectionConfidence = detected.detectionConfidence,
                        processingMillis = detected.processingMillis,
                        modelVersion = detected.modelVersion,
                        detectionIssues = detected.detectionIssues,
                        boundaryConfirmed = false,
                        boundaryDetectionPending = false,
                        boundaryDetectionComplete = true,
                        boundaryEditedByUser = active.boundaryEditedByUser ||
                            active.boundaryEditRevision != draft.boundaryEditRevision,
                        hasProcessedPreview = false,
                        cropPreviewPath = null,
                    ))
                }, onFailure = { error ->
                    current.copy(draft = active.copy(boundaryDetectionPending = false),
                        errorMessage = error.message ?: "边框检测失败，请手动调整或保留原图")
                })
            }
        }
    }

    private fun createdAtForNewPage(documentId: String): Long {
        val stamp = _state.value.documentCreatedAtMillis
        val hasPages = repository.loadPages().any { it.documentId == documentId }
        return if (!hasPages && stamp != null) stamp else System.currentTimeMillis()
    }

    private fun ScannerUiState.withDocuments(all: List<ScanPage>, activeId: String?) = copy(
        pages = if (activeId == null) emptyList() else all.filter { it.documentId == activeId },
        documents = all.toScanDocuments(),
        activeDocumentId = activeId,
    )

    private fun List<ScanPage>.toScanDocuments(): List<ScanDocument> =
        groupBy { it.documentId }
            .map { (id, pages) ->
                ScanDocument(
                    id = id,
                    createdAtMillis = pages.minOf { it.createdAtMillis },
                    pageCount = pages.size,
                    coverPath = pages.first().cachedThumbnailFile().takeIf { it.isFile }?.absolutePath
                        ?: pages.first().correctedPath,
                )
            }
            .sortedByDescending { it.createdAtMillis }

    private companion object {
        const val KEY_AUTO_CAPTURE = "auto_capture"
        const val KEY_AUTO_CROP = "auto_crop"
        const val KEY_DOCUMENT_CORRECTION = "document_correction"
        const val AUTO_CAPTURE_STABLE_FRAMES = 3
    }
}
