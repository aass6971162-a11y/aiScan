@file:OptIn(androidx.compose.material3.ExperimentalMaterial3Api::class)

package com.huaan.sales.myapplication6.ui

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import androidx.activity.compose.BackHandler
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.view.PreviewView
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.clickable
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.awaitEachGesture
import androidx.compose.foundation.gestures.awaitFirstDown
import kotlinx.coroutines.withTimeoutOrNull
import androidx.compose.foundation.gestures.detectDragGestures
import androidx.compose.foundation.gestures.drag
import androidx.compose.foundation.gestures.waitForUpOrCancellation
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.pager.HorizontalPager
import androidx.compose.foundation.pager.rememberPagerState
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.offset
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.grid.GridCells
import androidx.compose.foundation.lazy.grid.LazyVerticalGrid
import androidx.compose.foundation.lazy.grid.itemsIndexed
import androidx.compose.foundation.border
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.CameraAlt
import androidx.compose.material.icons.filled.Edit
import androidx.compose.material.icons.filled.Check
import androidx.compose.material.icons.filled.RotateLeft
import androidx.compose.material.icons.filled.TextFields
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.Crop
import androidx.compose.material.icons.filled.PhotoLibrary
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.Description
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.Share
import androidx.compose.material.icons.filled.FlashAuto
import androidx.compose.material.icons.filled.FlashOff
import androidx.compose.material.icons.filled.FlashOn
import androidx.compose.material.icons.filled.PhotoSizeSelectLarge
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.FilledIconButton
import androidx.compose.material3.FabPosition
import androidx.compose.material3.FloatingActionButton
import androidx.compose.material3.FilterChip
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Slider
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.material3.Surface
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.State
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.produceState
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.runtime.snapshotFlow
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Rect
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.LayoutCoordinates
import androidx.compose.ui.layout.onGloballyPositioned
import androidx.compose.ui.zIndex
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.drawscope.withTransform
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.platform.LocalDensity
import androidx.lifecycle.compose.LocalLifecycleOwner
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.IntOffset
import androidx.compose.ui.unit.IntSize
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import androidx.lifecycle.viewmodel.compose.viewModel
import com.huaan.sales.myapplication6.CaptureMode
import com.huaan.sales.myapplication6.R
import com.huaan.sales.myapplication6.ScannerScreen
import com.huaan.sales.myapplication6.ScannerUiState
import com.huaan.sales.myapplication6.ScannerViewModel
import com.huaan.sales.myapplication6.camera.CameraSession
import com.huaan.sales.myapplication6.camera.CaptureSize
import com.huaan.sales.myapplication6.camera.FlashSetting
import com.huaan.sales.myapplication6.domain.ColorMode
import com.huaan.sales.myapplication6.domain.PageEffect
import com.huaan.sales.myapplication6.domain.DocumentDetection
import com.huaan.sales.myapplication6.domain.DocumentGeometry
import com.huaan.sales.myapplication6.domain.documentTitle
import com.huaan.sales.myapplication6.domain.DraftPage
import com.huaan.sales.myapplication6.domain.NormalizedPoint
import com.huaan.sales.myapplication6.domain.QualityIssue
import com.huaan.sales.myapplication6.data.cachedThumbnailFile
import com.huaan.sales.myapplication6.BurstItem
import com.huaan.sales.myapplication6.domain.ScanPage
import com.huaan.sales.myapplication6.vision.decodeCachedThumbnail
import com.huaan.sales.myapplication6.vision.decodeOrientedBitmap
import com.huaan.sales.myapplication6.vision.DocumentEffects
import com.huaan.sales.myapplication6.vision.writeCachedThumbnail
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import kotlin.math.hypot
import kotlin.math.roundToInt

private val ScannerGreen = Color(0xFF2DD881)
private val ScannerAmber = Color(0xFFFFB547)
private val CameraBlack = Color(0xFF080A09)

@Composable
fun ScannerApp(
    viewModel: ScannerViewModel = viewModel(),
    onExit: () -> Unit = {},
) {
    val state by viewModel.state.collectAsState()
    val context = LocalContext.current
    val snackbar = remember { SnackbarHostState() }
    BackHandler {
        when (state.screen) {
            ScannerScreen.HOME -> onExit()
            ScannerScreen.CAMERA -> if (state.retakePageId != null || state.retakeBurstPath != null) {
                viewModel.cancelRetake()
            } else {
                onExit()
            }
            ScannerScreen.PAGES -> onExit()
            ScannerScreen.EDITOR -> viewModel.discardDraft()
            ScannerScreen.PROCESSING -> Unit
        }
    }

    LaunchedEffect(state.errorMessage) {
        state.errorMessage?.let {
            snackbar.showSnackbar(it)
            viewModel.clearError()
        }
    }
    LaunchedEffect(state.sharePdfPath) {
        state.sharePdfPath?.let { path ->
            val file = File(path)
            val uri = FileProvider.getUriForFile(context, "${context.packageName}.files", file)
            val intent = Intent(Intent.ACTION_SEND).apply {
                type = "application/pdf"
                putExtra(Intent.EXTRA_STREAM, uri)
                addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
            }
            context.startActivity(Intent.createChooser(intent, "分享扫描 PDF"))
            viewModel.clearShareRequest()
        }
    }

    Box(Modifier.fillMaxSize()) {
        when (state.screen) {
            ScannerScreen.HOME -> Unit
            ScannerScreen.CAMERA -> CameraScreen(
                state,
                viewModel,
                onBack = if (state.retakePageId != null || state.retakeBurstPath != null) viewModel::cancelRetake else onExit,
            )
            ScannerScreen.PROCESSING -> ProcessingScreen(state.processingMessage)
            ScannerScreen.EDITOR -> state.draft?.let {
                EditorScreen(
                    it,
                    state.isApplying,
                    state.editorCropActive,
                    state.documentCreatedAtMillis,
                    state.burstShots,
                    state.pages,
                    viewModel,
                )
            }
            ScannerScreen.PAGES -> PagesScreen(state, viewModel, onBack = onExit)
        }
        SnackbarHost(
            hostState = snackbar,
            modifier = Modifier.align(Alignment.BottomCenter).padding(16.dp),
        )
    }
}

@Composable
private fun CameraScreen(state: ScannerUiState, viewModel: ScannerViewModel, onBack: () -> Unit) {
    val context = LocalContext.current
    val importImages = rememberLauncherForActivityResult(ActivityResultContracts.PickMultipleVisualMedia()) { uris ->
        if (uris.isNotEmpty()) viewModel.importImages(uris)
    }
    var hasPermission by remember {
        mutableStateOf(ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED)
    }
    val permissionLauncher = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) {
        hasPermission = it
    }
    LaunchedEffect(Unit) {
        if (!hasPermission) permissionLauncher.launch(Manifest.permission.CAMERA)
    }

    if (!hasPermission) {
        PermissionScreen { permissionLauncher.launch(Manifest.permission.CAMERA) }
        return
    }

    val lifecycleOwner = LocalLifecycleOwner.current
    val previewView = remember { PreviewView(context) }
    val currentState by rememberUpdatedState(state)
    var session by remember { mutableStateOf<CameraSession?>(null) }
    var frozenPreview by remember { mutableStateOf<android.graphics.Bitmap?>(null) }
    DisposableEffect(lifecycleOwner, previewView) {
        var disposed = false
        previewView.post {
            if (disposed) return@post
            session = CameraSession(
                context = context,
                lifecycleOwner = lifecycleOwner,
                previewView = previewView,
                detector = viewModel.previewDetector,
                onDetection = viewModel::onDetection,
                onError = viewModel::onCameraError,
            ).also {
                it.setBorderScanEnabled(currentState.autoCrop)
                it.start(currentState.flashSetting, currentState.captureSize)
            }
        }
        onDispose {
            disposed = true
            session?.close()
            session = null
        }
    }

    LaunchedEffect(session, state.autoCrop) {
        session?.setBorderScanEnabled(state.autoCrop)
    }

    Box(Modifier.fillMaxSize().background(CameraBlack)) {
        AndroidView(factory = { previewView }, modifier = Modifier.fillMaxSize())
        frozenPreview?.let { snapshot ->
            Image(
                snapshot.asImageBitmap(),
                null,
                Modifier.fillMaxSize().background(Color.Black),
                contentScale = ContentScale.Fit,
            )
        }
        if (state.autoCrop) DetectionOverlay(state.detection, Modifier.fillMaxSize())
        CameraHeader(
            state = state,
            onBack = onBack,
            onFlash = { setting ->
                viewModel.setFlashSetting(setting)
                session?.updateFlash(setting)
            },
            onResolution = { size ->
                viewModel.setCaptureSize(size)
                session?.updateResolution(size)
            },
            onAutoCapture = viewModel::setAutoCapture,
            onAutoCrop = viewModel::setAutoCrop,
            onDocumentCorrection = viewModel::setDocumentCorrection,
            modifier = Modifier.align(Alignment.TopCenter),
        )
        val qualityIssue = state.detection?.qualityIssues?.let { issues ->
            listOf(QualityIssue.EDGE_CLIPPED, QualityIssue.TOO_SMALL, QualityIssue.NO_DOCUMENT,
                QualityIssue.TOO_BLURRY, QualityIssue.GLARE, QualityIssue.TOO_DARK,
                QualityIssue.TOO_BRIGHT, QualityIssue.LOW_CONFIDENCE).firstOrNull { it in issues }
        }
        val thumbnails = state.burstShots.map { shot ->
            val page = shot.pageId?.let { id -> state.pages.firstOrNull { it.id == id } }
            CaptureThumbnail(
                key = shot.path,
                path = page?.correctedPath ?: shot.path,
                preview = shot.preview,
                onClick = {
                    if (shot.path.startsWith("pending-")) return@CaptureThumbnail
                    if (page != null) viewModel.editPage(page) else viewModel.processCapture(File(shot.path))
                },
                onDelete = {
                    if (page != null) viewModel.deletePage(page.id)
                    viewModel.removeBurstShot(shot.path)
                },
            )
        }
        var capturing by remember { mutableStateOf(false) }
        fun takePhoto() {
            if (capturing || state.isApplying) return
            val file = viewModel.newCaptureFile()
            val mode = state.captureMode
            val replacingBurst = state.retakeBurstPath != null
            capturing = true
            viewModel.disarmAutoCapture()
            if (mode == CaptureMode.BURST && !replacingBurst) {
                viewModel.addPendingBurstShot(previewView.bitmap)
            } else if (mode == CaptureMode.SINGLE && !replacingBurst) {
                frozenPreview = previewView.bitmap
            }
            session?.capture(
                output = file,
                onSaved = { saved ->
                    frozenPreview = null
                    capturing = false
                    viewModel.acceptCapturedPhoto(saved, burst = mode == CaptureMode.BURST, replacingBurst = replacingBurst)
                },
                onFailed = {
                    frozenPreview = null
                    capturing = false
                    viewModel.dropPendingBurstShot()
                    viewModel.rearmAutoCapture()
                },
            ) ?: run {
                frozenPreview = null
                capturing = false
                viewModel.dropPendingBurstShot()
                viewModel.rearmAutoCapture()
            }
        }
        LaunchedEffect(state.pendingAutoCapture) {
            if (!state.pendingAutoCapture) return@LaunchedEffect
            viewModel.consumeAutoCapture()
            takePhoto()
        }
        Column(
            modifier = Modifier.align(Alignment.BottomCenter).fillMaxWidth().navigationBarsPadding(),
        ) {
            qualityIssue?.let { issue ->
                QualityBanner(
                    issue,
                    Modifier.align(Alignment.CenterHorizontally).padding(bottom = 10.dp),
                )
            }
            if (state.captureMode == CaptureMode.BURST && thumbnails.isNotEmpty()) {
                CaptureThumbnailStrip(thumbnails)
            }
            Row(
                modifier = Modifier.fillMaxWidth().padding(bottom = 4.dp),
                horizontalArrangement = Arrangement.Center,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                CaptureModeButton(
                    "单拍",
                    state.retakeBurstPath == null && (state.captureMode == CaptureMode.SINGLE || state.retakePageId != null),
                ) {
                    if (state.retakePageId == null && state.retakeBurstPath == null) viewModel.setCaptureMode(CaptureMode.SINGLE)
                }
                if (state.retakePageId == null && state.retakeBurstPath == null) {
                    Spacer(Modifier.width(28.dp))
                    CaptureModeButton("连拍", state.captureMode == CaptureMode.BURST) {
                        viewModel.setCaptureMode(CaptureMode.BURST)
                    }
                }
            }
            Row(
                modifier = Modifier.fillMaxWidth().padding(horizontal = 28.dp, vertical = 12.dp),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Box(Modifier.width(88.dp), contentAlignment = Alignment.CenterStart) {
                    IconButton(
                        onClick = {
                            importImages.launch(
                                PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly),
                            )
                        },
                        enabled = !state.isApplying,
                        modifier = Modifier.size(52.dp).background(Color.Black.copy(alpha = 0.58f), CircleShape),
                    ) {
                        Icon(Icons.Default.PhotoLibrary, "导入图片", tint = Color.White)
                    }
                }
                FilledIconButton(
                    onClick = { takePhoto() },
                    enabled = session != null && !capturing && !state.isApplying,
                    modifier = Modifier.size(76.dp),
                ) {
                    Icon(Icons.Default.Description, if (state.captureMode == CaptureMode.BURST) "连拍" else "单拍", modifier = Modifier.size(34.dp))
                }
                Box(Modifier.width(88.dp), contentAlignment = Alignment.CenterEnd) {
                    if (state.captureMode == CaptureMode.BURST && state.burstShots.isNotEmpty()) {
                        if (state.isApplying) {
                            CircularProgressIndicator(color = ScannerGreen, modifier = Modifier.size(28.dp))
                        } else {
                            TextButton(onClick = viewModel::finishBurst, enabled = !capturing) {
                                Text("已完成", color = ScannerGreen, fontSize = 16.sp, fontWeight = FontWeight.Medium)
                            }
                        }
                    }
                }
            }
        }
    }
}

private data class CaptureThumbnail(
    val key: String,
    val path: String,
    val preview: android.graphics.Bitmap? = null,
    val onClick: () -> Unit,
    val onDelete: () -> Unit,
)

private class ThumbnailDragState {
    var draggingKey by mutableStateOf<String?>(null)
    var dragOffset by mutableStateOf(Offset.Zero)
    var itemPosition by mutableStateOf(Offset.Zero)
    var deleteVisible by mutableStateOf(false)
    var deleteBounds by mutableStateOf<Rect?>(null)
    var stripCoordinates by mutableStateOf<LayoutCoordinates?>(null)
    private var pendingDelete: (() -> Unit)? = null

    fun start(key: String, onDelete: () -> Unit, itemCoordinates: LayoutCoordinates) {
        val strip = stripCoordinates ?: return
        draggingKey = key
        pendingDelete = onDelete
        itemPosition = strip.localPositionOf(itemCoordinates, Offset.Zero)
        dragOffset = Offset.Zero
        deleteVisible = false
        deleteBounds = null
    }

    fun move(delta: Offset, revealThreshold: Float) {
        dragOffset += delta
        if (dragOffset.y < -revealThreshold) deleteVisible = true
    }

    fun end(thumbSize: Size) {
        val thumbRect = Rect(itemPosition + dragOffset, thumbSize)
        val delete = pendingDelete
        val hit = deleteVisible && deleteBounds?.overlaps(thumbRect) == true
        reset()
        if (hit) delete?.invoke()
    }

    fun cancel() = reset()

    private fun reset() {
        draggingKey = null
        dragOffset = Offset.Zero
        deleteVisible = false
        deleteBounds = null
        pendingDelete = null
    }
}

@Composable
private fun CaptureThumbnailStrip(thumbnails: List<CaptureThumbnail>) {
    val listState = rememberLazyListState()
    val dragState = remember { ThumbnailDragState() }
    val dragging = thumbnails.firstOrNull { it.key == dragState.draggingKey }
    val density = LocalDensity.current
    val thumbSize = with(density) { Size(52.dp.toPx(), 68.dp.toPx()) }
    val draggedRect = Rect(dragState.itemPosition + dragState.dragOffset, thumbSize)
    val armed = dragState.deleteVisible && dragState.deleteBounds?.overlaps(draggedRect) == true
    LaunchedEffect(thumbnails.size) {
        if (dragState.draggingKey == null && thumbnails.isNotEmpty()) {
            listState.animateScrollToItem(thumbnails.lastIndex)
        }
    }
    Box(
        Modifier
            .fillMaxWidth()
            .onGloballyPositioned { dragState.stripCoordinates = it },
    ) {
        LazyRow(
                state = listState,
                userScrollEnabled = dragState.draggingKey == null,
                modifier = Modifier.fillMaxWidth().padding(bottom = 8.dp),
                contentPadding = PaddingValues(horizontal = 16.dp),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                items(thumbnails, key = { it.key }) { thumbnail ->
                    CaptureThumbnailItem(
                        thumbnail = thumbnail,
                        hidden = dragState.draggingKey == thumbnail.key,
                        dragState = dragState,
                    )
                }
            }
            if (dragState.deleteVisible) {
                DeleteThumbnailTarget(
                    armed = armed,
                    modifier = Modifier
                        .align(Alignment.TopCenter)
                        .offset(y = (-58).dp)
                        .zIndex(1f)
                        .onGloballyPositioned { coordinates ->
                            val strip = dragState.stripCoordinates ?: return@onGloballyPositioned
                            val topLeft = strip.localPositionOf(coordinates, Offset.Zero)
                            dragState.deleteBounds = Rect(
                                topLeft,
                                Size(coordinates.size.width.toFloat(), coordinates.size.height.toFloat()),
                            )
                        },
                )
            }
            if (dragging != null) {
                DraggedThumbnail(
                    path = dragging.path,
                    preview = dragging.preview,
                    modifier = Modifier
                        .zIndex(2f)
                        .offset {
                            IntOffset(
                                (dragState.itemPosition.x + dragState.dragOffset.x).roundToInt(),
                                (dragState.itemPosition.y + dragState.dragOffset.y).roundToInt(),
                            )
                        },
                )
            }
        }
}

@Composable
private fun DeleteThumbnailTarget(armed: Boolean, modifier: Modifier = Modifier) {
    Box(Modifier.fillMaxWidth().padding(bottom = 10.dp), contentAlignment = Alignment.Center) {
        Row(
            modifier = modifier
                .shadow(6.dp, RoundedCornerShape(22.dp))
                .background(if (armed) Color(0xFFD32F2F) else Color(0xF21A1A1A), RoundedCornerShape(22.dp))
                .padding(horizontal = 18.dp, vertical = 10.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Icon(Icons.Default.Delete, null, tint = Color.White)
            Spacer(Modifier.width(6.dp))
            Text("删除图片", color = Color.White, fontWeight = FontWeight.Medium)
        }
    }
}

@Composable
private fun DraggedThumbnail(path: String, preview: android.graphics.Bitmap?, modifier: Modifier = Modifier) {
    val decoded by rememberDisplayBitmap(path, maxEdge = 320)
    val bitmap = preview ?: decoded
    Box(
        modifier
            .size(width = 52.dp, height = 68.dp)
            .shadow(8.dp, RoundedCornerShape(6.dp))
            .clip(RoundedCornerShape(6.dp))
            .background(Color.White),
    ) {
        bitmap?.let {
            Image(it.asImageBitmap(), null, Modifier.fillMaxSize(), contentScale = ContentScale.Crop)
        }
    }
}

@Composable
private fun CaptureThumbnailItem(
    thumbnail: CaptureThumbnail,
    hidden: Boolean,
    dragState: ThumbnailDragState,
) {
    val decoded by rememberDisplayBitmap(thumbnail.path, maxEdge = 320)
    val bitmap = thumbnail.preview ?: decoded
    var itemCoordinates by remember { mutableStateOf<LayoutCoordinates?>(null) }
    Box(
        Modifier
            .size(width = 52.dp, height = 68.dp)
            .alpha(if (hidden) 0f else 1f)
            .onGloballyPositioned { itemCoordinates = it }
            .clip(RoundedCornerShape(6.dp))
            .background(Color.White)
            .pointerInput(thumbnail.key) {
                val revealThreshold = 16.dp.toPx()
                val thumbSize = Size(52.dp.toPx(), 68.dp.toPx())
                awaitEachGesture {
                    val down = awaitFirstDown()
                    val release = withTimeoutOrNull(viewConfiguration.longPressTimeoutMillis) {
                        waitForUpOrCancellation()
                    }
                    if (release != null) {
                        if ((release.position - down.position).getDistance() < viewConfiguration.touchSlop) {
                            thumbnail.onClick()
                        }
                        return@awaitEachGesture
                    }
                    val coordinates = itemCoordinates ?: return@awaitEachGesture
                    dragState.start(thumbnail.key, thumbnail.onDelete, coordinates)
                    val finished = drag(down.id) { change ->
                        dragState.move(change.position - change.previousPosition, revealThreshold)
                        change.consume()
                    }
                    if (finished) dragState.end(thumbSize) else dragState.cancel()
                }
            },
    ) {
        bitmap?.let {
            Image(it.asImageBitmap(), null, Modifier.fillMaxSize(), contentScale = ContentScale.Crop)
        }
    }
}

@Composable
private fun CaptureModeButton(label: String, selected: Boolean, onClick: () -> Unit) {
    TextButton(onClick = onClick, contentPadding = PaddingValues(horizontal = 12.dp, vertical = 4.dp)) {
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            Text(
                label,
                color = if (selected) ScannerGreen else Color.White.copy(alpha = 0.72f),
                fontWeight = if (selected) FontWeight.SemiBold else FontWeight.Normal,
                fontSize = 15.sp,
            )
            Spacer(Modifier.height(4.dp))
            Box(
                Modifier
                    .size(width = 18.dp, height = 3.dp)
                    .background(if (selected) ScannerGreen else Color.Transparent, RoundedCornerShape(2.dp)),
            )
        }
    }
}

/** Display bitmaps are bounded independently of the full-resolution files used for output. */
@Composable
internal fun rememberDisplayBitmap(
    path: String,
    maxEdge: Int,
    version: Long = File(path).lastModified(),
    rotationDegrees: Int = 0,
): State<android.graphics.Bitmap?> {
    val source = remember(path, maxEdge, version) { mutableStateOf<android.graphics.Bitmap?>(null) }
    LaunchedEffect(source) {
        source.value = withContext(Dispatchers.IO) {
            runCatching { decodeOrientedBitmap(File(path), maxEdge) }.getOrNull()
        }
    }
    val bitmap = source.value
    val displayed = remember(bitmap, rotationDegrees) { mutableStateOf<android.graphics.Bitmap?>(null) }
    LaunchedEffect(displayed) {
        if (bitmap == null) return@LaunchedEffect
        displayed.value = withContext(Dispatchers.Default) {
            if (rotationDegrees == 0) bitmap else android.graphics.Bitmap.createBitmap(
                bitmap, 0, 0, bitmap.width, bitmap.height,
                android.graphics.Matrix().apply { postRotate(rotationDegrees.toFloat()) }, false,
            )
        }
        // The renderer may retain the preceding frame after recomposition.
        // Bounded display bitmaps are reclaimed when all references are gone.
    }
    return displayed
}

/** List cells read the stored thumbnail only. They do not rotate or filter the original. */
@Composable
internal fun rememberCachedThumbnail(page: ScanPage): State<android.graphics.Bitmap?> {
    val thumb = page.cachedThumbnailFile()
    val version = page.processingMillis + thumb.lastModified()
    val source = remember(thumb.absolutePath, version) { mutableStateOf<android.graphics.Bitmap?>(null) }
    LaunchedEffect(thumb.absolutePath, version) {
        source.value = withContext(Dispatchers.IO) {
            if (!thumb.isFile) {
                File(page.originalPath).resolveSibling("${page.id}-thumb.jpg").delete()
                runCatching { writeCachedThumbnail(File(page.correctedPath), thumb) }
            }
            decodeCachedThumbnail(thumb)
        }
    }
    return source
}

@Composable
private fun CameraHeader(
    state: ScannerUiState,
    onBack: () -> Unit,
    onFlash: (FlashSetting) -> Unit,
    onResolution: (CaptureSize) -> Unit,
    onAutoCapture: (Boolean) -> Unit,
    onAutoCrop: (Boolean) -> Unit,
    onDocumentCorrection: (Boolean) -> Unit,
    modifier: Modifier = Modifier,
) {
    var flashMenu by remember { mutableStateOf(false) }
    var resolutionMenu by remember { mutableStateOf(false) }
    var settingsOpen by remember { mutableStateOf(false) }
    Row(
        modifier = modifier
            .fillMaxWidth()
            .background(Color.Black.copy(alpha = 0.62f))
            .statusBarsPadding()
            .padding(end = 4.dp, top = 2.dp, bottom = 2.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        IconButton(onClick = onBack) {
            Icon(Icons.AutoMirrored.Filled.ArrowBack, "返回首页", tint = Color.White)
        }
        Column(Modifier.weight(1f)) {
            Text("扫描文档", color = Color.White, fontWeight = FontWeight.SemiBold, fontSize = 19.sp)
            Spacer(Modifier.height(2.dp))
            Text(
                text = state.modelStatus.message,
                color = if (state.modelStatus.hasTrainedModels) ScannerGreen else Color.White.copy(alpha = 0.72f),
                fontSize = 12.sp,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis,
            )
        }
        Box {
            IconButton(onClick = { flashMenu = true }, modifier = Modifier.size(40.dp)) {
                Icon(state.flashSetting.icon(), "闪光灯", tint = Color.White)
            }
            DropdownMenu(expanded = flashMenu, onDismissRequest = { flashMenu = false }) {
                FlashSetting.entries.forEach { setting ->
                    DropdownMenuItem(
                        text = { Text(setting.label) },
                        leadingIcon = {
                            Icon(setting.icon(), null)
                        },
                        trailingIcon = if (setting == state.flashSetting) {
                            { Icon(Icons.Default.Check, null) }
                        } else {
                            null
                        },
                        onClick = {
                            flashMenu = false
                            onFlash(setting)
                        },
                    )
                }
            }
        }
        Box {
            IconButton(onClick = { resolutionMenu = true }, modifier = Modifier.size(40.dp)) {
                Icon(Icons.Default.PhotoSizeSelectLarge, "分辨率", tint = Color.White)
            }
            DropdownMenu(expanded = resolutionMenu, onDismissRequest = { resolutionMenu = false }) {
                CaptureSize.options.forEach { size ->
                    DropdownMenuItem(
                        text = { Text(size.label) },
                        trailingIcon = if (size == state.captureSize) {
                            { Icon(Icons.Default.Check, null) }
                        } else {
                            null
                        },
                        onClick = {
                            resolutionMenu = false
                            onResolution(size)
                        },
                    )
                }
            }
        }
        IconButton(onClick = { settingsOpen = true }, modifier = Modifier.size(40.dp)) {
            Icon(Icons.Default.Settings, "设置", tint = Color.White)
        }
    }
    if (settingsOpen) {
        AlertDialog(
            onDismissRequest = { settingsOpen = false },
            title = { Text("设置") },
            text = {
                Column {
                    SettingsSwitchRow("自动拍照", state.autoCapture, onAutoCapture)
                    SettingsSwitchRow("自动裁剪", state.autoCrop, onAutoCrop)
                    Text("拍照后显示候选边框，完成编辑时才裁剪", style = MaterialTheme.typography.bodySmall)
                    SettingsSwitchRow("文档校正", state.documentCorrection, onDocumentCorrection)
                    Text("裁剪时拉正文档透视，褶皱展平需单独开启", style = MaterialTheme.typography.bodySmall)
                }
            },
            confirmButton = {
                TextButton(onClick = { settingsOpen = false }) { Text("完成") }
            },
        )
    }
}

@Composable
private fun SettingsSwitchRow(label: String, checked: Boolean, onChecked: (Boolean) -> Unit) {
    Row(
        modifier = Modifier.fillMaxWidth().height(48.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Text(label, modifier = Modifier.weight(1f))
        Switch(checked = checked, onCheckedChange = onChecked)
    }
}

private fun FlashSetting.icon() = when (this) {
    FlashSetting.ON -> Icons.Default.FlashOn
    FlashSetting.OFF -> Icons.Default.FlashOff
    FlashSetting.AUTO -> Icons.Default.FlashAuto
}

@Composable
private fun DetectionOverlay(detection: DocumentDetection?, modifier: Modifier = Modifier) {
    Canvas(modifier) {
        if (detection?.corners?.size != 4) return@Canvas
        val rect = fittedRect(size.width, size.height, detection.sourceWidth, detection.sourceHeight)
        val points = detection.corners.map { rect.toOffset(it) }
        val path = Path().apply {
            moveTo(points[0].x, points[0].y)
            for (index in 1..3) lineTo(points[index].x, points[index].y)
            close()
        }
        val color = if (detection.isUsable && detection.qualityIssues.isEmpty()) ScannerGreen else ScannerAmber
        drawPath(path, color.copy(alpha = 0.13f))
        drawPath(path, color, style = Stroke(width = 3.dp.toPx(), cap = StrokeCap.Round))
        points.forEach { drawCircle(color, 5.dp.toPx(), it) }
    }
}

@Composable
private fun QualityBanner(issue: QualityIssue, modifier: Modifier = Modifier) {
    val message = when (issue) {
        QualityIssue.NO_DOCUMENT -> "未找到边框，可拍照后精检；请尽量拍全四角"
        QualityIssue.LOW_CONFIDENCE -> "边界不够清晰"
        QualityIssue.TOO_BLURRY -> "画面模糊，请保持稳定"
        QualityIssue.TOO_DARK -> "光线不足"
        QualityIssue.TOO_BRIGHT -> "画面过亮"
        QualityIssue.GLARE -> "检测到反光"
        QualityIssue.EDGE_CLIPPED -> "边缘贴近画面，请稍微后退并拍全四角"
        QualityIssue.TOO_SMALL -> "目标较远，请靠近一些并保持清晰"
    }
    Surface(color = Color.Black.copy(alpha = 0.76f), shape = RoundedCornerShape(6.dp), modifier = modifier) {
        Text(message, color = Color.White, modifier = Modifier.padding(horizontal = 14.dp, vertical = 9.dp), fontSize = 14.sp)
    }
}

@Composable
private fun PermissionScreen(onRequest: () -> Unit) {
    Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
        Column(horizontalAlignment = Alignment.CenterHorizontally, modifier = Modifier.padding(32.dp)) {
            Icon(Icons.Default.Description, null, modifier = Modifier.size(46.dp), tint = MaterialTheme.colorScheme.primary)
            Spacer(Modifier.height(18.dp))
            Text("需要相机权限", style = MaterialTheme.typography.titleLarge)
            Spacer(Modifier.height(8.dp))
            Text("照片仅在设备上处理，不会上传。", color = MaterialTheme.colorScheme.onSurfaceVariant)
            Spacer(Modifier.height(22.dp))
            Button(onClick = onRequest) { Text("授权相机") }
        }
    }
}

@Composable
private fun ProcessingScreen(message: String) {
    Box(Modifier.fillMaxSize().background(CameraBlack), contentAlignment = Alignment.Center) {
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            CircularProgressIndicator(color = ScannerGreen)
            Spacer(Modifier.height(18.dp))
            Text(message, color = Color.White)
        }
    }
}

@Composable
private fun EditorScreen(
    draft: DraftPage,
    isApplying: Boolean,
    showCropBorder: Boolean,
    documentCreatedAtMillis: Long?,
    burstShots: List<BurstItem>,
    pages: List<ScanPage>,
    viewModel: ScannerViewModel,
) {
    val effectFile = draft.effectPreviewPath?.let(::File)?.takeIf { it.isFile }
    val savedPage = pages.firstOrNull { it.id == draft.id }
    val cropResult = when {
        showCropBorder -> null
        draft.options.cropEnabled && draft.hasProcessedPreview && File(draft.previewPath).isFile -> draft.previewPath
        draft.options.cropEnabled && savedPage != null &&
            savedPage.geometry == draft.geometry && savedPage.options == draft.options &&
            File(savedPage.correctedPath).isFile -> savedPage.correctedPath
        else -> null
    }
    val cropPreview = draft.cropPreviewPath?.takeIf { !showCropBorder && File(it).isFile }
    val imagePath = when {
        showCropBorder -> effectFile?.absolutePath ?: draft.originalPath
        cropResult != null -> cropResult
        cropPreview != null -> cropPreview
        effectFile != null -> effectFile.absolutePath
        else -> draft.originalPath
    }
    val displayRotation = if (cropResult != null || cropPreview != null) 0 else draft.geometry.sourceRotationDegrees
    val appName = stringResource(R.string.app_name)
    val title = documentCreatedAtMillis?.let { documentTitle(appName, it) } ?: appName
    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text(title, maxLines = 1, overflow = TextOverflow.Ellipsis) },
                navigationIcon = {
                    IconButton(onClick = viewModel::discardDraft, enabled = !isApplying) {
                        Icon(Icons.Default.Close, "放弃")
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = Color(0xFF121416),
                    titleContentColor = Color.White,
                    navigationIconContentColor = Color.White,
                ),
            )
        },
        containerColor = Color(0xFF101214),
    ) { padding ->
        Column(Modifier.padding(padding).fillMaxSize().navigationBarsPadding()) {
            Box(Modifier.weight(1f).fillMaxWidth().background(Color(0xFF171A19))) {
                BurstEditorPager(
                    draft = draft,
                    imagePath = imagePath,
                    displayRotation = displayRotation,
                    showCropBorder = showCropBorder,
                    isApplying = isApplying,
                    burstShots = burstShots,
                    pages = pages,
                    onShowShot = viewModel::showBurstShot,
                    onPointChanged = { index, x, y -> if (!isApplying) viewModel.updateControlPoint(index, x, y) },
                    modifier = Modifier.fillMaxSize(),
                )
                if (showCropBorder) {
                    Column(Modifier.align(Alignment.BottomCenter).padding(12.dp)
                        .background(Color.Black.copy(alpha = .72f), RoundedCornerShape(12.dp))
                        .padding(horizontal = 12.dp, vertical = 6.dp),
                        horizontalAlignment = Alignment.CenterHorizontally) {
                        if (draft.boundaryDetectionPending) {
                            Text("正在检测边框，可手动调整", color = Color.White, fontSize = 12.sp)
                        } else if (draft.requiresBoundaryConfirmation) {
                            Text(if (QualityIssue.EDGE_CLIPPED in draft.detectionIssues)
                                "边缘贴近或超出照片，边框需确认；照片外的内容无法恢复"
                                else "边框不确定，请调整或确认后再完成", color = Color.White, fontSize = 12.sp)
                        }
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            TextButton(onClick = viewModel::keepFullPhoto, enabled = !isApplying) {
                                Text("保留完整照片")
                            }
                            Button(onClick = viewModel::confirmEditorCrop,
                                enabled = !isApplying && !draft.boundaryDetectionPending) {
                                Text(if (draft.requiresBoundaryConfirmation) "确认当前边框" else "确定")
                            }
                        }
                    }
                }
                if (isApplying) {
                    Column(
                        modifier = Modifier.align(Alignment.TopEnd).padding(12.dp),
                        horizontalAlignment = Alignment.CenterHorizontally,
                    ) {
                        CircularProgressIndicator(color = ScannerGreen, modifier = Modifier.size(28.dp))
                        Text("正在处理图片", color = Color.White, style = MaterialTheme.typography.labelSmall)
                    }
                }
            }
            if (draft.options.cropEnabled) Row(Modifier.fillMaxWidth().padding(horizontal = 12.dp),
                horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                FilterChip(selected = draft.options.perspectiveEnabled,
                    enabled = !isApplying && !draft.boundaryDetectionPending,
                    onClick = { viewModel.setPerspectiveEnabled(!draft.options.perspectiveEnabled) },
                    label = { Text("透视校正") })
                FilterChip(selected = draft.options.dewarpEnabled,
                    enabled = !isApplying && !draft.boundaryDetectionPending &&
                        "document_dewarp" in viewModel.state.value.modelStatus.readyModelIds,
                    onClick = { viewModel.setDewarpEnabled(!draft.options.dewarpEnabled) },
                    label = { Text("褶皱展平") })
            }
            EditorBottomBar(
                draft = draft,
                isApplying = isApplying,
                cropActive = showCropBorder,
                onRetake = viewModel::retakeCurrentPage,
                onRotateLeft = viewModel::rotateLeft,
                onCrop = viewModel::startEditorCrop,
                onExtractText = { viewModel.onCameraError("提取文字暂未开放") },
                onFilter = viewModel::setPageEffect,
                onConfirm = viewModel::saveDraft,
            )
        }
    }
}

@Composable
private fun BurstEditorPager(
    draft: DraftPage,
    imagePath: String,
    displayRotation: Int,
    showCropBorder: Boolean,
    isApplying: Boolean,
    burstShots: List<BurstItem>,
    pages: List<ScanPage>,
    onShowShot: (Int) -> Unit,
    onPointChanged: (Int, Float, Float) -> Unit,
    modifier: Modifier = Modifier,
) {
    val currentIndex = burstShots.indexOfFirst { it.pageId == draft.id || it.path == draft.originalPath }
    if (burstShots.size < 2 || currentIndex < 0) {
        DocumentEditor(
            imagePath = imagePath,
            imageVersion = File(imagePath).lastModified(),
            rotationDegrees = displayRotation,
            geometry = draft.geometry,
            showBorder = showCropBorder,
            onPointChanged = onPointChanged,
            modifier = modifier,
        )
        return
    }
    val pagerState = rememberPagerState(initialPage = currentIndex, pageCount = { burstShots.size })
    LaunchedEffect(pagerState, burstShots, currentIndex) {
        snapshotFlow { pagerState.settledPage }.collect { page ->
            if (page != currentIndex && page in burstShots.indices) onShowShot(page)
        }
    }
    LaunchedEffect(currentIndex) {
        if (currentIndex in burstShots.indices && pagerState.currentPage != currentIndex && !pagerState.isScrollInProgress) {
            pagerState.scrollToPage(currentIndex)
        }
    }
    Box(modifier) {
        HorizontalPager(
            state = pagerState,
            modifier = Modifier.fillMaxSize(),
            userScrollEnabled = !showCropBorder && !isApplying,
            beyondViewportPageCount = 1,
        ) { page ->
            if (page == currentIndex) {
                DocumentEditor(
                    imagePath = imagePath,
                    imageVersion = File(imagePath).lastModified(),
                    rotationDegrees = displayRotation,
                    geometry = draft.geometry,
                    showBorder = showCropBorder,
                    onPointChanged = onPointChanged,
                    modifier = Modifier.fillMaxSize(),
                )
            } else {
                val shot = burstShots[page]
                val path = pages.firstOrNull { it.id == shot.pageId }?.correctedPath ?: shot.path
                val bitmap by rememberDisplayBitmap(path, maxEdge = 1600)
                val image = bitmap
                if (image == null) {
                    Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                        CircularProgressIndicator(color = ScannerGreen)
                    }
                } else {
                    Image(
                        image.asImageBitmap(),
                        null,
                        Modifier.fillMaxSize(),
                        contentScale = ContentScale.Fit,
                    )
                }
            }
        }
        if (!showCropBorder) Text(
            "${currentIndex + 1} / ${burstShots.size}",
            color = Color.White,
            fontSize = 13.sp,
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .padding(bottom = 10.dp)
                .background(Color.Black.copy(alpha = 0.45f), RoundedCornerShape(10.dp))
                .padding(horizontal = 10.dp, vertical = 4.dp),
        )
    }
}

@Composable
private fun DocumentEditor(
    imagePath: String,
    imageVersion: Long,
    rotationDegrees: Int,
    geometry: DocumentGeometry,
    showBorder: Boolean,
    onPointChanged: (Int, Float, Float) -> Unit,
    modifier: Modifier = Modifier,
) {
    val bitmap by rememberDisplayBitmap(imagePath, maxEdge = 2048, version = imageVersion,
        rotationDegrees = rotationDegrees)
    val displayed = bitmap
    val currentGeometry by rememberUpdatedState(geometry)
    var activePoint by remember { mutableIntStateOf(-1) }
    if (displayed == null) {
        Box(modifier, contentAlignment = Alignment.Center) { CircularProgressIndicator(color = ScannerGreen) }
        return
    }
    val image = remember(displayed) { displayed.asImageBitmap() }
    Box(modifier) {
        Image(image, null, Modifier.fillMaxSize().testTag("document-editor-image"), contentScale = ContentScale.Fit)
        if (!showBorder) return@Box
        Canvas(
            Modifier.fillMaxSize().pointerInput(displayed.width, displayed.height) {
                detectDragGestures(
                    onDragStart = { position ->
                        val rect = fittedRect(size.width.toFloat(), size.height.toFloat(), displayed.width, displayed.height)
                        val controls = currentGeometry.corners + currentGeometry.edgeMidpoints
                        activePoint = controls.indices.minByOrNull { index ->
                            val handle = rect.toOffset(controls[index])
                            hypot(handle.x - position.x, handle.y - position.y)
                        }?.takeIf { index ->
                            val handle = rect.toOffset(controls[index])
                            hypot(handle.x - position.x, handle.y - position.y) <= 52.dp.toPx()
                        } ?: -1
                    },
                    onDragEnd = { activePoint = -1 },
                    onDragCancel = { activePoint = -1 },
                    onDrag = { change, _ ->
                        if (activePoint < 0) return@detectDragGestures
                        change.consume()
                        val rect = fittedRect(size.width.toFloat(), size.height.toFloat(), displayed.width, displayed.height)
                        val point = rect.toNormalized(change.position)
                        onPointChanged(activePoint, point.x, point.y)
                    },
                )
            },
        ) {
            val rect = fittedRect(size.width, size.height, displayed.width, displayed.height)
            val corners = geometry.corners.map(rect::toOffset)
            val edges = geometry.edgeMidpoints.map(rect::toOffset)
            val path = Path().apply {
                moveTo(corners[0].x, corners[0].y)
                for (index in 1..4) lineTo(corners[index % 4].x, corners[index % 4].y)
                close()
            }
            drawPath(path, ScannerGreen, style = Stroke(2.5.dp.toPx(), cap = StrokeCap.Round))
            corners.forEach { point ->
                drawCircle(Color.Black.copy(alpha = 0.55f), 11.dp.toPx(), point)
                drawCircle(ScannerGreen, 7.dp.toPx(), point)
            }
            edges.forEach { point ->
                drawCircle(Color.Black.copy(alpha = 0.55f), 9.dp.toPx(), point)
                drawCircle(Color.White, 5.dp.toPx(), point)
            }
            if (activePoint >= 0) {
                val control = (geometry.corners + geometry.edgeMidpoints)[activePoint]
                drawMagnifier(displayed, control, rect.toOffset(control))
            }
        }
    }
}

private data class EditorFilter(
    val label: String,
    val effect: PageEffect,
    val hot: Boolean = false,
)

private val EditorFilters = listOf(
    EditorFilter("原图", PageEffect.ORIGINAL),
    EditorFilter("增亮", PageEffect.BRIGHTEN),
    EditorFilter("增强锐化", PageEffect.SHARPEN),
    EditorFilter("去阴影", PageEffect.REMOVE_SHADOW),
    EditorFilter("黑白", PageEffect.MONOCHROME),
    EditorFilter("灰度", PageEffect.GRAYSCALE),
    EditorFilter("反色", PageEffect.INVERT),
)

@Composable
private fun EditorBottomBar(
    draft: DraftPage,
    isApplying: Boolean,
    cropActive: Boolean,
    onRetake: () -> Unit,
    onRotateLeft: () -> Unit,
    onCrop: () -> Unit,
    onExtractText: () -> Unit,
    onFilter: (PageEffect) -> Unit,
    onConfirm: () -> Unit,
) {
    val selected = draft.options.displayEffect()
    val previews by produceState<Map<PageEffect, android.graphics.Bitmap>>(emptyMap(),
        draft.originalPath, File(draft.originalPath).lastModified()) {
        val generated = withContext(Dispatchers.Default) {
            val source = runCatching { decodeOrientedBitmap(File(draft.originalPath), maxEdge = 320) }.getOrNull()
            if (source == null) emptyMap() else try {
                PageEffect.entries.mapNotNull { effect ->
                    runCatching { effect to DocumentEffects.thumbnail(source, effect) }.getOrNull()
                }.toMap()
            } finally { source.recycle() }
        }
        value = generated
    }
    Column(Modifier.fillMaxWidth().background(Color(0xFF121416))) {
        LazyRow(
            modifier = Modifier.fillMaxWidth().testTag("editor-filters").padding(top = 10.dp, bottom = 6.dp),
            contentPadding = PaddingValues(horizontal = 12.dp),
            horizontalArrangement = Arrangement.spacedBy(10.dp),
        ) {
            items(EditorFilters, key = { it.label }) { filter ->
                EditorFilterTile(
                    filter = filter,
                    preview = previews[filter.effect],
                    rotationDegrees = draft.geometry.sourceRotationDegrees,
                    selected = filter.effect == selected,
                    enabled = !isApplying && !draft.boundaryDetectionPending,
                    onClick = { onFilter(filter.effect) },
                )
            }
        }
        Row(
            modifier = Modifier.fillMaxWidth().padding(start = 4.dp, end = 12.dp, bottom = 8.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            EditorToolButton("重拍这张", Icons.Default.CameraAlt, onRetake, Modifier.weight(1f), enabled = !isApplying)
            EditorToolButton("左转", Icons.Default.RotateLeft, onRotateLeft, Modifier.weight(1f), enabled = !isApplying)
            EditorToolButton("裁剪", Icons.Default.Crop, onCrop, Modifier.weight(1f), selected = cropActive, enabled = !isApplying)
            EditorToolButton("提取文字", Icons.Default.TextFields, onExtractText, Modifier.weight(1f), enabled = !isApplying)
            FilledIconButton(
                onClick = onConfirm,
                enabled = !isApplying && (!draft.options.cropEnabled ||
                    (!draft.boundaryDetectionPending && !draft.requiresBoundaryConfirmation)),
                modifier = Modifier.size(46.dp),
                shape = RoundedCornerShape(10.dp),
            ) {
                Icon(Icons.Default.Check, "完成", modifier = Modifier.size(26.dp))
            }
        }
    }
}

@Composable
private fun EditorFilterTile(
    filter: EditorFilter,
    preview: android.graphics.Bitmap?,
    rotationDegrees: Int,
    selected: Boolean,
    enabled: Boolean,
    onClick: () -> Unit,
) {
    Column(horizontalAlignment = Alignment.CenterHorizontally, modifier = Modifier.width(68.dp).clickable(enabled = enabled, onClick = onClick)) {
        Box(
            Modifier
                .size(width = 64.dp, height = 48.dp)
                .clip(RoundedCornerShape(6.dp))
                .border(if (selected) 2.dp else 0.dp, Color(0xFF3DDCFF), RoundedCornerShape(6.dp))
                .background(Color(0xFF2A2D30)),
        ) {
            preview?.let {
                val image = remember(it) { it.asImageBitmap() }
                Canvas(Modifier.fillMaxSize()) {
                    val turned = rotationDegrees % 180 != 0
                    val fitScale = maxOf(size.width / (if (turned) image.height else image.width),
                        size.height / (if (turned) image.width else image.height))
                    withTransform({
                        translate(size.width / 2f, size.height / 2f)
                        rotate(rotationDegrees.toFloat(), Offset.Zero)
                        scale(fitScale, fitScale, Offset.Zero)
                    }) {
                        drawImage(image, Offset(-image.width / 2f, -image.height / 2f))
                    }
                }
            }
            if (filter.hot) {
                Text(
                    "HOT",
                    color = Color.White,
                    fontSize = 8.sp,
                    fontWeight = FontWeight.Bold,
                    modifier = Modifier
                        .align(Alignment.TopEnd)
                        .background(Color(0xFFE53935), RoundedCornerShape(bottomStart = 4.dp))
                        .padding(horizontal = 3.dp, vertical = 1.dp),
                )
            }
        }
        Spacer(Modifier.height(4.dp))
        Text(
            filter.label,
            color = if (selected) Color(0xFF3DDCFF) else Color.White.copy(alpha = 0.86f),
            fontSize = 11.sp,
            maxLines = 1,
            overflow = TextOverflow.Ellipsis,
        )
    }
}

@Composable
private fun EditorToolButton(
    label: String,
    icon: androidx.compose.ui.graphics.vector.ImageVector,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
    selected: Boolean = false,
    enabled: Boolean = true,
) {
    Column(
        modifier = modifier.clickable(enabled = enabled, onClick = onClick).padding(vertical = 4.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Icon(icon, label, tint = Color.White, modifier = Modifier.size(22.dp))
        Spacer(Modifier.height(4.dp))
        Text(label, color = Color.White, fontSize = 12.sp, maxLines = 1)
        Spacer(Modifier.height(6.dp))
        Box(
            Modifier
                .fillMaxWidth(0.55f)
                .height(2.dp)
                .background(if (selected) Color.White else Color.Transparent),
        )
    }
}

@Composable
private fun OptionSwitch(label: String, checked: Boolean, enabled: Boolean = true, onChecked: (Boolean) -> Unit) {
    Row(Modifier.fillMaxWidth().height(48.dp), verticalAlignment = Alignment.CenterVertically) {
        Text(label, modifier = Modifier.weight(1f))
        Switch(checked = checked, enabled = enabled, onCheckedChange = onChecked)
    }
}

@Composable
private fun PagesScreen(state: ScannerUiState, viewModel: ScannerViewModel, onBack: () -> Unit) {
    var editing by remember { mutableStateOf(false) }
    var ordered by remember { mutableStateOf(state.pages) }
    LaunchedEffect(state.pages, editing) {
        if (!editing) {
            ordered = state.pages
        } else {
            val current = state.pages.associateBy { it.id }
            val kept = ordered.mapNotNull { current[it.id] }
            ordered = kept + state.pages.filter { page -> kept.none { it.id == page.id } }
        }
    }
    var draggingId by remember { mutableStateOf<String?>(null) }
    var dragPoint by remember { mutableStateOf(Offset.Zero) }
    var gridCoordinates by remember { mutableStateOf<LayoutCoordinates?>(null) }
    val itemBounds = remember { androidx.compose.runtime.mutableStateMapOf<String, Rect>() }
    val appName = stringResource(R.string.app_name)
    val title = state.documentCreatedAtMillis?.let { documentTitle(appName, it) } ?: appName
    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text(title, maxLines = 1, overflow = TextOverflow.Ellipsis) },
                navigationIcon = {
                    IconButton(onClick = onBack) { Icon(Icons.AutoMirrored.Filled.ArrowBack, "返回") }
                },
                actions = {
                    IconButton(onClick = {
                        if (editing) viewModel.savePageOrder(ordered.map { it.id })
                        editing = !editing
                        draggingId = null
                    }) {
                        Icon(if (editing) Icons.Default.Check else Icons.Default.Edit, if (editing) "保存顺序" else "编辑")
                    }
                    IconButton(onClick = viewModel::exportPdf, enabled = state.pages.isNotEmpty() && !state.isApplying) {
                        Icon(Icons.Default.Share, "导出 PDF")
                    }
                },
            )
        },
        floatingActionButton = {
            FloatingActionButton(
                onClick = viewModel::showCamera,
                containerColor = ScannerGreen,
                contentColor = Color.Black,
            ) {
                Icon(Icons.Default.CameraAlt, "继续拍摄")
            }
        },
        floatingActionButtonPosition = FabPosition.End,
    ) { padding ->
        val pages = if (editing) ordered else state.pages
        if (pages.isEmpty()) {
            Box(Modifier.padding(padding).fillMaxSize(), contentAlignment = Alignment.Center) {
                Text("尚无已保存页面", color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
        } else {
            Box(Modifier.padding(padding).fillMaxSize().onGloballyPositioned { gridCoordinates = it }) {
                LazyVerticalGrid(
                    columns = GridCells.Fixed(4),
                    modifier = Modifier.fillMaxSize(),
                    contentPadding = PaddingValues(start = 12.dp, end = 12.dp, top = 12.dp, bottom = 88.dp),
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                    verticalArrangement = Arrangement.spacedBy(10.dp),
                    userScrollEnabled = draggingId == null,
                ) {
                    itemsIndexed(pages, key = { _, page -> page.id }) { index, page ->
                        PageTile(
                            index = index,
                            page = page,
                            editing = editing,
                            hidden = draggingId == page.id,
                            onRecrop = { viewModel.editPage(page, fromDocument = true) },
                            onDelete = {
                                ordered = ordered.filterNot { it.id == page.id }
                                viewModel.deletePage(page.id)
                            },
                            onDragStart = { local, coordinates ->
                                val grid = gridCoordinates ?: return@PageTile
                                draggingId = page.id
                                dragPoint = grid.localPositionOf(coordinates, local)
                            },
                            onDrag = { amount ->
                                dragPoint += amount
                                val target = itemBounds.entries.firstOrNull { (id, bounds) ->
                                    id != draggingId && bounds.contains(dragPoint)
                                }?.key ?: return@PageTile
                                val from = ordered.indexOfFirst { it.id == draggingId }
                                val to = ordered.indexOfFirst { it.id == target }
                                if (from < 0 || to < 0 || from == to) return@PageTile
                                ordered = ordered.toMutableList().apply { add(to, removeAt(from)) }
                            },
                            onDragEnd = { draggingId = null },
                            onPlaced = { coordinates ->
                                val grid = gridCoordinates ?: return@PageTile
                                val topLeft = grid.localPositionOf(coordinates, Offset.Zero)
                                itemBounds[page.id] = Rect(
                                    topLeft,
                                    Size(coordinates.size.width.toFloat(), coordinates.size.height.toFloat()),
                                )
                            },
                        )
                    }
                }
                val dragging = pages.firstOrNull { it.id == draggingId }
                val bounds = draggingId?.let(itemBounds::get)
                if (dragging != null && bounds != null) {
                    PageTile(
                        index = ordered.indexOfFirst { it.id == dragging.id }.coerceAtLeast(0),
                        page = dragging,
                        editing = true,
                        hidden = false,
                        onRecrop = {},
                        onDelete = {},
                        onDragStart = { _, _ -> },
                        onDrag = {},
                        onDragEnd = {},
                        onPlaced = {},
                        modifier = Modifier
                            .zIndex(2f)
                            .offset {
                                IntOffset(
                                    (dragPoint.x - bounds.width / 2f).roundToInt(),
                                    (dragPoint.y - bounds.height / 2f).roundToInt(),
                                )
                            }
                            .width(with(LocalDensity.current) { bounds.width.toDp() }),
                    )
                }
            }
        }
    }
}

@Composable
private fun PageTile(
    index: Int,
    page: ScanPage,
    editing: Boolean,
    hidden: Boolean,
    onRecrop: () -> Unit,
    onDelete: () -> Unit,
    onDragStart: (Offset, LayoutCoordinates) -> Unit,
    onDrag: (Offset) -> Unit,
    onDragEnd: () -> Unit,
    onPlaced: (LayoutCoordinates) -> Unit,
    modifier: Modifier = Modifier,
) {
    val bitmap by rememberCachedThumbnail(page)
    val image = bitmap
    var coordinates by remember { mutableStateOf<LayoutCoordinates?>(null) }
    Column(modifier.alpha(if (hidden) 0f else 1f), horizontalAlignment = Alignment.CenterHorizontally) {
        Box(
            Modifier
                .fillMaxWidth()
                .aspectRatio(1f)
                .clip(RoundedCornerShape(6.dp))
                .background(Color.White)
                .onGloballyPositioned {
                    coordinates = it
                    onPlaced(it)
                }
                .then(
                    if (!editing) Modifier else Modifier.pointerInput(page.id) {
                        detectDragGestures(
                            onDragStart = { local -> coordinates?.let { onDragStart(local, it) } },
                            onDragEnd = onDragEnd,
                            onDragCancel = onDragEnd,
                        ) { change, amount ->
                            change.consume()
                            onDrag(amount)
                        }
                    },
                ),
        ) {
            image?.let {
                Image(it.asImageBitmap(), "第 ${index + 1} 页",
                    Modifier.fillMaxSize().testTag("page-thumbnail-${page.id}"), contentScale = ContentScale.Fit)
            }
            if (editing) {
                Icon(
                    Icons.Default.Delete,
                    "删除",
                    tint = Color.White,
                    modifier = Modifier
                        .align(Alignment.TopEnd)
                        .padding(2.dp)
                        .size(22.dp)
                        .background(Color.Black.copy(alpha = 0.55f), CircleShape)
                        .clickable(onClick = onDelete)
                        .padding(3.dp),
                )
                Icon(
                    Icons.Default.Edit,
                    "重新校正",
                    tint = Color.White,
                    modifier = Modifier
                        .align(Alignment.TopStart)
                        .padding(2.dp)
                        .size(22.dp)
                        .background(Color.Black.copy(alpha = 0.55f), CircleShape)
                        .clickable(onClick = onRecrop)
                        .padding(3.dp),
                )
            }
        }
        Spacer(Modifier.height(4.dp))
        Text("${index + 1}", fontSize = 12.sp, fontWeight = FontWeight.Medium, maxLines = 1)
    }
}

private fun ColorMode.label(): String = when (this) {
    ColorMode.COLOR -> "彩色"
    ColorMode.GRAYSCALE -> "灰度"
    ColorMode.BLACK_AND_WHITE -> "黑白"
}

private fun fittedRect(containerWidth: Float, containerHeight: Float, imageWidth: Int, imageHeight: Int): Rect {
    if (imageWidth <= 0 || imageHeight <= 0) return Rect(0f, 0f, containerWidth, containerHeight)
    val scale = minOf(containerWidth / imageWidth, containerHeight / imageHeight)
    val width = imageWidth * scale
    val height = imageHeight * scale
    val left = (containerWidth - width) / 2f
    val top = (containerHeight - height) / 2f
    return Rect(left, top, left + width, top + height)
}

private fun Rect.toOffset(point: NormalizedPoint): Offset = Offset(
    left + point.x * width,
    top + point.y * height,
)

private fun Rect.toNormalized(point: Offset): NormalizedPoint = NormalizedPoint(
    x = ((point.x - left) / width).coerceIn(0f, 1f),
    y = ((point.y - top) / height).coerceIn(0f, 1f),
)

private fun androidx.compose.ui.graphics.drawscope.DrawScope.drawMagnifier(
    bitmap: android.graphics.Bitmap,
    point: NormalizedPoint,
    handle: Offset,
) {
    val radius = 54.dp.toPx()
    val center = Offset(
        x = handle.x.coerceIn(radius + 8.dp.toPx(), size.width - radius - 8.dp.toPx()),
        y = (handle.y - radius - 54.dp.toPx()).coerceAtLeast(radius + 8.dp.toPx()),
    )
    val sourceRadius = 44
    val sourceX = (point.x * bitmap.width).roundToInt()
    val sourceY = (point.y * bitmap.height).roundToInt()
    drawContext.canvas.save()
    val clip = Path().apply { addOval(Rect(center - Offset(radius, radius), center + Offset(radius, radius))) }
    drawContext.canvas.clipPath(clip)
    drawContext.canvas.nativeCanvas.drawBitmap(
        bitmap,
        android.graphics.Rect(
            (sourceX - sourceRadius).coerceIn(0, bitmap.width - 1),
            (sourceY - sourceRadius).coerceIn(0, bitmap.height - 1),
            (sourceX + sourceRadius).coerceIn(1, bitmap.width),
            (sourceY + sourceRadius).coerceIn(1, bitmap.height),
        ),
        android.graphics.RectF(center.x - radius, center.y - radius, center.x + radius, center.y + radius),
        android.graphics.Paint(android.graphics.Paint.ANTI_ALIAS_FLAG or android.graphics.Paint.FILTER_BITMAP_FLAG),
    )
    drawContext.canvas.restore()
    drawCircle(Color.Black.copy(alpha = 0.7f), radius + 3.dp.toPx(), center, style = Stroke(5.dp.toPx()))
    drawCircle(ScannerGreen, 3.dp.toPx(), center)
}
