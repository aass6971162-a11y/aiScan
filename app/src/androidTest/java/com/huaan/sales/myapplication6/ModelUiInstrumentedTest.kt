package com.huaan.sales.myapplication6

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import androidx.lifecycle.ViewModelProvider
import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.junit4.createEmptyComposeRule
import androidx.compose.ui.test.onAllNodesWithText
import androidx.compose.ui.test.onAllNodesWithTag
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.onNodeWithContentDescription
import androidx.compose.ui.test.onNodeWithTag
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.performScrollToNode
import androidx.compose.ui.test.hasText
import com.huaan.sales.myapplication6.domain.PageEffect
import com.huaan.sales.myapplication6.data.ScanRepository
import com.huaan.sales.myapplication6.vision.decodeOrientedBitmap
import androidx.test.core.app.ActivityScenario
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import kotlinx.coroutines.delay
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout
import org.junit.Assert.*
import org.junit.Test
import org.junit.Rule
import org.junit.runner.RunWith
import java.io.File
import org.opencv.android.OpenCVLoader
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Scalar
import org.opencv.core.Point
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc

@RunWith(AndroidJUnit4::class)
class ModelUiInstrumentedTest {
    @get:Rule val compose = createEmptyComposeRule()

    @Test
    fun fortyMegapixelPageCanRotateSaveAndReopenWithoutDrawingTheOriginalBitmap() {
        val context: Context = ApplicationProvider.getApplicationContext()
        val repository = ScanRepository(context)
        val root = File(context.cacheDir, "large-page-ui-${System.nanoTime()}").apply { mkdirs() }
        val original = File(root, "large.jpg")
        check(OpenCVLoader.initLocal())
        val pixels = Mat(5140, 7680, CvType.CV_8UC3, Scalar(235.0, 240.0, 245.0))
        try {
            Imgproc.rectangle(pixels, Point(100.0, 100.0), Point(2000.0, 1700.0), Scalar(15.0, 30.0, 210.0), -1)
            check(Imgcodecs.imwrite(original.absolutePath, pixels))
        } finally { pixels.release() }
        val originalBytes = original.readBytes()
        val page = repository.importOriginal(original, "large-ui-${System.nanoTime()}",
            System.currentTimeMillis(), processed = true)
        try {
            ActivityScenario.launch(MainActivity::class.java).use { scenario ->
                lateinit var vm: ScannerViewModel
                scenario.onActivity { vm = ViewModelProvider(it)[ScannerViewModel::class.java] }
                await { vm.state.value.modelStatus.modelVersion != "loading" }
                vm.setAutoCrop(false)
                vm.openDocument(page.documentId)
                compose.waitUntil(30_000) {
                    compose.onAllNodesWithText("尚无已保存页面").fetchSemanticsNodes().isEmpty() &&
                        vm.state.value.screen == ScannerScreen.PAGES
                }
                compose.onNodeWithTag("page-thumbnail-${page.id}").assertIsDisplayed()
                vm.editPage(page, fromDocument = true)
                await { vm.state.value.screen == ScannerScreen.EDITOR && !vm.state.value.isApplying }
                compose.onNodeWithTag("editor-filters").performScrollToNode(hasText("反色"))
                compose.onNodeWithText("反色").performClick()
                await { !vm.state.value.isApplying && vm.state.value.draft?.effectPreviewPath != null }
                val effectPath = vm.state.value.draft!!.effectPreviewPath!!
                val effectBytes = File(effectPath).readBytes()
                repeat(3) {
                    compose.onNodeWithContentDescription("左转").performClick()
                    assertFalse("Rotation must not start image processing", vm.state.value.isApplying)
                }
                assertEquals(90, vm.state.value.draft!!.geometry.sourceRotationDegrees)
                assertEquals(effectPath, vm.state.value.draft!!.effectPreviewPath)
                assertArrayEquals(effectBytes, File(effectPath).readBytes())
                assertArrayEquals(originalBytes, File(page.originalPath).readBytes())
                compose.waitUntil(30_000) {
                    compose.onAllNodesWithTag("document-editor-image").fetchSemanticsNodes().isNotEmpty()
                }
                compose.onNodeWithContentDescription("完成").performClick()
                await { !vm.state.value.isApplying && vm.state.value.draft == null }
                val saved = repository.loadPages().first { it.id == page.id }
                assertEquals(90, saved.geometry.sourceRotationDegrees)
                assertEquals(5140, saved.width)
                assertEquals(7680, saved.height)
                assertArrayEquals(originalBytes, File(saved.originalPath).readBytes())
                compose.onNodeWithTag("page-thumbnail-${page.id}").assertIsDisplayed()
                vm.editPage(saved, fromDocument = true)
                await { !vm.state.value.isApplying && vm.state.value.draft != null }
                assertEquals(90, vm.state.value.draft!!.geometry.sourceRotationDegrees)
                vm.discardDraft()
            }
        } finally {
            repository.delete(page.id)
            root.deleteRecursively()
        }
    }

    @Test
    fun sixFiltersCanBeSelectedSavedAndReopenedWithFullFramePreview() {
        val context: Context = ApplicationProvider.getApplicationContext()
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        instrumentation.uiAutomation.grantRuntimePermission(context.packageName, "android.permission.CAMERA")
        val root = File(context.cacheDir, "effects-ui-${System.nanoTime()}").apply { mkdirs() }
        val photo = File(root, "photo.jpg")
        instrumentation.context.assets.open("model-golden/complex-stacked-pages.jpg").use { input ->
            photo.outputStream().use { input.copyTo(it) }
        }
        ActivityScenario.launch(MainActivity::class.java).use { scenario ->
            lateinit var vm: ScannerViewModel
            scenario.onActivity { vm = ViewModelProvider(it)[ScannerViewModel::class.java] }
            await { vm.state.value.modelStatus.modelVersion != "loading" }
            val oldAutoCrop = vm.state.value.autoCrop
            var pageId: String? = null
            try {
                vm.setAutoCrop(true)
                vm.processCapture(photo)
                await { vm.state.value.screen == ScannerScreen.EDITOR }
                await { !vm.state.value.draft!!.boundaryDetectionPending }
                vm.confirmEditorCrop()
                await { !vm.state.value.isApplying && !vm.state.value.editorCropActive }
                val filters = listOf("增亮" to PageEffect.BRIGHTEN, "增强锐化" to PageEffect.SHARPEN,
                    "去阴影" to PageEffect.REMOVE_SHADOW, "黑白" to PageEffect.MONOCHROME,
                    "灰度" to PageEffect.GRAYSCALE, "反色" to PageEffect.INVERT)
                for ((label, effect) in filters) {
                    compose.onNodeWithTag("editor-filters").performScrollToNode(hasText(label))
                    compose.onNodeWithText(label).performClick()
                    await { !vm.state.value.isApplying && vm.state.value.draft?.options?.effect == effect }
                    assertNull(vm.state.value.errorMessage)
                    val draft=vm.state.value.draft!!
                    assertNotNull(draft.cropPreviewPath)
                    assertNotEquals(draft.previewPath,draft.cropPreviewPath)
                    compose.waitForIdle()
                    scenario.onActivity { screenshotActivity(context,it,"effects-ui-${effect.name}.png") }
                }
                pageId = vm.state.value.draft!!.id
                compose.onNodeWithContentDescription("完成").performClick()
                await { vm.state.value.draft == null && vm.state.value.pages.any { it.id == pageId } }
                val page = vm.state.value.pages.first { it.id == pageId }
                assertEquals(PageEffect.INVERT,page.options.effect)
                val savedBytes=File(page.correctedPath).readBytes()
                vm.editPage(page)
                await { vm.state.value.screen == ScannerScreen.EDITOR && !vm.state.value.isApplying }
                val reopened = vm.state.value.draft!!
                assertEquals(PageEffect.INVERT,reopened.options.effect)
                assertNotNull(reopened.effectPreviewPath)
                val original=decodeOrientedBitmap(File(reopened.originalPath))
                val effect=decodeOrientedBitmap(File(reopened.effectPreviewPath!!))
                try { assertEquals(original.width.toDouble()/original.height,effect.width.toDouble()/effect.height,.005) }
                finally { original.recycle();effect.recycle() }
                compose.waitForIdle()
                scenario.onActivity { screenshotActivity(context,it,"effects-ui-reopened.png") }
                compose.onNodeWithContentDescription("完成").performClick()
                await { vm.state.value.draft == null && !vm.state.value.isApplying }
                assertArrayEquals(savedBytes,File(page.correctedPath).readBytes())
            } finally {
                vm.setAutoCrop(oldAutoCrop)
                pageId?.let(vm::deletePage)
                if (vm.state.value.draft != null) vm.discardDraft()
            }
        }
        root.deleteRecursively()
    }

    @Test
    fun stackedPagesShowRecoveredMainPageForConfirmation() {
        val context: Context = ApplicationProvider.getApplicationContext()
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        instrumentation.uiAutomation.grantRuntimePermission(context.packageName, "android.permission.CAMERA")
        val root = File(context.cacheDir, "stacked-ui-${System.nanoTime()}").apply { mkdirs() }
        val photo = File(root, "complex.jpg")
        instrumentation.context.assets.open("model-golden/complex-stacked-pages.jpg").use { input ->
            photo.outputStream().use { input.copyTo(it) }
        }
        ActivityScenario.launch(MainActivity::class.java).use { scenario ->
            lateinit var viewModel: ScannerViewModel
            scenario.onActivity { viewModel = ViewModelProvider(it)[ScannerViewModel::class.java] }
            await { viewModel.state.value.modelStatus.modelVersion != "loading" }
            viewModel.setCaptureMode(CaptureMode.BURST)
            viewModel.processCapture(photo)
            await { viewModel.state.value.screen == ScannerScreen.EDITOR }
            viewModel.startEditorCrop()
            await { !viewModel.state.value.draft!!.boundaryDetectionPending }
            val draft = viewModel.state.value.draft!!
            assertTrue(draft.requiresBoundaryConfirmation)
            assertFalse(draft.hasProcessedPreview)
            assertFalse(File(draft.previewPath).exists())
            assertTrue(draft.geometry.corners[0].x < .10f)
            assertTrue(draft.geometry.corners[1].x in .78f..85f)
            assertTrue(draft.geometry.corners[2].y in .80f..88f)
            assertTrue(draft.geometry.corners[3].y > .90f)
            compose.onNodeWithText("确认当前边框").assertIsDisplayed()
            instrumentation.waitForIdleSync()
            scenario.onActivity { screenshotActivity(context, it, "complex-stacked-pages-confirm.png") }
            viewModel.discardDraft()
        }
        root.deleteRecursively()
    }

    @Test
    fun clippedSingleCaptureRequiresConfirmationBeforeCorrection() {
        val context: Context = ApplicationProvider.getApplicationContext()
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        instrumentation.uiAutomation.grantRuntimePermission(context.packageName, "android.permission.CAMERA")
        val root = File(context.cacheDir, "clipped-ui-${System.nanoTime()}").apply { mkdirs() }
        val photo = File(root, "photo.png")
        instrumentation.context.assets.open("model-golden/distance-clipped.png").use { input ->
            photo.outputStream().use { input.copyTo(it) }
        }
        ActivityScenario.launch(MainActivity::class.java).use { scenario ->
            lateinit var viewModel: ScannerViewModel
            scenario.onActivity { viewModel = ViewModelProvider(it)[ScannerViewModel::class.java] }
            await { viewModel.state.value.modelStatus.modelVersion != "loading" }
            viewModel.setCaptureMode(CaptureMode.SINGLE)
            viewModel.processCapture(photo)
            await { viewModel.state.value.screen == ScannerScreen.EDITOR }
            viewModel.startEditorCrop()
            await { !viewModel.state.value.draft!!.boundaryDetectionPending }
            assertTrue(viewModel.state.value.draft!!.requiresBoundaryConfirmation)
            assertFalse(File(viewModel.state.value.draft!!.previewPath).exists())
            compose.onNodeWithText("边缘贴近或超出照片，边框需确认；照片外的内容无法恢复").assertIsDisplayed()
            compose.onNodeWithText("确认当前边框").assertIsDisplayed()
            viewModel.saveDraft()
            assertNotNull(viewModel.state.value.draft)
            viewModel.clearError()
            scenario.onActivity { screenshotActivity(context, it, "distance-clipped-confirm.png") }
            compose.onNodeWithText("确认当前边框").performClick()
            await { !viewModel.state.value.isApplying && viewModel.state.value.draft!!.cropPreviewPath != null }
            assertFalse(viewModel.state.value.draft!!.requiresBoundaryConfirmation)
            File(viewModel.state.value.draft!!.cropPreviewPath!!)
                .copyTo(File(context.filesDir, "distance-clipped-corrected.jpg"), overwrite = true)
            compose.onNodeWithContentDescription("完成").assertIsDisplayed()
            compose.waitForIdle()
            scenario.onActivity { screenshotActivity(context, it, "distance-clipped-result.png") }
            assertNull(viewModel.state.value.errorMessage)
            val pageId = viewModel.state.value.draft!!.id
            compose.onNodeWithContentDescription("完成").performClick()
            await { viewModel.state.value.draft == null && viewModel.state.value.pages.any { it.id == pageId } }
            val page = viewModel.state.value.pages.first { it.id == pageId }
            assertTrue(File(page.correctedPath).length() > 1000)
            viewModel.deletePage(page.id)
        }
        root.deleteRecursively()
    }

    @Test
    fun realModelResultAppearsOnOriginalCropScreenAndOptionsInvalidatePreview() {
        val context: Context = ApplicationProvider.getApplicationContext()
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        instrumentation.uiAutomation.grantRuntimePermission(context.packageName, "android.permission.CAMERA")
        val root = File(context.cacheDir, "ai-ui-${System.nanoTime()}").apply { mkdirs() }
        val photo = File(root, "photo.jpg")
        instrumentation.context.assets.open("model-golden/photo_original.jpg").use { input ->
            photo.outputStream().use { input.copyTo(it) }
        }
        ActivityScenario.launch(MainActivity::class.java).use { scenario ->
            lateinit var viewModel: ScannerViewModel
            scenario.onActivity { viewModel = ViewModelProvider(it)[ScannerViewModel::class.java] }
            await { viewModel.state.value.modelStatus.modelVersion != "loading" }
            assertTrue(viewModel.state.value.modelStatus.hasTrainedModels)
            viewModel.processCapture(photo)
            await { viewModel.state.value.screen == ScannerScreen.EDITOR }
            viewModel.startEditorCrop()
            await { !viewModel.state.value.draft!!.boundaryDetectionPending }
            compose.waitUntil(timeoutMillis = 30_000) {
                compose.onAllNodesWithText("原图").fetchSemanticsNodes().isNotEmpty()
            }
            assertTrue(viewModel.state.value.draft!!.detectionConfidence > 0.55f)
            assertTrue(viewModel.state.value.editorCropActive)
            instrumentation.waitForIdleSync()
            scenario.onActivity { screenshotActivity(context, it, "model-ui-crop.png") }
            viewModel.confirmBoundary()
            viewModel.applyDraft()
            await { !viewModel.state.value.isApplying }
            assertTrue(viewModel.state.value.draft!!.hasProcessedPreview)
            viewModel.setColorMode(com.huaan.sales.myapplication6.domain.ColorMode.GRAYSCALE)
            assertFalse(viewModel.state.value.draft!!.hasProcessedPreview)
            viewModel.updateOptions { it.copy(dewarpEnabled = true, perspectiveEnabled = true) }
            viewModel.applyDraft()
            await { !viewModel.state.value.isApplying }
            assertNull(viewModel.state.value.errorMessage)
            assertTrue(viewModel.state.value.draft!!.hasProcessedPreview)
            instrumentation.waitForIdleSync()
            scenario.onActivity { screenshotActivity(context, it, "model-ui-preview-ready.png") }
            viewModel.discardDraft()
        }
        root.deleteRecursively()
    }

    private fun await(condition: () -> Boolean) = runBlocking {
        withTimeout(30_000) { while (!condition()) delay(50) }
    }

    private fun screenshotActivity(context: Context, activity: MainActivity, name: String) {
        val view = activity.window.decorView.rootView
        val screenshot = Bitmap.createBitmap(view.width, view.height, Bitmap.Config.ARGB_8888)
        view.draw(Canvas(screenshot))
        File(context.filesDir, name).outputStream().use {
            screenshot.compress(Bitmap.CompressFormat.PNG, 100, it)
        }
        screenshot.recycle()
    }
}
