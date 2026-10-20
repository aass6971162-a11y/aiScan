package com.huaan.sales.myapplication6

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.net.Uri
import androidx.compose.ui.graphics.asAndroidBitmap
import androidx.compose.ui.test.captureToImage
import androidx.compose.ui.test.junit4.createEmptyComposeRule
import androidx.compose.ui.test.onNodeWithTag
import androidx.compose.ui.test.onAllNodesWithTag
import androidx.exifinterface.media.ExifInterface
import androidx.lifecycle.ViewModelProvider
import androidx.test.core.app.ActivityScenario
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.huaan.sales.myapplication6.data.ScanRepository
import com.huaan.sales.myapplication6.domain.PageEffect
import com.huaan.sales.myapplication6.domain.ScanPage
import com.huaan.sales.myapplication6.vision.decodeOrientedBitmap
import kotlinx.coroutines.delay
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout
import org.junit.Assert.*
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

@RunWith(AndroidJUnit4::class)
class CropBehaviorInstrumentedTest {
    @get:Rule val compose = createEmptyComposeRule()
    private val context: Context = ApplicationProvider.getApplicationContext()
    private val repository = ScanRepository(context)

    @Test
    fun completedCapturedPhotoRemovesOnlyItsTemporaryCameraFile() = withScanner { vm, root ->
        val fixture = fixture(root)
        val capture = vm.newCaptureFile()
        fixture.copyTo(capture)
        val bytes = capture.readBytes()
        vm.setAutoCrop(false)
        vm.setCaptureMode(CaptureMode.SINGLE)
        vm.processCapture(capture)
        await { vm.state.value.screen == ScannerScreen.EDITOR }
        val page = save(vm)
        assertFalse(capture.exists())
        assertArrayEquals(bytes, File(page.originalPath).readBytes())
        assertArrayEquals(bytes, File(page.correctedPath).readBytes())
        assertTrue(fixture.exists())
    }

    @Test
    fun fullFrameCaptureOpensWithoutModelWaitAndDetectsOnlyWhenCropIsEnabled() = withScanner { vm, root ->
        val original = File(root, "document.jpg")
        InstrumentationRegistry.getInstrumentation().context.assets.open("model-golden/photo_original.jpg").use { input ->
            original.outputStream().use { input.copyTo(it) }
        }
        vm.setAutoCrop(false)
        vm.setCaptureMode(CaptureMode.SINGLE)
        vm.processCapture(original)
        await { vm.state.value.screen == ScannerScreen.EDITOR }
        assertEquals("original", vm.state.value.draft!!.modelVersion)
        assertFalse(vm.state.value.draft!!.options.cropEnabled)
        assertFalse(File(vm.state.value.draft!!.previewPath).exists())
        vm.setAutoCrop(true)
        assertEquals("original", vm.state.value.draft!!.modelVersion)
        assertFalse(vm.state.value.draft!!.options.cropEnabled)
        vm.startEditorCrop()
        await { !vm.state.value.draft!!.boundaryDetectionPending }
        assertNull(vm.state.value.errorMessage)
        assertTrue(vm.state.value.editorCropActive)
        assertTrue(vm.state.value.draft!!.options.cropEnabled)
        assertNotEquals("original", vm.state.value.draft!!.modelVersion)
        assertTrue(vm.state.value.draft!!.geometry.corners[0].x > 0f)
        assertFalse(File(vm.state.value.draft!!.previewPath).exists())
        assertNull(vm.state.value.draft!!.cropPreviewPath)
    }

    @Test
    fun manualBoundaryEditDuringDetectionIsKeptWhenDetectionFinishes() = withScanner { vm, root ->
        val original = fixture(root)
        vm.setAutoCrop(false)
        vm.processCapture(original)
        await { vm.state.value.screen == ScannerScreen.EDITOR }

        vm.startEditorCrop()
        val detectionWasPending = vm.state.value.draft!!.boundaryDetectionPending
        val editedCorner = .17f to .19f
        vm.updateControlPoint(0, editedCorner.first, editedCorner.second)

        if (detectionWasPending) assertTrue(vm.state.value.draft!!.boundaryDetectionPending)
        await { !vm.state.value.draft!!.boundaryDetectionPending }

        val draft = vm.state.value.draft!!
        assertEquals(editedCorner.first, draft.geometry.corners[0].x, .0001f)
        assertEquals(editedCorner.second, draft.geometry.corners[0].y, .0001f)
        assertTrue(draft.boundaryEditedByUser)
    }

    @Test
    fun burstBatchKeepsOrderAndOnlyOneCompleteIndexIsPublished() = withScanner { vm, root ->
        val first = fixture(root)
        val second = File(root, "second.jpg")
        first.copyTo(second)
        vm.showHome()
        vm.startNewScan()
        vm.setCaptureMode(CaptureMode.BURST)
        for (file in listOf(first, second)) {
            val capture = vm.newCaptureFile()
            file.copyTo(capture)
            vm.addBurstShot(capture)
        }
        vm.finishBurst()
        await { !vm.state.value.isApplying && vm.state.value.screen == ScannerScreen.PAGES }
        assertEquals(2, vm.state.value.pages.size)
        assertEquals(vm.state.value.pages.map { it.id },
            repository.loadPages().filter { it.documentId == vm.state.value.activeDocumentId }.map { it.id })
        vm.state.value.pages.forEach { page ->
            assertFalse(page.processed)
            assertArrayEquals(first.readBytes(), File(page.correctedPath).readBytes())
        }
    }

    @Test
    fun importAndBurstKeepEveryPixelWithEitherCropSettingAndShowFullThumbnails() = withScanner { vm, root ->
        val fixture = fixture(root)
        val bytes = fixture.readBytes()
        for (cropEnabled in listOf(false, true)) {
            vm.showHome()
            vm.setAutoCrop(cropEnabled)
            vm.importImages(listOf(Uri.fromFile(fixture)))
            await { !vm.state.value.isApplying && vm.state.value.screen == ScannerScreen.PAGES }
            val imported = vm.state.value.pages.single()
            // Give any list-triggered processing time to run, then revisit the document.
            compose.waitForIdle()
            runBlocking { delay(2_000) }
            vm.openDocument(imported.documentId)
            assertOriginal(imported, bytes)
            assertThumbnailCorners(imported)

            vm.showHome()
            vm.startNewScan()
            vm.setCaptureMode(CaptureMode.BURST)
            val shot = vm.newCaptureFile()
            fixture.copyTo(shot)
            vm.addBurstShot(shot)
            vm.finishBurst()
            await { !vm.state.value.isApplying && vm.state.value.screen == ScannerScreen.PAGES }
            val burst = vm.state.value.pages.single()
            compose.waitForIdle()
            runBlocking { delay(2_000) }
            vm.openDocument(burst.documentId)
            assertOriginal(burst, bytes)
            assertThumbnailCorners(burst)

            if (cropEnabled) {
                vm.editPage(burst, fromDocument = true)
                await { !vm.state.value.isApplying && vm.state.value.draft != null }
                assertTrue(vm.state.value.draft!!.options.cropEnabled)
                assertFalse(vm.state.value.draft!!.hasProcessedPreview)
                // Opening/detecting/discarding an edit must not replace the saved full photo.
                assertOriginal(burst, bytes)
                vm.discardDraft()
                assertOriginal(burst, bytes)
            }
        }
    }

    @Test
    fun onlyCompletingCropEnabledEditCropsAndDisablingItInvalidatesOldResults() = withScanner { vm, root ->
        val original = fixture(root)
        val bytes = original.readBytes()
        vm.setAutoCrop(false)
        vm.setCaptureMode(CaptureMode.SINGLE)
        vm.processCapture(original)
        await { vm.state.value.screen == ScannerScreen.EDITOR }
        assertFalse(vm.state.value.draft!!.hasProcessedPreview)
        assertFalse(File(vm.state.value.draft!!.previewPath).exists())
        // A deliberately small boundary must have no effect when cropping is off.
        setSmallBoundary(vm)
        var page = save(vm)
        assertArrayEquals(bytes, File(page.correctedPath).readBytes())
        assertEquals(800, page.width)
        assertEquals(500, page.height)
        assertFalse(page.options.cropEnabled)
        vm.editPage(page, fromDocument = true)
        await { !vm.state.value.isApplying && vm.state.value.draft != null }
        assertFalse(vm.state.value.draft!!.options.cropEnabled)
        assertArrayEquals(bytes, File(page.correctedPath).readBytes())
        vm.startEditorCrop()
        await { !vm.state.value.isApplying }
        assertNull(vm.state.value.errorMessage)
        assertTrue(vm.state.value.editorCropActive)
        assertFalse(vm.state.value.draft!!.hasProcessedPreview)
        vm.confirmEditorCrop()
        await { !vm.state.value.isApplying }
        assertNotNull(vm.state.value.draft!!.cropPreviewPath)
        assertFalse(File(vm.state.value.draft!!.previewPath).exists())
        await { !vm.state.value.isApplying && vm.state.value.draft != null }
        page = save(vm)
        assertTrue(page.options.cropEnabled)
        assertTrue(page.width in 395..405)
        assertTrue(page.height in 295..305)
        assertArrayEquals(bytes, File(page.originalPath).readBytes())
        val croppedBytes = File(page.correctedPath).readBytes()
        // The camera switch does not remove a crop the editor button already applied.
        vm.setAutoCrop(false)
        vm.editPage(page, fromDocument = true)
        await { !vm.state.value.isApplying && vm.state.value.draft != null }
        assertTrue(vm.state.value.draft!!.options.cropEnabled)
        assertFalse(vm.state.value.editorCropActive)
        assertArrayEquals(croppedBytes, File(page.correctedPath).readBytes())
        page = save(vm)
        assertTrue(page.options.cropEnabled)
        assertTrue(page.width in 395..405)
        assertTrue(page.height in 295..305)
        assertArrayEquals(croppedBytes, File(page.correctedPath).readBytes())
        vm.editPage(page, fromDocument = true)
        await { !vm.state.value.isApplying && vm.state.value.draft != null }
        vm.keepFullPhoto()
        page = save(vm)
        assertArrayEquals(bytes, File(page.correctedPath).readBytes())
    }

    @Test
    fun cropOffPreservesExifUntilUserExplicitlyAppliesRotationAndFilter() = withScanner { vm, root ->
        val original = fixture(root)
        ExifInterface(original).apply {
            setAttribute(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_ROTATE_90.toString())
            saveAttributes()
        }
        val bytes = original.readBytes()
        vm.setAutoCrop(false)
        vm.setCaptureMode(CaptureMode.SINGLE)
        vm.processCapture(original)
        await { vm.state.value.screen == ScannerScreen.EDITOR }
        setSmallBoundary(vm)
        var page = save(vm)
        assertArrayEquals(bytes, File(page.correctedPath).readBytes())
        assertEquals(500, page.width)
        assertEquals(800, page.height)

        vm.editPage(page, fromDocument = true)
        await { !vm.state.value.isApplying && vm.state.value.draft != null }
        vm.setPageEffect(PageEffect.INVERT)
        await { !vm.state.value.isApplying }
        assertNull(vm.state.value.errorMessage)
        val effectPath = vm.state.value.draft!!.effectPreviewPath!!
        val effectBytes = File(effectPath).readBytes()
        vm.rotateLeft()
        assertEquals(270, vm.state.value.draft!!.geometry.sourceRotationDegrees)
        assertArrayEquals(effectBytes, File(effectPath).readBytes())
        page = save(vm)
        // EXIF 90° plus an explicit left turn returns the whole 800 x 500 photo.
        assertEquals(800, page.width)
        assertEquals(500, page.height)
        assertArrayEquals(bytes, File(page.originalPath).readBytes())
        val result = decodeOrientedBitmap(File(page.correctedPath))
        try {
            assertColor(Color.CYAN, result.getPixel(30, 30))
            assertColor(Color.MAGENTA, result.getPixel(770, 30))
            assertColor(Color.YELLOW, result.getPixel(770, 470))
            assertColor(Color.BLUE, result.getPixel(30, 470))
        } finally { result.recycle() }
        val savedBytes = File(page.correctedPath).readBytes()
        vm.editPage(page, fromDocument = true)
        await { !vm.state.value.isApplying && vm.state.value.draft != null }
        page = save(vm)
        assertEquals(800, page.width)
        assertEquals(500, page.height)
        assertArrayEquals(savedBytes, File(page.correctedPath).readBytes())
    }

    private fun setSmallBoundary(vm: ScannerViewModel) {
        listOf(.25f to .2f, .75f to .2f, .75f to .8f, .25f to .8f).forEachIndexed { index, point ->
            vm.updateControlPoint(index, point.first, point.second)
        }
    }

    private fun save(vm: ScannerViewModel): ScanPage {
        val id = vm.state.value.draft!!.id
        vm.saveDraft()
        await { !vm.state.value.isApplying }
        assertNull(vm.state.value.errorMessage)
        assertNull(vm.state.value.draft)
        return repository.loadPages().first { it.id == id }
    }

    private fun assertOriginal(page: ScanPage, bytes: ByteArray) {
        val saved = repository.loadPages().first { it.id == page.id }
        assertFalse(saved.processed)
        assertFalse(saved.options.cropEnabled)
        assertEquals(800, saved.width)
        assertEquals(500, saved.height)
        assertArrayEquals(bytes, File(saved.originalPath).readBytes())
        assertArrayEquals(bytes, File(saved.correctedPath).readBytes())
    }

    private fun assertThumbnailCorners(page: ScanPage) {
        compose.waitUntil(30_000) {
            compose.onAllNodesWithTag("page-thumbnail-${page.id}").fetchSemanticsNodes().isNotEmpty()
        }
        val image = compose.onNodeWithTag("page-thumbnail-${page.id}").captureToImage().asAndroidBitmap()
        val fittedHeight = image.width * 500f / 800
        val top = (image.height - fittedHeight) / 2f
        assertColor(Color.RED, image.getPixel((image.width * .04f).toInt(), (top + fittedHeight * .06f).toInt()))
        assertColor(Color.GREEN, image.getPixel((image.width * .96f).toInt(), (top + fittedHeight * .06f).toInt()))
        assertColor(Color.BLUE, image.getPixel((image.width * .96f).toInt(), (top + fittedHeight * .94f).toInt()))
        assertColor(Color.YELLOW, image.getPixel((image.width * .04f).toInt(), (top + fittedHeight * .94f).toInt()))
    }

    private fun assertColor(expected: Int, actual: Int) {
        assertEquals(Color.red(expected).toDouble(), Color.red(actual).toDouble(), 30.0)
        assertEquals(Color.green(expected).toDouble(), Color.green(actual).toDouble(), 30.0)
        assertEquals(Color.blue(expected).toDouble(), Color.blue(actual).toDouble(), 30.0)
    }

    private fun fixture(root: File): File {
        val bitmap = Bitmap.createBitmap(800, 500, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(bitmap)
        val paint = Paint()
        canvas.drawColor(Color.DKGRAY)
        paint.color = Color.WHITE
        canvas.drawRect(150f, 80f, 650f, 420f, paint)
        listOf(Color.RED to (0f to 0f), Color.GREEN to (720f to 0f),
            Color.BLUE to (720f to 420f), Color.YELLOW to (0f to 420f)).forEach { (color, point) ->
            paint.color = color
            canvas.drawRect(point.first, point.second, point.first + 80, point.second + 80, paint)
        }
        return File(root, "photo.jpg").also { file ->
            file.outputStream().use { bitmap.compress(Bitmap.CompressFormat.JPEG, 98, it) }
            bitmap.recycle()
        }
    }

    private fun withScanner(block: (ScannerViewModel, File) -> Unit) {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        instrumentation.uiAutomation.grantRuntimePermission(context.packageName, "android.permission.CAMERA")
        val before = repository.loadPages().map { it.id }.toSet()
        val root = File(context.cacheDir, "crop-regression-${System.nanoTime()}").apply { mkdirs() }
        try {
            ActivityScenario.launch(MainActivity::class.java).use { scenario ->
                lateinit var vm: ScannerViewModel
                scenario.onActivity { vm = ViewModelProvider(it)[ScannerViewModel::class.java] }
                await { vm.state.value.modelStatus.modelVersion != "loading" }
                val oldCrop = vm.state.value.autoCrop
                try { block(vm, root) } finally {
                    if (vm.state.value.draft != null) vm.discardDraft()
                    vm.setAutoCrop(oldCrop)
                }
            }
        } finally {
            repository.loadPages().filterNot { it.id in before }.forEach { repository.delete(it.id) }
            root.deleteRecursively()
        }
    }

    private fun await(condition: () -> Boolean) = runBlocking {
        withTimeout(30_000) { while (!condition()) delay(50) }
    }
}
