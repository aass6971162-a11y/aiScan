package com.huaan.sales.myapplication6.data

import android.content.Context
import com.huaan.sales.myapplication6.domain.ColorMode
import com.huaan.sales.myapplication6.domain.PageEffect
import com.huaan.sales.myapplication6.domain.DocumentGeometry
import com.huaan.sales.myapplication6.domain.DraftPage
import com.huaan.sales.myapplication6.domain.NormalizedPoint
import com.huaan.sales.myapplication6.domain.ProcessingOptions
import com.huaan.sales.myapplication6.domain.LEGACY_DOCUMENT_ID
import com.huaan.sales.myapplication6.domain.ScanPage
import com.huaan.sales.myapplication6.vision.orientedImageSize
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.nio.file.Files
import java.nio.file.StandardCopyOption

class ScanRepository(context: Context) {
    private val root = File(context.filesDir, "document_scanner")
    private val pagesDirectory = File(root, "pages")
    private val captureDirectory = File(context.cacheDir, "scanner-drafts")
    private val indexFile = File(root, "pages.json")

    init {
        pagesDirectory.mkdirs()
    }

    @Synchronized
    fun loadPages(): List<ScanPage> {
        if (!indexFile.exists()) return emptyList()
        return runCatching {
            val array = JSONArray(indexFile.readText())
            buildList {
                for (index in 0 until array.length()) {
                    val page = array.getJSONObject(index).toPage()
                    if (File(page.originalPath).isFile && File(page.correctedPath).isFile) add(page)
                }
            }
        }.getOrDefault(emptyList())
    }

    @Synchronized
    fun saveDraft(
        draft: DraftPage,
        documentId: String = LEGACY_DOCUMENT_ID,
        createdAtMillis: Long = System.currentTimeMillis(),
    ): ScanPage {
        val original = File(pagesDirectory, "${draft.id}-original.jpg")
        val corrected = File(pagesDirectory, "${draft.id}-corrected.jpg")
        linkOrCopyAtomically(File(draft.originalPath), original)
        linkOrCopyAtomically(File(draft.previewPath), corrected)
        // The list creates this small cache lazily; do not block saving the full image.
        thumbnailFile(draft.id).delete()
        val (width, height) = orientedImageSize(corrected)
        val page = ScanPage(
            id = draft.id,
            originalPath = original.absolutePath,
            correctedPath = corrected.absolutePath,
            width = width,
            height = height,
            createdAtMillis = createdAtMillis,
            processingMillis = draft.processingMillis,
            modelVersion = draft.modelVersion,
            manuallyAdjusted = draft.geometry.manuallyAdjusted,
            geometry = draft.geometry,
            options = draft.options,
            documentId = documentId,
            processed = true,
        )
        val pages = loadPages().filterNot { it.id == page.id } + page
        writeIndex(pages)
        return page
    }

    @Synchronized
    fun updateDraft(draft: DraftPage): ScanPage {
        val old = loadPages().firstOrNull { it.id == draft.id } ?: return saveDraft(draft, LEGACY_DOCUMENT_ID)
        val original = File(old.originalPath)
        if (File(draft.originalPath).absolutePath != original.absolutePath) {
            linkOrCopyAtomically(File(draft.originalPath), original)
            File(draft.originalPath).delete()
        }
        val corrected = File(old.correctedPath)
        linkOrCopyAtomically(File(draft.previewPath), corrected)
        thumbnailFile(old.id).delete()
        val (width, height) = orientedImageSize(corrected)
        val updated = old.copy(
            width = width,
            height = height,
            processingMillis = draft.processingMillis,
            modelVersion = draft.modelVersion,
            manuallyAdjusted = draft.geometry.manuallyAdjusted,
            geometry = draft.geometry,
            options = draft.options,
            processed = true,
        )
        writeIndex(loadPages().map { if (it.id == updated.id) updated else it })
        return updated
    }

    fun toDraft(page: ScanPage, workingDirectory: File): DraftPage {
        workingDirectory.mkdirs()
        val preview = File(workingDirectory, "draft-${page.id}.jpg")
        return DraftPage(
            id = page.id,
            originalPath = page.originalPath,
            previewPath = preview.absolutePath,
            geometry = page.geometry,
            options = page.options,
            detectionConfidence = 1f,
            processingMillis = page.processingMillis,
            modelVersion = page.modelVersion,
            // The repository already has the output; generate a new preview only
            // after an actual edit, instead of copying a large photo to cache.
            hasProcessedPreview = false,
            boundaryDetectionComplete = page.options.cropEnabled || page.geometry.manuallyAdjusted,
            boundaryEditedByUser = page.geometry.manuallyAdjusted,
        )
    }

    /** Save an untouched JPEG without decoding, re-encoding or copying it three times. */
    @Synchronized
    fun saveOriginalDraft(
        draft: DraftPage, documentId: String, createdAtMillis: Long,
    ): ScanPage {
        require(!draft.options.cropEnabled &&
            draft.options.displayEffect() == PageEffect.ORIGINAL &&
            draft.geometry.sourceRotationDegrees == 0)
        val old = loadPages().firstOrNull { it.id == draft.id }
        if (old != null) {
            if (draft.originalPath == old.originalPath && old.modelVersion == "original" &&
                old.options == draft.options && old.geometry == draft.geometry && old.processed &&
                File(old.correctedPath).isFile) return old
            val original = File(old.originalPath)
            if (draft.originalPath != original.absolutePath) {
                copyAtomically(File(draft.originalPath), original)
            }
            linkOrCopyAtomically(original, File(old.correctedPath))
            thumbnailFile(draft.id).delete()
            val (width, height) = orientedImageSize(original)
            val updated = old.copy(width = width, height = height, processingMillis = 0,
                modelVersion = "original", manuallyAdjusted = draft.geometry.manuallyAdjusted,
                geometry = draft.geometry, options = draft.options, processed = true)
            writeIndex(loadPages().map { if (it.id == updated.id) updated else it })
            return updated
        }
        val original = File(pagesDirectory, "${draft.id}-original.jpg")
        val corrected = File(pagesDirectory, "${draft.id}-corrected.jpg")
        try {
            linkOrCopyAtomically(File(draft.originalPath), original)
            linkOrCopyAtomically(original, corrected)
            val (width, height) = orientedImageSize(original)
            val page = ScanPage(draft.id, original.absolutePath, corrected.absolutePath,
                width, height, createdAtMillis, 0, "original",
                draft.geometry.manuallyAdjusted, draft.geometry, draft.options, documentId, true)
            writeIndex(loadPages() + page)
            return page
        } catch (error: Throwable) {
            original.delete()
            corrected.delete()
            throw error
        }
    }

    @Synchronized
    fun importOriginal(
        source: File,
        documentId: String,
        createdAtMillis: Long,
        processed: Boolean = false,
    ): ScanPage {
        val id = java.util.UUID.randomUUID().toString()
        val original = File(pagesDirectory, "$id-original.jpg")
        val corrected = File(pagesDirectory, "$id-corrected.jpg")
        linkOrCopyAtomically(source, original)
        linkOrCopyAtomically(original, corrected)
        thumbnailFile(id).delete()
        val (width, height) = orientedImageSize(original)
        val page = ScanPage(
            id = id,
            originalPath = original.absolutePath,
            correctedPath = corrected.absolutePath,
            width = width,
            height = height,
            createdAtMillis = createdAtMillis,
            processingMillis = 0,
            modelVersion = "original",
            manuallyAdjusted = false,
            geometry = DocumentGeometry.fromCorners(
                listOf(
                    NormalizedPoint(0f, 0f),
                    NormalizedPoint(1f, 0f),
                    NormalizedPoint(1f, 1f),
                    NormalizedPoint(0f, 1f),
                ),
            ),
            options = ProcessingOptions(removeShadows = false, cropEnabled = false),
            documentId = documentId,
            processed = processed,
        )
        writeIndex(loadPages().filterNot { it.id == page.id } + page)
        return page
    }

    /** One durable index write for a burst, with originals retained on failure. */
    @Synchronized
    fun importOriginalBatch(
        sources: List<Pair<File, Long>>, documentId: String,
    ): List<ScanPage> {
        val added = mutableListOf<ScanPage>()
        try {
            for ((source, createdAtMillis) in sources) {
                require(source.isFile) { "拍摄文件不存在: ${source.name}" }
                val id = java.util.UUID.randomUUID().toString()
                val original = File(pagesDirectory, "$id-original.jpg")
                val corrected = File(pagesDirectory, "$id-corrected.jpg")
                try {
                    linkOrCopyAtomically(source, original)
                    linkOrCopyAtomically(original, corrected)
                    val (width, height) = orientedImageSize(original)
                    val page = ScanPage(id, original.absolutePath, corrected.absolutePath,
                        width, height, createdAtMillis, 0, "original", false,
                        DocumentGeometry.fromCorners(listOf(
                            NormalizedPoint(0f, 0f), NormalizedPoint(1f, 0f),
                            NormalizedPoint(1f, 1f), NormalizedPoint(0f, 1f),
                        )), ProcessingOptions(removeShadows = false, cropEnabled = false),
                        documentId, false)
                    added += page
                } catch (error: Throwable) {
                    original.delete()
                    corrected.delete()
                    throw error
                }
            }
            writeIndex(loadPages() + added)
            return added
        } catch (error: Throwable) {
            added.forEach { File(it.originalPath).delete(); File(it.correctedPath).delete() }
            throw error
        }
    }

    @Synchronized
    fun applyCorrection(pageId: String, draft: DraftPage): ScanPage {
        val old = loadPages().first { it.id == pageId }
        copyAtomically(File(draft.previewPath), File(old.correctedPath))
        thumbnailFile(pageId).delete()
        val bounds = android.graphics.BitmapFactory.Options().also { it.inJustDecodeBounds = true }
        android.graphics.BitmapFactory.decodeFile(old.correctedPath, bounds)
        val updated = old.copy(
            width = bounds.outWidth,
            height = bounds.outHeight,
            processingMillis = draft.processingMillis,
            modelVersion = draft.modelVersion,
            manuallyAdjusted = draft.geometry.manuallyAdjusted,
            geometry = draft.geometry,
            options = draft.options,
            processed = true,
        )
        writeIndex(loadPages().map { if (it.id == updated.id) updated else it })
        return updated
    }

    @Synchronized
    fun markProcessed(pageId: String): List<ScanPage> {
        val pages = loadPages().map { if (it.id == pageId) it.copy(processed = true) else it }
        writeIndex(pages)
        return pages
    }

    @Synchronized
    fun deleteDocument(documentId: String): List<ScanPage> {
        val pages = loadPages()
        pages.filter { it.documentId == documentId }.forEach {
            File(it.originalPath).delete()
            File(it.correctedPath).delete()
            thumbnailFile(it.id).delete()
            File(pagesDirectory, "${it.id}-thumb.jpg").delete()
        }
        return pages.filterNot { it.documentId == documentId }.also(::writeIndex)
    }

    @Synchronized
    fun delete(pageId: String): List<ScanPage> {
        val pages = loadPages()
        pages.firstOrNull { it.id == pageId }?.let {
            File(it.originalPath).delete()
            File(it.correctedPath).delete()
            thumbnailFile(it.id).delete()
            File(pagesDirectory, "${it.id}-thumb.jpg").delete()
        }
        return pages.filterNot { it.id == pageId }.also(::writeIndex)
    }

    @Synchronized
    fun reorder(documentId: String, orderedIds: List<String>): List<ScanPage> {
        val pages = loadPages()
        val byId = pages.associateBy { it.id }
        val orderedIdsSet = orderedIds.toSet()
        val ordered = orderedIds.mapNotNull(byId::get).filter { it.documentId == documentId }
        val missing = pages.filter { it.documentId == documentId && it.id !in orderedIdsSet }
        val replacement = ordered + missing
        val result = mutableListOf<ScanPage>()
        var inserted = false
        for (page in pages) {
            if (page.documentId == documentId) {
                if (!inserted) {
                    result += replacement
                    inserted = true
                }
            } else {
                result += page
            }
        }
        if (!inserted) result += replacement
        writeIndex(result)
        return result
    }

    @Synchronized
    fun move(pageId: String, offset: Int): List<ScanPage> {
        val pages = loadPages().toMutableList()
        val from = pages.indexOfFirst { it.id == pageId }
        if (from == -1) return pages
        val to = (from + offset).coerceIn(0, pages.lastIndex)
        if (from != to) pages.add(to, pages.removeAt(from))
        writeIndex(pages)
        return pages
    }

    private fun writeIndex(pages: List<ScanPage>) {
        root.mkdirs()
        val temporary = File(root, "pages.json.tmp")
        temporary.writeText(JSONArray(pages.map { it.toJson() }).toString())
        FileOutputStream(temporary, true).use { it.fd.sync() }
        Files.move(temporary.toPath(), indexFile.toPath(), StandardCopyOption.REPLACE_EXISTING)
    }

    private fun thumbnailFile(id: String) = File(pagesDirectory, "$id-list.jpg")

    private fun copyAtomically(source: File, target: File) {
        require(source.isFile) { "源文件不存在: ${source.name}" }
        target.parentFile?.mkdirs()
        val temporary = File(target.parentFile, "${target.name}.tmp")
        try {
            FileInputStream(source).use { input ->
                FileOutputStream(temporary).use { output ->
                    input.copyTo(output)
                    output.fd.sync()
                }
            }
            Files.move(temporary.toPath(), target.toPath(), StandardCopyOption.REPLACE_EXISTING)
        } finally { temporary.delete() }
    }

    private fun linkOrCopyAtomically(source: File, target: File) {
        require(source.isFile) { "源文件不存在: ${source.name}" }
        if (source.absolutePath == target.absolutePath) return
        val temporary = File(target.parentFile, "${target.name}.tmp")
        temporary.delete()
        try {
            val appOwnedSource = source.canonicalPath.let { path ->
                path.startsWith(pagesDirectory.canonicalPath + File.separator) ||
                    path.startsWith(captureDirectory.canonicalPath + File.separator)
            }
            val linked = appOwnedSource && runCatching {
                Files.createLink(temporary.toPath(), source.toPath())
            }.isSuccess
            if (!linked) {
                FileInputStream(source).use { input ->
                    FileOutputStream(temporary).use { output ->
                        input.copyTo(output)
                        output.fd.sync()
                    }
                }
            }
            Files.move(temporary.toPath(), target.toPath(), StandardCopyOption.REPLACE_EXISTING)
        } finally { temporary.delete() }
    }
}

fun ScanPage.cachedThumbnailFile(): File = File(originalPath).resolveSibling("$id-list.jpg")

private fun ScanPage.toJson(): JSONObject = JSONObject().apply {
    put("id", id)
    put("originalPath", originalPath)
    put("correctedPath", correctedPath)
    put("width", width)
    put("height", height)
    put("createdAtMillis", createdAtMillis)
    put("processingMillis", processingMillis)
    put("modelVersion", modelVersion)
    put("manuallyAdjusted", manuallyAdjusted)
    put("geometry", geometry.toJson())
    put("options", options.toJson())
    put("documentId", documentId)
    put("processed", processed)
}

private fun JSONObject.toPage(): ScanPage = ScanPage(
    id = getString("id"),
    originalPath = getString("originalPath"),
    correctedPath = getString("correctedPath"),
    width = getInt("width"),
    height = getInt("height"),
    createdAtMillis = getLong("createdAtMillis"),
    processingMillis = getLong("processingMillis"),
    modelVersion = getString("modelVersion"),
    manuallyAdjusted = getBoolean("manuallyAdjusted"),
    geometry = getJSONObject("geometry").toGeometry(),
    options = getJSONObject("options").toOptions(defaultCropEnabled = getString("modelVersion") != "original"),
    documentId = optString("documentId", LEGACY_DOCUMENT_ID).ifBlank { LEGACY_DOCUMENT_ID },
    processed = optBoolean("processed", true),
)

private fun DocumentGeometry.toJson(): JSONObject = JSONObject().apply {
    put("corners", JSONArray(corners.map { JSONArray(listOf(it.x, it.y)) }))
    put("edgeMidpoints", JSONArray(edgeMidpoints.map { JSONArray(listOf(it.x, it.y)) }))
    put("manuallyAdjusted", manuallyAdjusted)
    put("sourceRotationDegrees", sourceRotationDegrees)
}

private fun JSONObject.toGeometry(): DocumentGeometry = DocumentGeometry(
    corners = getJSONArray("corners").toPoints(),
    edgeMidpoints = getJSONArray("edgeMidpoints").toPoints(),
    manuallyAdjusted = optBoolean("manuallyAdjusted"),
    sourceRotationDegrees = optInt("sourceRotationDegrees", 0),
)

private fun JSONArray.toPoints(): List<NormalizedPoint> = buildList {
    for (index in 0 until length()) {
        val point = getJSONArray(index)
        add(NormalizedPoint(point.getDouble(0).toFloat(), point.getDouble(1).toFloat()))
    }
}

private fun ProcessingOptions.toJson(): JSONObject = JSONObject().apply {
    put("cropEnabled", cropEnabled)
    put("perspectiveEnabled", perspectiveEnabled)
    put("dewarpEnabled", dewarpEnabled)
    put("dewarpStrength", dewarpStrength)
    put("removeShadows", removeShadows)
    put("colorMode", colorMode.name)
    put("jpegQuality", jpegQuality)
    put("effect", effect.name)
}

private fun JSONObject.toOptions(defaultCropEnabled: Boolean): ProcessingOptions = ProcessingOptions(
    // Earlier versions always cropped completed edits; unedited imports kept the original.
    cropEnabled = optBoolean("cropEnabled", defaultCropEnabled),
    perspectiveEnabled = optBoolean("perspectiveEnabled", true),
    dewarpEnabled = optBoolean("dewarpEnabled", true),
    dewarpStrength = optDouble("dewarpStrength", 0.5).toFloat(),
    removeShadows = optBoolean("removeShadows", true),
    colorMode = runCatching { ColorMode.valueOf(optString("colorMode", ColorMode.COLOR.name)) }
        .getOrDefault(ColorMode.COLOR),
    jpegQuality = optInt("jpegQuality", 94),
    effect = if (has("effect")) {
        runCatching { PageEffect.valueOf(getString("effect")) }.getOrDefault(PageEffect.ORIGINAL)
    } else {
        PageEffect.ORIGINAL
    },
)
