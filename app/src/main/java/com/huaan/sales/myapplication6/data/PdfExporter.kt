package com.huaan.sales.myapplication6.data

import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.graphics.pdf.PdfDocument
import com.huaan.sales.myapplication6.domain.ScanPage
import com.huaan.sales.myapplication6.vision.decodeOrientedBitmap
import java.io.File
import java.io.FileOutputStream

class PdfExporter(private val exportDirectory: File) {
    init {
        exportDirectory.mkdirs()
    }

    fun export(pages: List<ScanPage>): File {
        require(pages.isNotEmpty()) { "至少需要一页才能导出 PDF" }
        val target = File(exportDirectory, "scan-${java.util.UUID.randomUUID()}.pdf")
        val temporary = File(exportDirectory, "${target.name}.tmp")
        val document = PdfDocument()
        try {
            pages.forEachIndexed { index, page ->
                // Bound Canvas allocations while retaining the existing PDF image detail.
                // PDF page coordinates do not limit the embedded image resolution.
                val bitmap = decodeOrientedBitmap(File(page.correctedPath), maxEdge = 3508)
                try {
                    val portrait = bitmap.height >= bitmap.width
                    val pageWidth = if (portrait) 1240 else 1754
                    val pageHeight = if (portrait) 1754 else 1240
                    val info = PdfDocument.PageInfo.Builder(pageWidth, pageHeight, index + 1).create()
                    val pdfPage = document.startPage(info)
                    pdfPage.canvas.drawColor(Color.WHITE)
                    val margin = 28f
                    val scale = minOf(
                        (pageWidth - margin * 2) / bitmap.width,
                        (pageHeight - margin * 2) / bitmap.height,
                    )
                    val drawWidth = bitmap.width * scale
                    val drawHeight = bitmap.height * scale
                    val left = (pageWidth - drawWidth) / 2f
                    val top = (pageHeight - drawHeight) / 2f
                    pdfPage.canvas.drawBitmap(
                        bitmap,
                        null,
                        RectF(left, top, left + drawWidth, top + drawHeight),
                        Paint(Paint.ANTI_ALIAS_FLAG or Paint.FILTER_BITMAP_FLAG),
                    )
                    document.finishPage(pdfPage)
                } finally { bitmap.recycle() }
            }
            FileOutputStream(temporary).use { output ->
                document.writeTo(output)
                output.fd.sync()
            }
            check(temporary.renameTo(target)) { "无法生成 PDF" }
            return target
        } finally {
            document.close()
            temporary.delete()
        }
    }
}
