@file:OptIn(androidx.compose.material3.ExperimentalMaterial3Api::class)

package com.huaan.sales.myapplication6.ui

import android.graphics.BitmapFactory
import androidx.annotation.DrawableRes
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.combinedClickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.produceState
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.res.stringResource
import com.huaan.sales.myapplication6.R
import com.huaan.sales.myapplication6.domain.ScanDocument
import com.huaan.sales.myapplication6.domain.documentTitle
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

private data class HomeAction(
    val title: String,
    @DrawableRes val iconRes: Int,
    val imageBackground: Color,
    val onClick: () -> Unit,
)

@Composable
fun HomeScreen(
    documents: List<ScanDocument>,
    onScanDocument: () -> Unit,
    onOpenDocument: (String) -> Unit,
    onDeleteDocument: (String) -> Unit,
) {
    val snackbar = remember { SnackbarHostState() }
    val scope = rememberCoroutineScope()
    var pendingDelete by remember { mutableStateOf<ScanDocument?>(null) }
    val actions = listOf(
        HomeAction("智能扫描", R.drawable.ic_home_smart_scan, Color(0xFFE7F8EF)) {
            scope.launch { snackbar.showSnackbar("智能扫描暂未开放") }
        },
        HomeAction("扫描文档", R.drawable.ic_home_scan_document, Color(0xFFE5F6FA), onScanDocument),
    )
    Scaffold(
        topBar = { TopAppBar(title = { Text("清页扫描") }) },
        snackbarHost = { SnackbarHost(snackbar) },
    ) { padding ->
        Column(Modifier.padding(padding).fillMaxSize()) {
            Row(
                modifier = Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 20.dp),
                horizontalArrangement = Arrangement.spacedBy(10.dp),
            ) {
                actions.forEach { action ->
                    Box(Modifier.weight(1f)) { HomeActionButton(action) }
                }
                repeat(4 - actions.size) { Spacer(Modifier.weight(1f)) }
            }
            Text(
                "文档",
                modifier = Modifier.padding(start = 16.dp, end = 16.dp, bottom = 4.dp),
                style = MaterialTheme.typography.titleMedium,
            )
            if (documents.isEmpty()) {
                Box(Modifier.fillMaxWidth().weight(1f), contentAlignment = Alignment.Center) {
                    Text("还没有文档", color = MaterialTheme.colorScheme.onSurfaceVariant)
                }
            } else {
                LazyColumn(Modifier.weight(1f).fillMaxWidth()) {
                    items(documents, key = { it.id }) { document ->
                        DocumentListItem(
                            document,
                            onClick = { onOpenDocument(document.id) },
                            onLongClick = { pendingDelete = document },
                        )
                        HorizontalDivider(Modifier.padding(start = 84.dp))
                    }
                }
            }
        }
    }
    pendingDelete?.let { document ->
        val title = documentTitle(stringResource(R.string.app_name), document.createdAtMillis)
        AlertDialog(
            onDismissRequest = { pendingDelete = null },
            title = { Text("删除文档") },
            text = { Text("删除「$title」？删除后无法恢复。") },
            confirmButton = {
                TextButton(onClick = {
                    onDeleteDocument(document.id)
                    pendingDelete = null
                }) { Text("删除") }
            },
            dismissButton = {
                TextButton(onClick = { pendingDelete = null }) { Text("取消") }
            },
        )
    }
}

@Composable
private fun DocumentListItem(document: ScanDocument, onClick: () -> Unit, onLongClick: () -> Unit) {
    val cover by produceState<ImageBitmap?>(null, document.coverPath) {
        value = withContext(Dispatchers.IO) { decodeCover(document.coverPath) }
    }
    val title = documentTitle(stringResource(R.string.app_name), document.createdAtMillis)
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .combinedClickable(onClick = onClick, onLongClick = onLongClick)
            .padding(horizontal = 16.dp, vertical = 10.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Box(
            modifier = Modifier.size(56.dp).clip(RoundedCornerShape(8.dp)).background(MaterialTheme.colorScheme.surfaceVariant),
            contentAlignment = Alignment.Center,
        ) {
            cover?.let {
                Image(it, contentDescription = null, modifier = Modifier.fillMaxSize(), contentScale = ContentScale.Crop)
            }
        }
        Column(Modifier.padding(start = 12.dp).weight(1f)) {
            Text(title, fontWeight = FontWeight.Medium, maxLines = 1, overflow = TextOverflow.Ellipsis)
            Text("${document.pageCount} 页", color = MaterialTheme.colorScheme.onSurfaceVariant, fontSize = 13.sp)
        }
    }
}

private fun decodeCover(path: String): ImageBitmap? {
    val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
    BitmapFactory.decodeFile(path, bounds)
    if (bounds.outWidth <= 0 || bounds.outHeight <= 0) return null
    var sample = 1
    while (bounds.outWidth / sample > 200 || bounds.outHeight / sample > 200) sample *= 2
    val options = BitmapFactory.Options().apply { inSampleSize = sample }
    return BitmapFactory.decodeFile(path, options)?.asImageBitmap()
}

@Composable
private fun HomeActionButton(action: HomeAction) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(12.dp))
            .background(MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.55f))
            .clickable(onClick = action.onClick)
            .padding(top = 8.dp, bottom = 4.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Box(
            modifier = Modifier
                .size(36.dp)
                .clip(RoundedCornerShape(8.dp))
                .background(action.imageBackground),
            contentAlignment = Alignment.Center,
        ) {
            Image(
                painter = painterResource(action.iconRes),
                contentDescription = null,
                modifier = Modifier.size(26.dp),
                contentScale = ContentScale.Fit,
            )
        }
        Spacer(Modifier.height(4.dp))
        Text(
            text = action.title,
            modifier = Modifier.padding(horizontal = 2.dp),
            fontSize = 11.sp,
            fontWeight = FontWeight.Medium,
            textAlign = TextAlign.Center,
            maxLines = 2,
            overflow = TextOverflow.Ellipsis,
            lineHeight = 13.sp,
        )
    }
}
