package com.huaan.sales.myapplication6.ml

import android.content.Context
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream
import java.io.InputStream
import java.security.MessageDigest
import org.tensorflow.lite.Interpreter
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession

enum class InferenceBackend { GPU, VENDOR_NPU, XNNPACK_CPU, ONNX_CPU, OPENCV_FALLBACK }

data class ModelDescriptor(
    val id: String,
    val file: String,
    val version: String,
    val sha256: String,
)

data class ModelRuntimeStatus(
    val backend: InferenceBackend,
    val modelVersion: String,
    val readyModelIds: Set<String>,
    val missingModelIds: Set<String>,
    val message: String,
) {
    // Capabilities are independent: missing shadow weights must not disable corner inference.
    val hasTrainedModels: Boolean get() = "document_corners" in readyModelIds
}

class ModelAssetRegistry(private val context: Context) {
    fun descriptors(): List<ModelDescriptor> {
        val manifest = context.assets.open(MANIFEST_PATH).bufferedReader().use { it.readText() }
        val models = JSONObject(manifest).getJSONArray("models")
        return buildList {
            for (index in 0 until models.length()) {
                val item = models.getJSONObject(index)
                add(
                    ModelDescriptor(
                        id = item.getString("id"),
                        file = item.getString("file"),
                        version = item.getString("version"),
                        sha256 = item.optString("sha256"),
                    ),
                )
            }
        }
    }

    fun verify(): ModelRuntimeStatus {
        val descriptors = descriptors()
        val ready = mutableSetOf<String>()
        val missing = mutableSetOf<String>()
        descriptors.forEach { descriptor ->
            if (descriptor.version == "pending-training" || descriptor.sha256.length != 64) {
                missing += descriptor.id
                return@forEach
            }
            val digest = runCatching {
                context.assets.open("models/${descriptor.file}").use { sha256(it) }
            }.getOrNull()
            if (digest != descriptor.sha256.lowercase()) {
                missing += descriptor.id
            } else {
                ready += descriptor.id
            }
        }
        val versions = descriptors.filter { it.id in ready }.joinToString("+") { it.version }
        return if (missing.isEmpty()) {
            ModelRuntimeStatus(
                backend = InferenceBackend.ONNX_CPU,
                modelVersion = versions,
                readyModelIds = ready,
                missingModelIds = emptySet(),
                message = "离线模型文件已校验，正在加载",
            )
        } else {
            ModelRuntimeStatus(
                backend = InferenceBackend.OPENCV_FALLBACK,
                modelVersion = "opencv-fallback-1",
                readyModelIds = ready,
                missingModelIds = missing,
                message = "部分模型文件不可用，正在检查可用功能",
            )
        }
    }

    @Synchronized
    fun copyVerifiedModelToCache(descriptor: ModelDescriptor): File {
        require(descriptor.sha256.length == 64)
        val target = File(context.codeCacheDir, descriptor.file)
        if (!target.exists() || target.inputStream().use { sha256(it) } != descriptor.sha256.lowercase()) {
            val temporary = File(target.parentFile, "${target.name}.tmp")
            try {
                context.assets.open("models/${descriptor.file}").use { input ->
                    FileOutputStream(temporary).use { output -> input.copyTo(output); output.fd.sync() }
                }
                require(temporary.inputStream().use { sha256(it) } == descriptor.sha256.lowercase()) {
                    "Model digest mismatch for ${descriptor.id}"
                }
                check(temporary.renameTo(target)) { "Unable to install model ${descriptor.id}" }
            } finally {
                temporary.delete()
            }
        }
        return target
    }

    fun openInterpreter(modelId: String): Interpreter {
        val descriptor = descriptors().firstOrNull { it.id == modelId }
            ?: error("Unknown model: $modelId")
        val status = verify()
        require(modelId in status.readyModelIds) { "Model is not verified: $modelId" }
        return Interpreter(
            copyVerifiedModelToCache(descriptor),
            Interpreter.Options()
                .setUseXNNPACK(true)
                .setNumThreads(minOf(4, Runtime.getRuntime().availableProcessors())),
        )
    }

    fun openOnnxSession(modelId: String): OrtSession {
        val descriptor = descriptors().firstOrNull { it.id == modelId }
            ?: error("Unknown model: $modelId")
        require(descriptor.file.endsWith(".onnx"))
        val file = copyVerifiedModelToCache(descriptor)
        return OrtSession.SessionOptions().use { options ->
            // Explicit CPU execution is portable and numerically checked. Hardware delegates
            // need per-device validation before they can be selected automatically.
            options.setIntraOpNumThreads(minOf(4, Runtime.getRuntime().availableProcessors()))
            options.setInterOpNumThreads(1)
            OrtEnvironment.getEnvironment().createSession(file.absolutePath, options)
        }
    }

    private fun sha256(input: InputStream): String {
        val digest = MessageDigest.getInstance("SHA-256")
        val buffer = ByteArray(64 * 1024)
        while (true) {
            val count = input.read(buffer)
            if (count < 0) break
            digest.update(buffer, 0, count)
        }
        return digest.digest().joinToString("") { "%02x".format(it) }
    }

    companion object {
        private const val MANIFEST_PATH = "models/manifest.json"
    }
}
