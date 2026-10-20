package com.huaan.sales.myapplication6.camera

import android.util.Size
import androidx.camera.core.AspectRatio
import androidx.camera.core.ImageCapture
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy

enum class FlashSetting(val label: String, val cameraXMode: Int) {
    ON("开启", ImageCapture.FLASH_MODE_ON),
    OFF("关闭", ImageCapture.FLASH_MODE_OFF),
    AUTO("智能", ImageCapture.FLASH_MODE_AUTO),
}

data class CaptureSize(
    val label: String,
    val width: Int? = null,
    val height: Int? = null,
) {
    fun toResolutionSelector(): ResolutionSelector {
        val targetWidth = width
        val targetHeight = height
        if (targetWidth == null || targetHeight == null) {
            // Regular still size for this camera: largest JPEG that does not drop
            // into a slow ultra-high-resolution mode. 4:3 matches typical sensors.
            return ResolutionSelector.Builder()
                .setAllowedResolutionMode(ResolutionSelector.PREFER_CAPTURE_RATE_OVER_HIGHER_RESOLUTION)
                .setAspectRatioStrategy(AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY)
                .setResolutionStrategy(ResolutionStrategy.HIGHEST_AVAILABLE_STRATEGY)
                .build()
        }
        val ratio = if (targetWidth * 9 == targetHeight * 16 || targetHeight * 9 == targetWidth * 16) {
            AspectRatio.RATIO_16_9
        } else {
            AspectRatio.RATIO_4_3
        }
        return ResolutionSelector.Builder()
            .setAspectRatioStrategy(
                AspectRatioStrategy(ratio, AspectRatioStrategy.FALLBACK_RULE_AUTO),
            )
            .setResolutionStrategy(
                ResolutionStrategy(
                    Size(targetWidth, targetHeight),
                    ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER,
                ),
            )
            .build()
    }

    companion object {
        val Default = CaptureSize("3264×2448", 3264, 2448)
        val options = listOf(
            Default,
            CaptureSize("4000×3000", 4000, 3000),
            CaptureSize("2592×1944", 2592, 1944),
            CaptureSize("1920×1080", 1920, 1080),
            CaptureSize("1280×720", 1280, 720),
        )
    }
}
