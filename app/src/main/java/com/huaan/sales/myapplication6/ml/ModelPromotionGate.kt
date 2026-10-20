package com.huaan.sales.myapplication6.ml

/** A versioned independent report is required; a successful model load is not an accuracy test. */
data class ModelPromotionReport(
    val independentPhotos: Int = 0,
    val independent: Boolean = false,
    val cornerErrorReduction: Double = 0.0,
    val ordinaryErrorIncrease: Double = Double.POSITIVE_INFINITY,
    val failureRecall: Double = 0.0,
    val deviceP95Millis: Map<String, Double> = emptyMap(),
    val modelSha256: String = "",
    val reportId: String = "",
)

object ModelPromotionGate {
    val requiredDevices = setOf("snapdragon_8_elite", "dimensity_9400", "google_tensor")
    const val FASTVIT_SHA256 = "950070fd4a46e25d46db7c35d0ed77612d9063f70d4543c7c504f0da6a943a89"

    fun failures(report: ModelPromotionReport): List<String> = buildList {
        if (!report.independent || report.independentPhotos < 200) add("至少 200 张独立实拍盲测")
        if (!report.cornerErrorReduction.isFinite() || report.cornerErrorReduction < .15) add("角点误差至少降低 15%")
        if (!report.ordinaryErrorIncrease.isFinite() || report.ordinaryErrorIncrease > 0) add("普通场景不退化")
        if (!report.failureRecall.isFinite() || report.failureRecall < .95) add("失败召回率至少 95%")
        if (!requiredDevices.all { device ->
                report.deviceP95Millis[device]?.let { it.isFinite() && it > 0 && it <= 2000 } == true
            }) add("三类旗舰真机 P95 不超过 2 秒")
        if (report.modelSha256 != FASTVIT_SHA256 || report.reportId.isBlank()) add("固定权重和可追溯报告")
    }
}
