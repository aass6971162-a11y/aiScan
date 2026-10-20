package com.huaan.sales.myapplication6.ml

import com.huaan.sales.myapplication6.domain.NormalizedPoint
import kotlin.math.abs
import kotlin.math.hypot
import kotlin.math.max

/** A crop or padded canvas expressed in coordinates of the original image. */
internal data class CornerSearchWindow(
    val left: Float = 0f,
    val top: Float = 0f,
    val width: Float = 1f,
    val height: Float = 1f,
) {
    fun toSource(point: NormalizedPoint) = NormalizedPoint(left + point.x * width, top + point.y * height)
}

internal object CornerSearchGeometry {
    fun area(points: List<NormalizedPoint>): Float = if (points.size != 4) 0f else
        abs(points.indices.sumOf { i ->
            val a = points[i]; val b = points[(i + 1) % 4]
            (a.x * b.y - b.x * a.y).toDouble()
        }).toFloat() / 2f

    fun valid(points: List<NormalizedPoint>): Boolean {
        if (points.size != 4 || points.any { !it.x.isFinite() || !it.y.isFinite() ||
                it.x !in -0.04f..1.04f || it.y !in -0.04f..1.04f }) return false
        val turns = points.indices.map { i ->
            val a = points[i]; val b = points[(i + 1) % 4]; val c = points[(i + 2) % 4]
            (b.x - a.x) * (c.y - b.y) - (b.y - a.y) * (c.x - b.x)
        }
        // Small, clear documents are legitimate; reject degeneracy rather than requiring 6% area.
        return turns.all { it > 0.000001f } && area(points) in 0.003f..1.08f &&
            points.indices.all { points[it].distanceTo(points[(it + 1) % 4]) > 0.015f }
    }

    fun touchesFrame(points: List<NormalizedPoint>, margin: Float = 0.025f) = points.any {
        it.x < margin || it.x > 1f - margin || it.y < margin || it.y > 1f - margin
    }

    fun agrees(first: List<NormalizedPoint>, second: List<NormalizedPoint>): Boolean {
        if (first.size != 4 || second.size != 4) return false
        val diagonal = hypot(first.maxOf { it.x } - first.minOf { it.x },
            first.maxOf { it.y } - first.minOf { it.y })
        val tolerance = max(0.015f, diagonal * 0.18f)
        return first.indices.all { first[it].distanceTo(second[it]) <= tolerance }
    }

    fun consensusAgrees(first: List<NormalizedPoint>, second: List<NormalizedPoint>): Boolean {
        if (first.size != 4 || second.size != 4) return false
        val tolerance = consensusTolerance(first)
        return first.indices.all { first[it].distanceTo(second[it]) <= tolerance }
    }

    fun consensusTolerance(points: List<NormalizedPoint>): Float = max(0.012f,
        hypot(points.maxOf { it.x } - points.minOf { it.x },
            points.maxOf { it.y } - points.minOf { it.y }) * 0.035f)
}
