package com.huaan.sales.myapplication6

import com.huaan.sales.myapplication6.ml.DewarpGrid
import org.junit.Assert.*
import org.junit.Test

class DewarpGridTest {
    @Test fun usesAlignedCornersAndBilinearInterpolation() {
        val grid = DewarpGrid(floatArrayOf(-1f, 1f, -1f, 1f, -1f, -1f, 1f, 1f), 2, 2)
        grid.validate()
        assertEquals(0f, grid.sample(0f, 0f).x, 0f)
        assertEquals(1f, grid.sample(1f, 1f).y, 0f)
        assertEquals(0.25f, grid.sample(0.25f, 0.75f).x, 1e-6f)
        assertEquals(0.75f, grid.sample(0.25f, 0.75f).y, 1e-6f)
    }

    @Test fun rejectsFoldedOrNonFiniteGrid() {
        val folded = DewarpGrid(floatArrayOf(1f, -1f, 1f, -1f, -1f, -1f, 1f, 1f), 2, 2)
        assertThrows(IllegalArgumentException::class.java) { folded.validate() }
        val invalid = DewarpGrid(FloatArray(8) { Float.NaN }, 2, 2)
        assertThrows(IllegalArgumentException::class.java) { invalid.validate() }
    }
}
