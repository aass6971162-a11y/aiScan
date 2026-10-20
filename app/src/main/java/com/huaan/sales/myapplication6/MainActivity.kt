package com.huaan.sales.myapplication6

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import com.huaan.sales.myapplication6.ui.AppRoot
import com.huaan.sales.myapplication6.ui.theme.MyApplication6Theme
import org.opencv.android.OpenCVLoader

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        check(OpenCVLoader.initLocal()) { "OpenCV initialization failed" }
        enableEdgeToEdge()
        setContent {
            MyApplication6Theme(dynamicColor = false) {
                AppRoot()
            }
        }
    }
}
