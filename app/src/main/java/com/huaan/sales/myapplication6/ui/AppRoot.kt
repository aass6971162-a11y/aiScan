package com.huaan.sales.myapplication6.ui

import androidx.compose.animation.core.tween
import androidx.compose.animation.slideInHorizontally
import androidx.compose.animation.slideOutHorizontally
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import com.huaan.sales.myapplication6.ScannerScreen
import com.huaan.sales.myapplication6.ScannerViewModel

private const val HomeRoute = "home"
private const val ScanDocumentRoute = "scan_document"

@Composable
fun AppRoot(viewModel: ScannerViewModel = viewModel()) {
    val navController = rememberNavController()
    val state by viewModel.state.collectAsState()

    LaunchedEffect(state.screen) {
        val onScanPage = navController.currentDestination?.route == ScanDocumentRoute
        if (state.screen != ScannerScreen.HOME && !onScanPage) {
            navController.navigate(ScanDocumentRoute) { launchSingleTop = true }
        }
    }

    NavHost(
        navController = navController,
        startDestination = HomeRoute,
        modifier = Modifier.fillMaxSize(),
        enterTransition = { slideInHorizontally(tween(280)) { it } },
        exitTransition = { slideOutHorizontally(tween(280)) { -it / 4 } },
        popEnterTransition = { slideInHorizontally(tween(280)) { -it / 4 } },
        popExitTransition = { slideOutHorizontally(tween(280)) { it } },
    ) {
        composable(HomeRoute) {
            HomeScreen(
                documents = state.documents,
                onScanDocument = viewModel::startNewScan,
                onOpenDocument = viewModel::openDocument,
                onDeleteDocument = viewModel::deleteDocument,
            )
        }
        composable(ScanDocumentRoute) {
            ScannerApp(
                viewModel = viewModel,
                onExit = {
                    viewModel.showHome()
                    navController.popBackStack()
                },
            )
        }
    }
}
