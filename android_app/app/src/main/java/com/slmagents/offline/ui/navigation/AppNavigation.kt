package com.slmagents.offline.ui.navigation

import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.unit.dp
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.currentBackStackEntryAsState
import androidx.navigation.compose.rememberNavController
import com.slmagents.offline.ui.MainViewModel
import com.slmagents.offline.ui.screens.AgentsScreen
import com.slmagents.offline.ui.screens.ChatScreen
import com.slmagents.offline.ui.screens.KnowledgeBaseScreen
import com.slmagents.offline.ui.screens.ModelScreen
import com.slmagents.offline.ui.theme.PrimaryCyan

sealed class Screen(val route: String, val title: String, val icon: ImageVector) {
    object Chat : Screen("chat", "Chat", Icons.Default.ChatBubble)
    object Agents : Screen("agents", "Agents", Icons.Default.SmartToy)
    object Knowledge : Screen("knowledge", "Knowledge", Icons.Default.MenuBook)
    object Models : Screen("models", "Models", Icons.Default.Memory)
}

@Composable
fun AppNavigation(viewModel: MainViewModel) {
    val navController = rememberNavController()
    val items = listOf(Screen.Chat, Screen.Agents, Screen.Knowledge, Screen.Models)
    val navBackStackEntry by navController.currentBackStackEntryAsState()
    val currentRoute = navBackStackEntry?.destination?.route

    Scaffold(
        bottomBar = {
            NavigationBar(
                containerColor = MaterialTheme.colorScheme.surface,
                tonalElevation = 8.dp
            ) {
                items.forEach { screen ->
                    val isSelected = currentRoute == screen.route
                    NavigationBarItem(
                        icon = { Icon(screen.icon, contentDescription = screen.title) },
                        label = { Text(screen.title) },
                        selected = isSelected,
                        colors = NavigationBarItemDefaults.colors(
                            selectedIconColor = PrimaryCyan,
                            selectedTextColor = PrimaryCyan,
                            indicatorColor = MaterialTheme.colorScheme.surfaceVariant
                        ),
                        onClick = {
                            if (currentRoute != screen.route) {
                                navController.navigate(screen.route) {
                                    popUpTo(navController.graph.startDestinationId) {
                                        saveState = true
                                    }
                                    launchSingleTop = true
                                    restoreState = true
                                }
                            }
                        }
                    )
                }
            }
        }
    ) { innerPadding ->
        NavHost(
            navController = navController,
            startDestination = Screen.Chat.route,
            modifier = Modifier.padding(innerPadding)
        ) {
            composable(Screen.Chat.route) {
                ChatScreen(viewModel = viewModel)
            }
            composable(Screen.Agents.route) {
                AgentsScreen(viewModel = viewModel)
            }
            composable(Screen.Knowledge.route) {
                KnowledgeBaseScreen(viewModel = viewModel)
            }
            composable(Screen.Models.route) {
                ModelScreen(viewModel = viewModel)
            }
        }
    }
}
