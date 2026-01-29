package com.prismstyle_ai.app

import android.os.Bundle
import android.util.Log
import io.flutter.embedding.android.FlutterFragmentActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel
import io.flutter.plugins.GeneratedPluginRegistrant
import com.prismstyle_ai.mobile.FashionAIHandler

class MainActivity: FlutterFragmentActivity() {
    private var fashionAIHandler: FashionAIHandler? = null
    
    companion object {
        private const val TAG = "PrismStyle_MainActivity"
        private const val CLIP_CHANNEL = "com.prismstyle_ai/clip_encoder"
        private const val ONNX_CHANNEL = "com.prismstyle_ai/onnx"
    }
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        Log.d(TAG, "MainActivity onCreate")
    }
    
    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)
        GeneratedPluginRegistrant.registerWith(flutterEngine)
        
        // Initialize FashionAIHandler
        fashionAIHandler = FashionAIHandler(this)
        Log.d(TAG, "✅ FashionAIHandler initialized")
        
        // Setup CLIP Encoder Method Channel
        setupCLIPChannel(flutterEngine)
        
        // Setup ONNX Method Channel for generic ONNX operations
        setupONNXChannel(flutterEngine)
        
        Log.d(TAG, "✅ All method channels configured")
    }
    
    private fun setupCLIPChannel(flutterEngine: FlutterEngine) {
        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, CLIP_CHANNEL)
            .setMethodCallHandler { call, result ->
                when (call.method) {
                    "initialize" -> {
                        val success = fashionAIHandler?.initialize() ?: false
                        Log.d(TAG, "CLIP initialize: $success")
                        result.success(success)
                    }
                    
                    "getEmbedding" -> {
                        val imageData = call.argument<ByteArray>("imageData")
                        if (imageData == null) {
                            result.error("INVALID_ARGS", "imageData required", null)
                            return@setMethodCallHandler
                        }
                        
                        val embedding = fashionAIHandler?.getEmbedding(imageData)
                        if (embedding != null) {
                            result.success(embedding.toList())
                        } else {
                            result.error("EMBEDDING_FAILED", "Failed to generate embedding", null)
                        }
                    }
                    
                    "classify" -> {
                        val imageData = call.argument<ByteArray>("imageData")
                        if (imageData == null) {
                            result.error("INVALID_ARGS", "imageData required", null)
                            return@setMethodCallHandler
                        }
                        
                        val classification = fashionAIHandler?.classify(imageData)
                        result.success(classification)
                    }
                    
                    "findSimilar" -> {
                        val imageData = call.argument<ByteArray>("imageData")
                        val topK = call.argument<Int>("topK") ?: 5
                        
                        if (imageData == null) {
                            result.error("INVALID_ARGS", "imageData required", null)
                            return@setMethodCallHandler
                        }
                        
                        val similar = fashionAIHandler?.findSimilar(imageData, topK)
                        val response = similar?.map { item ->
                            mapOf(
                                "path" to item.path,
                                "score" to item.score,
                                "metadata" to item.metadata
                            )
                        } ?: emptyList()
                        result.success(response)
                    }
                    
                    "addToIndex" -> {
                        val imageData = call.argument<ByteArray>("imageData")
                        val path = call.argument<String>("path")
                        val metadata = call.argument<Map<String, Any>>("metadata") ?: emptyMap()
                        
                        if (imageData == null || path == null) {
                            result.error("INVALID_ARGS", "imageData and path required", null)
                            return@setMethodCallHandler
                        }
                        
                        val success = fashionAIHandler?.addToWardrobeIndex(imageData, path, metadata) ?: false
                        result.success(success)
                    }
                    
                    "recommendOutfit" -> {
                        val category = call.argument<String>("category") ?: ""
                        val occasion = call.argument<String>("occasion") ?: ""
                        val weather = call.argument<String>("weather") ?: ""
                        
                        val recommendations = fashionAIHandler?.recommendOutfit(category, occasion, weather)
                        result.success(recommendations)
                    }
                    
                    "loadWardrobeIndex" -> {
                        val indexPath = call.argument<String>("indexPath")
                        if (indexPath == null) {
                            result.error("INVALID_ARGS", "indexPath required", null)
                            return@setMethodCallHandler
                        }
                        
                        val success = fashionAIHandler?.loadWardrobeIndex(indexPath) ?: false
                        result.success(success)
                    }
                    
                    "saveWardrobeIndex" -> {
                        val indexPath = call.argument<String>("indexPath")
                        if (indexPath == null) {
                            result.error("INVALID_ARGS", "indexPath required", null)
                            return@setMethodCallHandler
                        }
                        
                        val success = fashionAIHandler?.saveWardrobeIndex(indexPath) ?: false
                        result.success(success)
                    }
                    
                    "getStatus" -> {
                        val isInitialized = fashionAIHandler?.isInitialized() ?: false
                        val wardrobeSize = fashionAIHandler?.getWardrobeSize() ?: 0
                        
                        result.success(mapOf(
                            "initialized" to isInitialized,
                            "wardrobeSize" to wardrobeSize,
                            "modelLoaded" to isInitialized
                        ))
                    }
                    
                    else -> result.notImplemented()
                }
            }
        
        Log.d(TAG, "✅ CLIP Encoder MethodChannel registered")
    }
    
    private fun setupONNXChannel(flutterEngine: FlutterEngine) {
        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, ONNX_CHANNEL)
            .setMethodCallHandler { call, result ->
                when (call.method) {
                    "loadModel" -> {
                        val modelPath = call.argument<String>("modelPath")
                        if (modelPath == null) {
                            result.error("INVALID_ARGS", "modelPath required", null)
                            return@setMethodCallHandler
                        }
                        
                        val success = fashionAIHandler?.loadModel(modelPath) ?: false
                        result.success(success)
                    }
                    
                    "runInference" -> {
                        val imageData = call.argument<ByteArray>("imageData")
                        if (imageData == null) {
                            result.error("INVALID_ARGS", "imageData required", null)
                            return@setMethodCallHandler
                        }
                        
                        val output = fashionAIHandler?.runInference(imageData)
                        result.success(output?.toList())
                    }
                    
                    else -> result.notImplemented()
                }
            }
        
        Log.d(TAG, "✅ ONNX MethodChannel registered")
    }
    
    override fun onDestroy() {
        fashionAIHandler?.cleanup()
        fashionAIHandler = null
        super.onDestroy()
        Log.d(TAG, "MainActivity destroyed, resources cleaned up")
    }
}