package com.prismstyle_ai.mobile

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import ai.onnxruntime.*
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.io.ObjectInputStream
import java.io.ObjectOutputStream
import java.nio.FloatBuffer
import kotlin.math.sqrt

/**
 * Fashion AI Handler for Android
 * Provides:
 * 1. CLIP image encoding for embeddings (outfit matching)
 * 2. Clothing classification (category detection)
 * 3. Similarity search with local index
 */
class FashionAIHandler(private val context: Context) {
    
    companion object {
        private const val TAG = "FashionAIHandler"
        
        // CLIP model parameters
        private const val CLIP_INPUT_SIZE = 224
        private const val CLIP_EMBEDDING_DIM = 512
        
        // CLIP normalization (OpenCLIP ViT-B-32)
        private val CLIP_MEAN = floatArrayOf(0.48145466f, 0.4578275f, 0.40821073f)
        private val CLIP_STD = floatArrayOf(0.26862954f, 0.26130258f, 0.27577711f)
        
        // DeepFashion2 categories
        val CATEGORIES = arrayOf(
            "short_sleeve_top", "long_sleeve_top", "short_sleeve_outwear",
            "long_sleeve_outwear", "vest", "sling", "shorts", "trousers",
            "skirt", "short_sleeve_dress", "long_sleeve_dress", "vest_dress", "sling_dress"
        )
        
        // Category to app category mapping
        val CATEGORY_MAPPING = mapOf(
            "short_sleeve_top" to "Tops",
            "long_sleeve_top" to "Tops",
            "short_sleeve_outwear" to "Outerwear",
            "long_sleeve_outwear" to "Outerwear",
            "vest" to "Tops",
            "sling" to "Tops",
            "shorts" to "Bottoms",
            "trousers" to "Bottoms",
            "skirt" to "Bottoms",
            "short_sleeve_dress" to "Dresses",
            "long_sleeve_dress" to "Dresses",
            "vest_dress" to "Dresses",
            "sling_dress" to "Dresses"
        )
    }
    
    data class SimilarItem(
        val path: String,
        val score: Float,
        val metadata: Map<String, Any>
    )
    
    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private var clipSession: OrtSession? = null
    private var classifierSession: OrtSession? = null
    private var customSession: OrtSession? = null
    private var _isInitialized = false
    
    // Wardrobe index for similarity search
    private val wardrobeEmbeddings = mutableListOf<FloatArray>()
    private val wardrobePaths = mutableListOf<String>()
    private val wardrobeMetadata = mutableListOf<Map<String, Any>>()
    
    /**
     * Check if initialized
     */
    fun isInitialized(): Boolean = _isInitialized
    
    /**
     * Get wardrobe size
     */
    fun getWardrobeSize(): Int = wardrobePaths.size
    
    /**
     * Copy asset file to internal storage (needed for ONNX models with external data)
     * Flutter assets are located at flutter_assets/assets/models/
     */
    private fun copyAssetToFile(assetName: String, outputFileName: String): File? {
        return try {
            val outputFile = File(context.filesDir, outputFileName)
            if (outputFile.exists()) {
                android.util.Log.d(TAG, "Model already copied: $outputFileName")
                return outputFile
            }
            
            // Flutter assets path: flutter_assets/assets/models/
            context.assets.open("flutter_assets/assets/models/$assetName").use { input ->
                FileOutputStream(outputFile).use { output ->
                    input.copyTo(output)
                }
            }
            android.util.Log.d(TAG, "Copied asset to: ${outputFile.absolutePath}")
            outputFile
        } catch (e: Exception) {
            android.util.Log.w(TAG, "Failed to copy asset $assetName: ${e.message}")
            null
        }
    }
    
    /**
     * Initialize the AI models
     * Models with external data (.onnx.data) are copied to internal storage
     */
    fun initialize(): Boolean {
        try {
            // Copy CLIP encoder and its data file to internal storage
            val clipModelFile = copyAssetToFile("clip_image_encoder.onnx", "clip_image_encoder.onnx")
            val clipDataFile = copyAssetToFile("clip_image_encoder.onnx.data", "clip_image_encoder.onnx.data")
            
            if (clipModelFile != null) {
                val sessionOptions = OrtSession.SessionOptions()
                sessionOptions.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
                clipSession = env.createSession(clipModelFile.absolutePath, sessionOptions)
                android.util.Log.d(TAG, "✅ CLIP encoder loaded from: ${clipModelFile.absolutePath}")
            }
            
            // Copy classifier and its data file to internal storage
            val classifierModelFile = copyAssetToFile("clothing_classifier.onnx", "clothing_classifier.onnx")
            val classifierDataFile = copyAssetToFile("clothing_classifier.onnx.data", "clothing_classifier.onnx.data")
            
            if (classifierModelFile != null) {
                val sessionOptions = OrtSession.SessionOptions()
                sessionOptions.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
                classifierSession = env.createSession(classifierModelFile.absolutePath, sessionOptions)
                android.util.Log.d(TAG, "✅ Clothing classifier loaded from: ${classifierModelFile.absolutePath}")
            }
            
            _isInitialized = clipSession != null
            return _isInitialized
            
        } catch (e: Exception) {
            android.util.Log.e(TAG, "Failed to initialize FashionAI: ${e.message}")
            e.printStackTrace()
            return false
        }
    }
    
    /**
     * Load model from assets folder (for models without external data)
     * Flutter assets path: flutter_assets/assets/models/
     */
    private fun loadModelFromAssets(filename: String): ByteArray? {
        return try {
            context.assets.open("flutter_assets/assets/models/$filename").readBytes()
        } catch (e: Exception) {
            android.util.Log.w(TAG, "Model not found in assets: $filename")
            null
        }
    }
    
    /**
     * Load a custom ONNX model from path
     */
    fun loadModel(modelPath: String): Boolean {
        return try {
            val file = File(modelPath)
            if (!file.exists()) {
                android.util.Log.e(TAG, "Model file not found: $modelPath")
                return false
            }
            
            val sessionOptions = OrtSession.SessionOptions()
            sessionOptions.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            customSession = env.createSession(file.absolutePath, sessionOptions)
            android.util.Log.d(TAG, "✅ Custom model loaded: $modelPath")
            true
        } catch (e: Exception) {
            android.util.Log.e(TAG, "Failed to load model: ${e.message}")
            false
        }
    }
    
    /**
     * Run inference on custom model
     */
    fun runInference(imageData: ByteArray): FloatArray? {
        val session = customSession ?: clipSession ?: return null
        
        val bitmap = BitmapFactory.decodeByteArray(imageData, 0, imageData.size) ?: return null
        
        try {
            val inputData = preprocessForClip(bitmap)
            val dims = longArrayOf(1, 3, CLIP_INPUT_SIZE.toLong(), CLIP_INPUT_SIZE.toLong())
            
            val inputTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(inputData), dims)
            
            // Get input name
            val inputName = session.inputNames.firstOrNull() ?: "image"
            val results = session.run(mapOf(inputName to inputTensor))
            
            val output = results[0].value
            
            return when (output) {
                is Array<*> -> (output as Array<FloatArray>)[0]
                is FloatArray -> output
                else -> null
            }
            
        } catch (e: Exception) {
            android.util.Log.e(TAG, "Inference failed: ${e.message}")
            return null
        }
    }
    
    /**
     * Preprocess image for CLIP encoding
     */
    private fun preprocessForClip(bitmap: Bitmap): FloatArray {
        val resized = Bitmap.createScaledBitmap(bitmap, CLIP_INPUT_SIZE, CLIP_INPUT_SIZE, true)
        val pixels = IntArray(CLIP_INPUT_SIZE * CLIP_INPUT_SIZE)
        resized.getPixels(pixels, 0, CLIP_INPUT_SIZE, 0, 0, CLIP_INPUT_SIZE, CLIP_INPUT_SIZE)
        
        // CHW format with normalization
        val result = FloatArray(3 * CLIP_INPUT_SIZE * CLIP_INPUT_SIZE)
        
        for (i in pixels.indices) {
            val r = ((pixels[i] shr 16) and 0xFF) / 255.0f
            val g = ((pixels[i] shr 8) and 0xFF) / 255.0f
            val b = (pixels[i] and 0xFF) / 255.0f
            
            // RGB channels in CHW format with CLIP normalization
            result[i] = (r - CLIP_MEAN[0]) / CLIP_STD[0]  // R channel
            result[CLIP_INPUT_SIZE * CLIP_INPUT_SIZE + i] = (g - CLIP_MEAN[1]) / CLIP_STD[1]  // G
            result[2 * CLIP_INPUT_SIZE * CLIP_INPUT_SIZE + i] = (b - CLIP_MEAN[2]) / CLIP_STD[2]  // B
        }
        
        return result
    }
    
    /**
     * Get CLIP embedding for an image
     */
    fun getEmbedding(bitmap: Bitmap): FloatArray? {
        if (clipSession == null) {
            android.util.Log.e(TAG, "CLIP session not initialized")
            return null
        }
        
        try {
            val inputData = preprocessForClip(bitmap)
            val dims = longArrayOf(1, 3, CLIP_INPUT_SIZE.toLong(), CLIP_INPUT_SIZE.toLong())
            
            val inputTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(inputData), dims)
            val results = clipSession!!.run(mapOf("image" to inputTensor))
            
            val output = results[0].value as Array<FloatArray>
            val embedding = output[0]
            
            // L2 normalize
            val norm = sqrt(embedding.map { it * it }.sum())
            return if (norm > 0) {
                embedding.map { it / norm }.toFloatArray()
            } else {
                embedding
            }
            
        } catch (e: Exception) {
            android.util.Log.e(TAG, "Embedding failed: ${e.message}")
            return null
        }
    }
    
    /**
     * Get CLIP embedding from image bytes
     */
    fun getEmbedding(imageData: ByteArray): FloatArray? {
        val bitmap = BitmapFactory.decodeByteArray(imageData, 0, imageData.size)
        return bitmap?.let { getEmbedding(it) }
    }
    
    /**
     * Classify clothing category
     */
    fun classify(bitmap: Bitmap): Map<String, Float> {
        if (classifierSession == null) {
            // Fallback: use CLIP embedding similarity with category text embeddings
            return emptyMap()
        }
        
        try {
            val resized = Bitmap.createScaledBitmap(bitmap, 224, 224, true)
            val pixels = IntArray(224 * 224)
            resized.getPixels(pixels, 0, 224, 0, 0, 224, 224)
            
            // ImageNet normalization
            val imagenetMean = floatArrayOf(0.485f, 0.456f, 0.406f)
            val imagenetStd = floatArrayOf(0.229f, 0.224f, 0.225f)
            
            val inputData = FloatArray(3 * 224 * 224)
            for (i in pixels.indices) {
                val r = ((pixels[i] shr 16) and 0xFF) / 255.0f
                val g = ((pixels[i] shr 8) and 0xFF) / 255.0f
                val b = (pixels[i] and 0xFF) / 255.0f
                
                inputData[i] = (r - imagenetMean[0]) / imagenetStd[0]
                inputData[224 * 224 + i] = (g - imagenetMean[1]) / imagenetStd[1]
                inputData[2 * 224 * 224 + i] = (b - imagenetMean[2]) / imagenetStd[2]
            }
            
            val dims = longArrayOf(1, 3, 224, 224)
            val inputTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(inputData), dims)
            val results = classifierSession!!.run(mapOf("input" to inputTensor))
            
            val output = results[0].value as Array<FloatArray>
            val logits = output[0]
            
            // Softmax
            val maxLogit = logits.maxOrNull() ?: 0f
            val expLogits = logits.map { kotlin.math.exp((it - maxLogit).toDouble()).toFloat() }
            val sumExp = expLogits.sum()
            val probs = expLogits.map { it / sumExp }
            
            return CATEGORIES.indices.associate { CATEGORIES[it] to probs[it] }
            
        } catch (e: Exception) {
            android.util.Log.e(TAG, "Classification failed: ${e.message}")
            return emptyMap()
        }
    }
    
    /**
     * Classify from image bytes
     */
    fun classify(imageData: ByteArray): Map<String, Float> {
        val bitmap = BitmapFactory.decodeByteArray(imageData, 0, imageData.size)
        return bitmap?.let { classify(it) } ?: emptyMap()
    }
    
    /**
     * Add item to wardrobe index
     */
    fun addToWardrobeIndex(bitmap: Bitmap, path: String, metadata: Map<String, Any> = emptyMap()): Boolean {
        val embedding = getEmbedding(bitmap) ?: return false
        
        wardrobeEmbeddings.add(embedding)
        wardrobePaths.add(path)
        wardrobeMetadata.add(metadata)
        
        android.util.Log.d(TAG, "Added to wardrobe index: $path (total: ${wardrobePaths.size})")
        return true
    }
    
    /**
     * Add item from image bytes
     */
    fun addToWardrobeIndex(imageData: ByteArray, path: String, metadata: Map<String, Any> = emptyMap()): Boolean {
        val bitmap = BitmapFactory.decodeByteArray(imageData, 0, imageData.size)
        return bitmap?.let { addToWardrobeIndex(it, path, metadata) } ?: false
    }
    
    /**
     * Find similar items in wardrobe
     */
    fun findSimilar(queryEmbedding: FloatArray, topK: Int = 5): List<SimilarItem> {
        if (wardrobeEmbeddings.isEmpty()) {
            return emptyList()
        }
        
        // Compute cosine similarities
        val similarities = wardrobeEmbeddings.mapIndexed { idx, embedding ->
            val dotProduct = queryEmbedding.zip(embedding).sumOf { (a, b) -> (a * b).toDouble() }.toFloat()
            Triple(idx, wardrobePaths[idx], dotProduct)
        }
        
        // Sort by similarity (descending) and take top K
        return similarities
            .sortedByDescending { it.third }
            .take(topK)
            .map { (idx, path, score) ->
                SimilarItem(
                    path = path,
                    score = score,
                    metadata = wardrobeMetadata.getOrNull(idx) ?: emptyMap()
                )
            }
    }
    
    /**
     * Find similar items to a given image
     */
    fun findSimilar(imageData: ByteArray, topK: Int = 5): List<SimilarItem> {
        val bitmap = BitmapFactory.decodeByteArray(imageData, 0, imageData.size) ?: return emptyList()
        val embedding = getEmbedding(bitmap) ?: return emptyList()
        return findSimilar(embedding, topK)
    }
    
    /**
     * Recommend outfit based on category, occasion, and weather
     */
    fun recommendOutfit(category: String, occasion: String, weather: String): Map<String, Any> {
        val result = mutableMapOf<String, Any>()
        
        result["request"] = mapOf(
            "category" to category,
            "occasion" to occasion,
            "weather" to weather
        )
        
        if (wardrobeEmbeddings.isEmpty()) {
            result["error"] = "Wardrobe index is empty"
            return result
        }
        
        // Filter by category if metadata is available
        val filtered = wardrobeMetadata.mapIndexedNotNull { idx, meta ->
            val itemCategory = meta["category"] as? String ?: ""
            if (category.isEmpty() || itemCategory.contains(category, ignoreCase = true)) {
                Triple(idx, wardrobePaths[idx], wardrobeEmbeddings[idx])
            } else {
                null
            }
        }
        
        // Return random items from filtered list
        val recommendations = filtered.shuffled().take(5).map { (idx, path, _) ->
            mapOf(
                "path" to path,
                "metadata" to (wardrobeMetadata.getOrNull(idx) ?: emptyMap())
            )
        }
        
        result["recommendations"] = recommendations
        result["total_matches"] = filtered.size
        
        return result
    }
    
    /**
     * Load wardrobe index from file
     */
    fun loadWardrobeIndex(indexPath: String): Boolean {
        return try {
            val file = File(indexPath)
            if (!file.exists()) {
                android.util.Log.w(TAG, "Index file not found: $indexPath")
                return false
            }
            
            ObjectInputStream(FileInputStream(file)).use { ois ->
                @Suppress("UNCHECKED_CAST")
                val data = ois.readObject() as Map<String, Any>
                
                val embeddings = data["embeddings"] as? List<FloatArray> ?: emptyList()
                val paths = data["paths"] as? List<String> ?: emptyList()
                val metadata = data["metadata"] as? List<Map<String, Any>> ?: emptyList()
                
                wardrobeEmbeddings.clear()
                wardrobePaths.clear()
                wardrobeMetadata.clear()
                
                wardrobeEmbeddings.addAll(embeddings)
                wardrobePaths.addAll(paths)
                wardrobeMetadata.addAll(metadata)
                
                android.util.Log.d(TAG, "✅ Loaded wardrobe index: ${paths.size} items")
                true
            }
        } catch (e: Exception) {
            android.util.Log.e(TAG, "Failed to load wardrobe index: ${e.message}")
            false
        }
    }
    
    /**
     * Save wardrobe index to file
     */
    fun saveWardrobeIndex(indexPath: String): Boolean {
        return try {
            val file = File(indexPath)
            file.parentFile?.mkdirs()
            
            ObjectOutputStream(FileOutputStream(file)).use { oos ->
                val data = mapOf(
                    "embeddings" to wardrobeEmbeddings.toList(),
                    "paths" to wardrobePaths.toList(),
                    "metadata" to wardrobeMetadata.toList()
                )
                oos.writeObject(data)
            }
            
            android.util.Log.d(TAG, "✅ Saved wardrobe index: ${wardrobePaths.size} items")
            true
        } catch (e: Exception) {
            android.util.Log.e(TAG, "Failed to save wardrobe index: ${e.message}")
            false
        }
    }
    
    /**
     * Clean up resources
     */
    fun cleanup() {
        try {
            clipSession?.close()
            classifierSession?.close()
            customSession?.close()
            wardrobeEmbeddings.clear()
            wardrobePaths.clear()
            wardrobeMetadata.clear()
            _isInitialized = false
            android.util.Log.d(TAG, "Resources cleaned up")
        } catch (e: Exception) {
            android.util.Log.e(TAG, "Cleanup error: ${e.message}")
        }
    }
}
