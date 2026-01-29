package com.prismstyle.ai

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import ai.onnxruntime.*
import java.nio.FloatBuffer
import kotlin.math.sqrt

/**
 * Fashion AI Handler for Android
 * Provides:
 * 1. CLIP image encoding for embeddings (outfit matching)
 * 2. Clothing classification (category detection)
 * 3. Similarity search with FAISS index
 */
class FashionAIHandler(private val context: Context) {
    
    companion object {
        // CLIP model parameters
        private const val CLIP_INPUT_SIZE = 224
        private const val CLIP_EMBEDDING_DIM = 512
        
        // CLIP normalization (OpenCLIP ViT-B-32)
        private val CLIP_MEAN = floatArrayOf(0.48145466f, 0.4578275f, 0.40821073f)
        private val CLIP_STD = floatArrayOf(0.26862954f, 0.26130258f, 0.27577711f)
        
        // Classifier normalization (ImageNet)
        private val CLASSIFIER_MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
        private val CLASSIFIER_STD = floatArrayOf(0.229f, 0.224f, 0.225f)
        
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
    
    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private var clipSession: OrtSession? = null
    private var classifierSession: OrtSession? = null
    private var isInitialized = false
    
    // Wardrobe embeddings for similarity search
    private var wardrobeEmbeddings: Array<FloatArray>? = null
    private var wardrobePaths: Array<String>? = null
    
    /**
     * Initialize the AI models
     */
    fun initialize(): Boolean {
        try {
            // Load CLIP encoder
            val clipBytes = loadModelFromAssets("clip_image_encoder.onnx")
            if (clipBytes != null) {
                val sessionOptions = OrtSession.SessionOptions()
                sessionOptions.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
                clipSession = env.createSession(clipBytes, sessionOptions)
                println("✅ CLIP encoder loaded")
            }
            
            // Load classifier
            val classifierBytes = loadModelFromAssets("clothing_classifier.onnx")
            if (classifierBytes != null) {
                val sessionOptions = OrtSession.SessionOptions()
                sessionOptions.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
                classifierSession = env.createSession(classifierBytes, sessionOptions)
                println("✅ Clothing classifier loaded")
            }
            
            isInitialized = clipSession != null || classifierSession != null
            return isInitialized
            
        } catch (e: Exception) {
            println("❌ Failed to initialize FashionAI: ${e.message}")
            return false
        }
    }
    
    /**
     * Load model from assets folder
     */
    private fun loadModelFromAssets(filename: String): ByteArray? {
        return try {
            context.assets.open("models/$filename").readBytes()
        } catch (e: Exception) {
            println("⚠️ Model not found: $filename")
            null
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
            println("❌ CLIP session not initialized")
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
            return embedding.map { it / norm }.toFloatArray()
            
        } catch (e: Exception) {
            println("❌ Embedding failed: ${e.message}")
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
            return emptyMap()
        }
        
        try {
            // Preprocess with ImageNet normalization
            val resized = Bitmap.createScaledBitmap(bitmap, 224, 224, true)
            val pixels = IntArray(224 * 224)
            resized.getPixels(pixels, 0, 224, 0, 0, 224, 224)
            
            val inputData = FloatArray(3 * 224 * 224)
            for (i in pixels.indices) {
                val r = ((pixels[i] shr 16) and 0xFF) / 255.0f
                val g = ((pixels[i] shr 8) and 0xFF) / 255.0f
                val b = (pixels[i] and 0xFF) / 255.0f
                
                inputData[i] = (r - CLASSIFIER_MEAN[0]) / CLASSIFIER_STD[0]
                inputData[224 * 224 + i] = (g - CLASSIFIER_MEAN[1]) / CLASSIFIER_STD[1]
                inputData[2 * 224 * 224 + i] = (b - CLASSIFIER_MEAN[2]) / CLASSIFIER_STD[2]
            }
            
            val dims = longArrayOf(1, 3, 224, 224)
            val inputTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(inputData), dims)
            val results = classifierSession!!.run(mapOf("input" to inputTensor))
            
            val output = results[0].value as Array<FloatArray>
            val logits = output[0]
            
            // Softmax
            val maxLogit = logits.max()
            val expLogits = logits.map { kotlin.math.exp(it - maxLogit) }
            val sumExp = expLogits.sum()
            val probs = expLogits.map { (it / sumExp).toFloat() }
            
            // Map to categories
            return CATEGORIES.indices.associate { CATEGORIES[it] to probs[it] }
            
        } catch (e: Exception) {
            println("❌ Classification failed: ${e.message}")
            return emptyMap()
        }
    }
    
    /**
     * Load wardrobe embeddings for similarity search
     */
    fun loadWardrobeIndex(embeddings: Array<FloatArray>, paths: Array<String>) {
        wardrobeEmbeddings = embeddings
        wardrobePaths = paths
        println("✅ Loaded wardrobe index with ${paths.size} items")
    }
    
    /**
     * Find similar items in wardrobe
     */
    fun findSimilar(queryEmbedding: FloatArray, topK: Int = 5): List<Pair<String, Float>> {
        val embeddings = wardrobeEmbeddings ?: return emptyList()
        val paths = wardrobePaths ?: return emptyList()
        
        // Compute cosine similarities
        val similarities = embeddings.mapIndexed { idx, embedding ->
            val dotProduct = queryEmbedding.zip(embedding).map { (a, b) -> a * b }.sum()
            Pair(paths[idx], dotProduct)
        }
        
        // Sort by similarity (descending) and take top K
        return similarities.sortedByDescending { it.second }.take(topK)
    }
    
    /**
     * Find similar items to a given image
     */
    fun findSimilarTo(bitmap: Bitmap, topK: Int = 5): List<Pair<String, Float>> {
        val embedding = getEmbedding(bitmap) ?: return emptyList()
        return findSimilar(embedding, topK)
    }
    
    /**
     * Get outfit recommendation based on seed item
     */
    fun recommendOutfit(seedBitmap: Bitmap): Map<String, Any> {
        val result = mutableMapOf<String, Any>()
        
        // Get seed item classification
        val classification = classify(seedBitmap)
        val topCategory = classification.maxByOrNull { it.value }?.key ?: "unknown"
        val appCategory = CATEGORY_MAPPING[topCategory] ?: "Unknown"
        
        result["seed_category"] = topCategory
        result["seed_app_category"] = appCategory
        
        // Get embedding
        val embedding = getEmbedding(seedBitmap)
        if (embedding != null) {
            result["embedding"] = embedding.toList()
            
            // Find similar items (for complementary suggestions)
            val similar = findSimilar(embedding, 10)
            result["similar_items"] = similar.map { mapOf("path" to it.first, "score" to it.second) }
        }
        
        return result
    }
    
    /**
     * Clean up resources
     */
    fun close() {
        clipSession?.close()
        classifierSession?.close()
        env.close()
    }
}
