package com.prismstyle.ai

import android.content.Context
import ai.onnxruntime.*
import java.nio.FloatBuffer

class OnnxHandler(context: Context, modelResId: Int) {
    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession

    init {
        val modelBytes = context.resources.openRawResource(modelResId).readBytes()
        session = env.createSession(modelBytes)
    }

    fun runInference(inputData: FloatArray, dims: LongArray): FloatArray {
        val container = "input"
        val inputTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(inputData), dims)
        
        val results = session.run(mapOf(container to inputTensor))
        val output = results[0].value as Array<FloatArray>
        
        return output[0]
    }

    fun close() {
        session.close()
        env.close()
    }
}
