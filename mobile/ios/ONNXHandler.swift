import Foundation
import onnxruntime_objc

class ONNXHandler {
    private var session: ORTSession?
    private var env: ORTEnv
    
    init(modelName: String) throws {
        self.env = try ORTEnv(loggingLevel: .verbose)
        let modelPath = Bundle.main.path(forResource: modelName, ofType: "onnx")!
        self.session = try ORTSession(env: env, modelPath: modelPath, sessionOptions: nil)
    }
    
    func runInference(inputData: Data, dims: [NSNumber]) throws -> [Float] {
        let inputTensor = try ORTValue(tensorData: NSMutableData(data: inputData),
                                      elementType: .float,
                                      shape: dims)
        
        let output = try session?.run(withInputs: ["input": inputTensor],
                                     outputNames: ["output"],
                                     runOptions: nil)
        
        let outputValue = output?["output"]
        let outputData = try outputValue?.tensorData() as Data?
        
        // Convert Data to [Float]
        return outputData!.withUnsafeBytes {
            Array($0.bindMemory(to: Float.self))
        }
    }
}
