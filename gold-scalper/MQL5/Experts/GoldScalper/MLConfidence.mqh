//+------------------------------------------------------------------+
//| MLConfidence.mqh — ONNX model inference inside MQ5              |
//|                                                                  |
//| HOW IT WORKS:                                                    |
//| 1. Train XGBoost/LightGBM in Python on tick data (offline)      |
//| 2. Export model to ONNX format                                   |
//| 3. Load ONNX in MQ5 using OnnxCreate/OnnxRun                    |
//| 4. Run inference in OnTick() — sub-millisecond                  |
//|                                                                  |
//| This gives you ML power WITHOUT Python running live!             |
//| The ONNX runtime is built into MT5 since build 3390+            |
//+------------------------------------------------------------------+
#ifndef ML_CONFIDENCE_MQH
#define ML_CONFIDENCE_MQH

class CMLConfidence {
private:
    long     m_session;          // ONNX session handle
    bool     m_loaded;           // Model loaded successfully
    int      m_feature_count;    // Number of input features
    string   m_model_path;       // Path to .onnx file
    double   m_last_confidence;  // Last inference result
    datetime m_last_inference;   // Timestamp of last inference
    int      m_inference_count;  // Total inferences run
    int      m_error_count;      // Total errors
    
    // Input/output tensors
    matrixf  m_input_data;
    vectorf  m_output_data;

public:
    CMLConfidence(string model_name = "GoldScalper.onnx", int features = 30) 
        : m_feature_count(features), m_model_path(model_name) {
        m_session = INVALID_HANDLE;
        m_loaded = false;
        m_last_confidence = 0.5;
        m_last_inference = 0;
        m_inference_count = 0;
        m_error_count = 0;
    }
    
    ~CMLConfidence() {
        if(m_session != INVALID_HANDLE) {
            OnnxRelease(m_session);
            Print("ONNX session released");
        }
    }
    
    bool Load() {
        if(m_loaded) return true;
        
        // Build full path — MQL5/Files/ is the default ONNX search location
        // The .onnx file should be placed in MQL5/Files/ or specify full path
        string path = m_model_path;
        
        // Try to create ONNX session
        m_session = OnnxCreate(path, ONNX_DEFAULT);
        
        if(m_session == INVALID_HANDLE) {
            // Try alternate path in Files directory
            path = "Files\\" + m_model_path;
            m_session = OnnxCreate(path, ONNX_DEFAULT);
        }
        
        if(m_session == INVALID_HANDLE) {
            Print("ONNX model load FAILED: ", m_model_path);
            Print("  Place the .onnx file in: <MT5_Data_Folder>/MQL5/Files/");
            Print("  Or disable ML in config: INP_USE_ML = false");
            m_loaded = false;
            return false;
        }
        
        // Prepare input/output shapes
        // Input: [1, feature_count] — single sample, N features
        // Output: [1, 1] or [1, 2] — probability or [prob_down, prob_up]
        ulong input_shape[] = {1, (ulong)m_feature_count};
        if(!OnnxSetInputShape(m_session, 0, input_shape)) {
            Print("ONNX input shape FAILED");
            OnnxRelease(m_session);
            m_session = INVALID_HANDLE;
            m_loaded = false;
            return false;
        }
        
        // Output shape — try [1, 1] first (single probability)
        ulong output_shape[] = {1, 1};
        if(!OnnxSetOutputShape(m_session, 0, output_shape)) {
            // Maybe [1, 2] output — try that
            ulong output_shape2[] = {1, 2};
            if(!OnnxSetOutputShape(m_session, 0, output_shape2)) {
                Print("ONNX output shape FAILED — trying with dynamic shape");
                // Some models have dynamic output, this may still work
            }
        }
        
        m_loaded = true;
        Print("ONNX model loaded: ", path, " (", m_feature_count, " features)");
        return true;
    }
    
    double Predict(double &features[]) {
        if(!m_loaded) return 0.5;
        
        if(ArraySize(features) != m_feature_count) {
            Print("Feature count mismatch: got ", ArraySize(features), 
                  " expected ", m_feature_count);
            return 0.5;
        }
        
        // Prepare input matrix [1, feature_count]
        m_input_data.Resize(1, m_feature_count);
        for(int i = 0; i < m_feature_count; i++)
            m_input_data[0][i] = (float)features[i];
        
        // Run ONNX inference
        m_output_data.Resize(1);
        
        if(!OnnxRun(m_session, ONNX_DEFAULT, m_input_data, m_output_data)) {
            m_error_count++;
            if(m_error_count <= 3) // Only print first few errors
                Print("ONNX inference error #", m_error_count);
            return 0.5;
        }
        
        // Parse output
        double confidence = 0.5;
        if(m_output_data.Size() >= 2) {
            // Binary classification output [prob_down, prob_up]
            // Use the probability of the positive class
            confidence = (double)m_output_data[1];
        } else if(m_output_data.Size() == 1) {
            // Single probability output
            confidence = (double)m_output_data[0];
        }
        
        // Clamp to [0, 1]
        confidence = MathMax(0.0, MathMin(1.0, confidence));
        
        m_last_confidence = confidence;
        m_last_inference = TimeCurrent();
        m_inference_count++;
        
        return confidence;
    }
    
    bool   IsLoaded()          const { return m_loaded; }
    double GetConfidence()     const { return m_last_confidence; }
    int    GetInferenceCount() const { return m_inference_count; }
    int    GetErrorCount()     const { return m_error_count; }
};

#endif
//+------------------------------------------------------------------+
