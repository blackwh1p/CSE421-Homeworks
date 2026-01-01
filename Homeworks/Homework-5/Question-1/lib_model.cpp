#include "lib_model.h"

static tflite::MicroInterpreter* interpreter = nullptr;
static const tflite::Model* model = nullptr;

// CHANGE: Define a resolver with enough space for our specific ops (5 is safe)
static tflite::MicroMutableOpResolver<5> resolver; 

int8_t LIB_MODEL_Init(const unsigned char *model_data, TfLiteTensor **input, uint8_t *tensor_arena, int arena_size) {
    // 1. Load Model
    model = tflite::GetModel(model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        printf("Model schema version mismatch!\n");
        return -1;
    }

    // 2. Add ONLY the necessary operations for your HAR model
    // The HAR MLP model uses Dense layers (FullyConnected) and Activations.
    if (resolver.AddFullyConnected() != kTfLiteOk) return -1;
    if (resolver.AddSoftmax() != kTfLiteOk) return -1;
    if (resolver.AddRelu() != kTfLiteOk) return -1;
    // If your model uses quantization, you might need AddDequantize or AddQuantize, 
    // but for float models, the above are usually sufficient.

    // 3. Build Interpreter
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, arena_size, nullptr
    );
    interpreter = &static_interpreter;

    // 4. Allocate Tensors
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        printf("AllocateTensors failed.\n");
        return -1;
    }

    *input = interpreter->input(0);
    return 0;
}

int8_t LIB_MODEL_Run(TfLiteTensor **output) {
    if (interpreter->Invoke() != kTfLiteOk) {
        printf("Invoke failed.\n");
        return -1;
    }
    *output = interpreter->output(0);
    return 0;
}