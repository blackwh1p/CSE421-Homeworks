#include "lib_model.h"
#include "mbed.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_log.h" 
#include "tensorflow/lite/schema/schema_generated.h"

// Reduced Arena for Int8 (120KB is usually enough for 32x32 Int8 EfficientNet)
const int kTensorArenaSize = 120 * 1024; 
static uint8_t tensor_arena[kTensorArenaSize] __attribute__((aligned(16)));

static tflite::MicroMutableOpResolver<50> resolver;
static tflite::MicroInterpreter* interpreter = nullptr;
static TfLiteTensor* input_tensor = nullptr;
static TfLiteTensor* output_tensor = nullptr;

int InitModel(const unsigned char* model_data, uint8_t* optional_arena_ptr, int optional_size) {
    const tflite::Model* model = tflite::GetModel(model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        MicroPrintf("Schema version mismatch!");
        return MODEL_ERROR;
    }

    // Register Ops
    resolver.AddConv2D();
    resolver.AddDepthwiseConv2D();
    resolver.AddAveragePool2D(); 
    resolver.AddFullyConnected();
    resolver.AddSoftmax();
    resolver.AddRelu6();
    resolver.AddAdd();
    resolver.AddMul();      
    resolver.AddLogistic(); 
    resolver.AddReshape();
    resolver.AddMinimum();  
    resolver.AddMaximum();  

    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kTensorArenaSize
    );
    interpreter = &static_interpreter;

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        MicroPrintf("AllocateTensors FAILED!");
        return MODEL_ERROR;
    }

    input_tensor = interpreter->input(0);
    output_tensor = interpreter->output(0);
    return MODEL_OK;
}

int RunInferenceInt8(int8_t* input_data, int8_t* output_data) {
    if (interpreter == nullptr) return MODEL_ERROR;

    // Direct Int8 Copy
    if (input_tensor->type == kTfLiteInt8) {
        memcpy(input_tensor->data.int8, input_data, input_tensor->bytes);
    } else {
        MicroPrintf("Type mismatch! Expected Int8.");
        return MODEL_ERROR;
    }

    if (interpreter->Invoke() != kTfLiteOk) {
        MicroPrintf("Invoke FAILED!");
        return MODEL_ERROR;
    }

    memcpy(output_data, output_tensor->data.int8, output_tensor->bytes);
    return MODEL_OK;
}