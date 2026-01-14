#include "lib_model.h"
#include "mbed.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Arena Size: 120 KB (MobileNet V1 Alpha 0.25 Float32 için ideal)
const int kTensorArenaSize = 120 * 1024; 
static uint8_t tensor_arena[kTensorArenaSize] __attribute__((aligned(16)));

static tflite::MicroMutableOpResolver<40> resolver;
static tflite::MicroInterpreter* interpreter = nullptr;
static TfLiteTensor* input_tensor = nullptr;
static TfLiteTensor* output_tensor = nullptr;
static const tflite::Model* model = nullptr;

int InitModel(const unsigned char* model_data, uint8_t* optional_arena_ptr, int optional_size) {
    // printf("[DEBUG] InitModel: Started.\n"); // KAPALI

    model = tflite::GetModel(model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        return MODEL_ERROR;
    }

    // MobileNet V1 Operators
    resolver.AddConv2D();
    resolver.AddDepthwiseConv2D();
    resolver.AddMean(); 
    resolver.AddAveragePool2D();
    resolver.AddGlobalAveragePool2D();
    resolver.AddFullyConnected();
    resolver.AddSoftmax();
    resolver.AddRelu();
    resolver.AddReshape();
    resolver.AddPad();
    resolver.AddAdd();
    resolver.AddMul();
    resolver.AddConcatenation();
    
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kTensorArenaSize
    );
    interpreter = &static_interpreter;

    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    
    if (allocate_status != kTfLiteOk) {
        // printf("[DEBUG] AllocateTensors FAILED! Code: %d\n", allocate_status); // KAPALI
        return MODEL_ERROR;
    }

    input_tensor = interpreter->input(0);
    output_tensor = interpreter->output(0);

    return MODEL_OK;
}

int RunInference(float* input_data, float* output_data) {
    if (interpreter == nullptr) return MODEL_ERROR;

    // Direct Copy (Float -> Float)
    if (input_tensor->type == kTfLiteFloat32) {
        for (int i = 0; i < input_tensor->bytes / sizeof(float); i++) {
            input_tensor->data.f[i] = input_data[i];
        }
    } else {
        return MODEL_ERROR; 
    }

    if (interpreter->Invoke() != kTfLiteOk) {
        return MODEL_ERROR;
    }

    if (output_tensor->type == kTfLiteFloat32) {
        for (int i = 0; i < output_tensor->bytes / sizeof(float); i++) {
            output_data[i] = output_tensor->data.f[i];
        }
    }

    return MODEL_OK;
}