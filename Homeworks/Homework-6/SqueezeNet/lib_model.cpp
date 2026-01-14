#include "lib_model.h"
#include "mbed.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
// REMOVED: micro_log.h and any error reporting headers
#include "tensorflow/lite/schema/schema_generated.h"

// 16-byte alignment is required for TFLite Micro on STM32
alignas(16) static uint8_t tensor_arena[160 * 1024]; 

// Move these to global scope to keep them off the limited stack
static tflite::MicroMutableOpResolver<15> resolver;
static tflite::MicroInterpreter* interpreter = nullptr;

int InitModel(const unsigned char* model_data, uint8_t* opt_ptr, int opt_size) {
    const tflite::Model* model = tflite::GetModel(model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) return MODEL_ERROR;

    // Register only necessary operators for SqueezeNet
    resolver.AddConv2D();
    resolver.AddMaxPool2D();
    resolver.AddConcatenation();
    resolver.AddRelu();
    resolver.AddReshape();
    resolver.AddMean(); 
    resolver.AddSoftmax();
    resolver.AddQuantize();
    resolver.AddDequantize();

    // Static instance ensures allocation is in Data section, not on Stack
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, sizeof(tensor_arena)
    );
    interpreter = &static_interpreter;

    if (interpreter->AllocateTensors() != kTfLiteOk) return MODEL_ERROR;
    return MODEL_OK;
}

int RunInferenceInt8(int8_t* input_data, int8_t* output_data) {
    if (interpreter == nullptr) return MODEL_ERROR;
    
    TfLiteTensor* input = interpreter->input(0);
    TfLiteTensor* output = interpreter->output(0);

    // Use standard C memcpy (safe for bare-metal)
    memcpy(input->data.int8, input_data, input->bytes);

    if (interpreter->Invoke() != kTfLiteOk) return MODEL_ERROR;

    memcpy(output_data, output->data.int8, output->bytes);
    return MODEL_OK;
}