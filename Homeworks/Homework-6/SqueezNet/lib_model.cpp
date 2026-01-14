#include "lib_model.h"
#include "mbed.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_log.h" 
#include "tensorflow/lite/schema/schema_generated.h"

// 1. Move the Arena to the global section and ensure 16-byte alignment
// 80KB is safe for a 50KB model.
alignas(16) static uint8_t tensor_arena[80 * 1024]; 

// 2. Move the Resolver to global scope so it doesn't sit on the main stack
static tflite::MicroMutableOpResolver<15> resolver;
static tflite::MicroInterpreter* interpreter = nullptr;

int InitModel(const unsigned char* model_data, uint8_t* optional_arena_ptr, int optional_size) {
    // Check model validity
    const tflite::Model* model = tflite::GetModel(model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) return MODEL_ERROR;

    // 3. Register only necessary ops to save memory
    resolver.AddConv2D();
    resolver.AddMaxPool2D();
    resolver.AddAveragePool2D();
    resolver.AddConcatenation();
    resolver.AddRelu();
    resolver.AddReshape();
    resolver.AddMean(); 
    resolver.AddSoftmax();

    // 4. Use a 'static' interpreter instance inside InitModel.
    // This ensures it lives in the global data section, not on the stack.
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, sizeof(tensor_arena)
    );
    interpreter = &static_interpreter;

    // 5. This is where the HardFault usually happens if stack is too small
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        return MODEL_ERROR;
    }

    return MODEL_OK;
}

int RunInferenceInt8(int8_t* input_data, int8_t* output_data) {
    if (interpreter == nullptr) return MODEL_ERROR;
    
    // Use local pointers to avoid complex global access during Invoke
    TfLiteTensor* input = interpreter->input(0);
    TfLiteTensor* output = interpreter->output(0);

    // Copy input data directly into the tensor
    memcpy(input->data.int8, input_data, input->bytes);

    // 6. Execute inference
    if (interpreter->Invoke() != kTfLiteOk) {
        return MODEL_ERROR;
    }

    // Copy output data
    memcpy(output_data, output->data.int8, output->bytes);
    return MODEL_OK;
}