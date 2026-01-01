/*
 * lib_model.cpp
 * Mbed Implementation using MicroMutableOpResolver
 */
#include "lib_model.h"
#include "mbed.h"

// We use MicroMutableOpResolver instead of AllOpsResolver to save memory
// and avoid missing header errors.

// Reserve space for 6 operations. If your model uses more layers (like Conv2D), increase this number.
static tflite::MicroMutableOpResolver<6> micro_op_resolver; 
static tflite::MicroInterpreter* interpreter;
static const tflite::Model* model = nullptr;

int8_t LIB_MODEL_Init(const void *tfliteModel, TfLiteTensor **inputTensor, uint8_t *buffer, uint32_t bufferSize)
{
    model = tflite::GetModel(tfliteModel);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        printf("Model schema version mismatch!\r\n");
        return MODEL_ERROR;
    }

    // --- MANUALLY ADD OPERATIONS ---
    // These are the standard layers used in the 'kws_mlp' model from the book.
    // If you get an error that an operation is missing during runtime, add it here.
    
    // 1. Fully Connected (Dense) Layer
    if (micro_op_resolver.AddFullyConnected() != kTfLiteOk) return MODEL_ERROR;
    
    // 2. Relu Activation
    if (micro_op_resolver.AddRelu() != kTfLiteOk) return MODEL_ERROR;
    
    // 3. Softmax (Final Output Layer)
    if (micro_op_resolver.AddSoftmax() != kTfLiteOk) return MODEL_ERROR;
    
    // 4. Reshape (Often used to flatten input)
    if (micro_op_resolver.AddReshape() != kTfLiteOk) return MODEL_ERROR;

    // Initialize Interpreter
    static tflite::MicroInterpreter static_interpreter(
        model, micro_op_resolver, buffer, bufferSize);
    
    interpreter = &static_interpreter;

    // Allocate memory from the tensor_arena
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        printf("AllocateTensors failed.\r\n");
        return MODEL_ERROR;
    }

    *inputTensor = interpreter->input(0);
    return MODEL_OK;
}

int8_t LIB_MODEL_Run(TfLiteTensor **output)
{
    if (interpreter->Invoke() != kTfLiteOk) {
        printf("Invoke failed.\r\n");
        return MODEL_ERROR;
    }
    *output = interpreter->output(0);
    return MODEL_OK;
}