import os
from sklearn2c import LinearRegressor

# Detect current directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load the model we just trained with the new main.py
model_path = os.path.join(script_dir, "../temperature_pred_linreg.joblib")
export_path = os.path.join(script_dir, "../linear_reg_config")

print(f"Exporting model from: {model_path}")

try:
    # Load the model (Now it will be the correct sklearn2c type)
    linear_regressor = LinearRegressor.load(model_path)
    # Export to .h and .c
    linear_regressor.export(export_path)
    
    print("\nSUCCESS!")
    print(f"Files generated in: {script_dir}")
    print("- linear_reg_config.h")
    print("- linear_reg_config.c")
    
    # Print the values for your reference
    print("\nCheck your linear_reg_config.h file.")
    print("You will need the COEFFS and OFFSET values for Mbed.")
    
except Exception as e:
    print(f"ERROR: {e}")