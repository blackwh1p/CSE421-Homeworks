import pandas as pd
import numpy as np

def read_data(file_path):
    print(f"Loading data from {file_path}...")
    column_names = ["user", "activity", "timestamp", "x-accel", "y-accel", "z-accel"]
    
    try:
        # 'on_bad_lines="skip"' tells pandas to ignore lines with errors (like line 134634)
        # Note: If you are using a very old version of pandas (<1.3), use 'error_bad_lines=False'
        df = pd.read_csv(file_path, header=None, names=column_names, on_bad_lines='skip')
        
        # Clean the z-accel column (remove trailing ';')
        df["z-accel"] = df["z-accel"].astype(str).str.replace(";", "", regex=False)
        
        # Convert z-accel to numeric, turning unparseable text into NaN (safe conversion)
        df["z-accel"] = pd.to_numeric(df["z-accel"], errors='coerce')
        
        # Drop rows with missing values
        df.dropna(inplace=True)
        
        print(f"Data loaded successfully.")
        print(f"Number of columns: {df.shape[1]}")
        print(f"Number of rows: {df.shape[0]}")
        return df

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"An error occurred loading data: {e}")
        return None