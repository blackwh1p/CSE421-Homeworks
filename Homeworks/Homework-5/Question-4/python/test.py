import serial
import struct
import time
import random

SERIAL_PORT = 'COM6' 
BAUD_RATE = 115200

def sync_with_board(ser):
    print("Waiting for board... (Press Reset on Board)")
    while True:
        if ser.in_waiting:
            try:
                line = ser.read(ser.in_waiting)
                if b"READY" in line:
                    print("✅ Board Found! Sending 'GO'...")
                    ser.write(b'G')
                    time.sleep(0.5) 
                    ser.reset_input_buffer()
                    return True
            except:
                pass
        time.sleep(0.1)

if __name__ == "__main__":
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=2)
        print(f"Connected to {SERIAL_PORT}")
    except:
        print("Check Port"); exit()

    sync_with_board(ser)

    print("-" * 50)
    print(f"{'Inputs':<35} | {'Pred':<6}")
    print("-" * 50)

    base_temps = [18.0, 18.2, 18.5, 18.9, 19.2]

    for i in range(20):
        # Generate Data
        inputs = [x + random.uniform(-0.5, 0.5) for x in base_temps]
        
        # --- FIX: Send Sync Byte '$' FIRST ---
        ser.write(b'$') 
        # Send 5 Floats
        ser.write(struct.pack('<5f', *inputs))
        
        # Read Result
        raw_data = ser.read(4)
        
        if len(raw_data) == 4:
            pred = struct.unpack('<f', raw_data)[0]
            print(f"{str([round(x,1) for x in inputs]):<35} | {pred:.2f}°C")
        else:
            print("❌ Timeout (Sync Error?)")
            
        time.sleep(0.1)

    ser.close()