import serial
import serial.tools.list_ports
import matplotlib.pyplot as plt
import time
import numpy as np

# ------------------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------------------
# Set this to your specific COM port (e.g., 'COM3' on Windows, '/dev/ttyACM0' on Linux)
# If set to None, the script will try to auto-detect the Tiva Board.
SERIAL_PORT = None 
BAUD_RATE = 115200
DURATION = 5.0  # Time in seconds to record data before plotting

def find_tiva_port():
    """Attempts to auto-detect the Tiva C Launchpad port."""
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        # Tiva boards often show up as "Stellaris" or "Tiva"
        if "Stellaris" in p.description or "Tiva" in p.description:
            return p.device
    
    # Fallback: Return the first available port if specific one not found
    if len(ports) > 0:
        return ports[0].device
    return None

def collect_data(port_name, duration_sec):
    """
    Reads data synchronously for a set duration.
    Protocol: [0xA5] [Low Byte] [High Byte]
    """
    captured_data = []
    start_time = time.time()
    average = []
    
    try:
        # Open serial port
        with serial.Serial(port_name, BAUD_RATE, timeout=0.1) as ser:
            print(f"Connected to {port_name}. Recording for {duration_sec} seconds...")
            ser.reset_input_buffer()
            
            # Loop until time is up
            i = 0
            temp_array = []
            while (time.time() - start_time) < duration_sec:
                i += 1
                if ser.in_waiting > 0:
                    # 1. Read one byte to check for Sync Header (0xA5)
                    header = ser.read(1)
                    
                    if header == b'\xA5':
                        # 2. We found the header, now read the next 2 bytes (Low, High)
                        data_bytes = ser.read(2)
                        
                        if len(data_bytes) == 2:
                            low_byte = data_bytes[0]
                            high_byte = data_bytes[1]
                            
                            # Reconstruct 12-bit ADC value
                            adc_raw = low_byte | (high_byte << 8)
                            
                            # Convert to Voltage (0 - 4095 -> 0V - 3.3V)
                            voltage = (adc_raw / 4095.0) * 3.3
                            temp_array.append(voltage)
                            if i % 80 == 0:
                                average.append(np.average(temp_array))
                                temp_array = []
                            
                            captured_data.append(voltage)
                            
    except serial.SerialException as e:
        print(f"Serial Error: {e}")
    except Exception as e:
        print(f"Error: {e}")
        
    return captured_data, average

def main():
    # 1. Detect Port
    target_port = SERIAL_PORT
    if target_port is None:
        target_port = find_tiva_port()
    
    if target_port is None:
        print("Error: No serial ports found. Connect the Tiva board.")
        return

    # 2. Collect Data
    print("Starting capture...")
    voltages, average = collect_data(target_port, DURATION)
    
    if not voltages:
        print("No data collected. Check connections/baud rate.")
        return

    print(f"Capture complete. Collected {len(voltages)} samples.")

    # 3. Plot Data
    plt.figure(figsize=(10, 6))
    # plt.plot(voltages, 'b-', linewidth=1)
    plt.plot(average, 'r-', linewidth=2, label='Average (every 100 samples)')
    plt.legend()
    
    plt.title(f"Recorded Voltage (ADC) - {target_port}")
    plt.ylabel("Voltage (V)")
    plt.xlabel("Sample Number")
    plt.ylim(-0.1, 3.5) # Set Y-axis limits (0 to 3.3V with padding)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    print("Displaying plot...")
    plt.show()

if __name__ == "__main__":
    main()