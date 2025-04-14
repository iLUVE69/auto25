import serial
import re
import sys

# Ensure UTF-8 encoding to support emojis
sys.stdout.reconfigure(encoding='utf-8')

# Set the correct COM port and baud rate
COM_PORT = 'COM12'  # Update this with your Arduino port
BAUD_RATE = 115200

# Regular expression to extract LAT and LON from the received string
gps_pattern = re.compile(r".*LAT=\s*([-+]?\d+\.\d+)\s*LON=\s*([-+]?\d+\.\d+).*", re.IGNORECASE)

def read_gps_data():
    """
    Reads GPS data from Arduino and prints LAT, LON values.
    """
    try:
        ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=2)
        print(f"✅ Connected to {COM_PORT}. Waiting for GPS data...\n")

        while True:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            match = gps_pattern.search(line)

            if match:
                lat, lon = float(match.group(1)), float(match.group(2))
                print(f"📍 GPS Position - Latitude: {lat}, Longitude: {lon}")
    
    except serial.SerialException as e:
        print(f"❌ Serial error: {e}")
    except KeyboardInterrupt:
        print("\n🛑 Stopping GPS data reading.")
    finally:
        ser.close()
        print("🔌 Serial connection closed.")

# Run the GPS reader
read_gps_data()
