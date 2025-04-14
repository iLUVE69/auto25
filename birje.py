import serial
import re
import time
import threading

# Serial configurations
GPS_COM_PORT = 'COM10'  # Update with your Arduino's GPS port
ENCODER_COM_PORT = 'COM4'  # Update with your Arduino's encoder port
GPS_BAUD_RATE = 115200  # Must match Serial.begin() in Arduino
ENCODER_BAUD_RATE = 9600  # Must match Serial.begin() in Arduino


# List of target GPS coordinates
TARGET_LOCATIONS = [
    (23.2112928, 72.6859224),  # Turn 1
    (23.211306, 72.686403),    # Turn 2
    (23.2133064, 72.6864152)     # Turn 3
]

# GPS pattern to extract latitude and longitude
gps_pattern = re.compile(r"LAT=(-?\d+\.\d+),LON=(-?\d+\.\d+)")

# Cooldown time before triggering at different corners
COOLDOWN_TIME = 10  # seconds

last_trigger_time = 0  # To track last encoder trigger time

# Track visited targets (so each triggers only once)
visited_targets = set()

# Minimum movement threshold (prevents multiple triggers if cart is slow)
MIN_MOVEMENT_THRESHOLD = 0.00002  # Approx. ~2 meters

# Previous GPS position
previous_lat, previous_lon = None, None


def set_encoder_position(position):
    """
    Sends a command to set the encoder position on the Arduino.
    :param position: Integer value of the target encoder position.
    """
    try:
        ser = serial.Serial(ENCODER_COM_PORT, ENCODER_BAUD_RATE, timeout=2)
        time.sleep(2)  # Allow time for Arduino to reset

        # Send command
        command = f"SET: {position}\n"
        ser.write(command.encode('utf-8'))
        print(f" Command sent: {command.strip()}")

        # Wait for Arduino acknowledgment
        response = ser.readline().decode('utf-8', errors='ignore').strip()
        print(f" Arduino Response: {response}")

        ser.close()
    except serial.SerialException as e:
        print(f" Serial error: {e}")
    except Exception as e:
        print(f" Unexpected error: {e}")


def is_near_target(lat, lon, threshold=0.000065):
    """
    Checks if the given latitude and longitude are near any target location.
    :param lat: Current latitude
    :param lon: Current longitude
    :param threshold: Threshold distance to consider as "near" (~7m)
    :return: Target coordinates if near a target, otherwise None
    """
    for target in TARGET_LOCATIONS:
        if abs(lat - target[0]) < threshold and abs(lon - target[1]) < threshold:
            return target  # Return the specific target location
    return None


def read_gps_data():
    """
    Reads GPS data from the serial port and triggers encoder only once per corner.
    """
    global last_trigger_time, previous_lat, previous_lon

    try:
        with serial.Serial(GPS_COM_PORT, GPS_BAUD_RATE, timeout=2) as ser:
            print(f"Connected to {GPS_COM_PORT}. Waiting for GPS data...\n")

            while True:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                match = gps_pattern.search(line)

                if match:
                    lat, lon = float(match.group(1)), float(match.group(2))
                    print(f"Latitude: {lat}, Longitude: {lon}")

                    # Ignore first GPS reading or minimal movement
                    if previous_lat is not None and abs(lat - previous_lat) < MIN_MOVEMENT_THRESHOLD and abs(lon - previous_lon) < MIN_MOVEMENT_THRESHOLD:
                        continue  # Ignore repeated readings if not moving significantly
                    
                    previous_lat, previous_lon = lat, lon  # Update last known position

                    # Check if we are near a new target
                    target = is_near_target(lat, lon)
                    if target and target not in visited_targets and (time.time() - last_trigger_time > COOLDOWN_TIME):
                        visited_targets.add(target)  # Mark as visitedf
                        last_trigger_time = time.time()  # Update last trigger time
                        print(f" Target GPS location reached at {target}! Triggering encoder...")

                        # Run encoder command in a separate thread to avoid blocking GPS updates
                        threading.Thread(target=set_encoder_position, args=(-4500,)).start()
                        time.sleep(8)  # Wait for 4 seconds
                        threading.Thread(target=set_encoder_position, args=(4500,)).start()

    
    except serial.SerialException as e:
        print(f"Serial error: {e}")
    except KeyboardInterrupt:
        print("\nStopping GPS data reading.")
    finally:
        print("Serial connection closed.")


# Run the GPS reader
if __name__ == "__main__":
    read_gps_data()
