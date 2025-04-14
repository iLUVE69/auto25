import serial
import time

# Set the correct COM port and baud rate
COM_PORT = 'COM4'  # Update this with your Arduino port
BAUD_RATE = 9600

def set_encoder_position(position):
    """
    Sends a command to set the encoder position on the Arduino.
    :param position: Integer value of the target encoder position.
    """
    try:
        # Open serial connection
        ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=2)
        time.sleep(2)  # Allow time for Arduino to reset

        # Send command
        command = f"SET: {position}\n"
        ser.write(command.encode('utf-8'))
        print(f" Command sent: {command.strip()}")

        # Wait for Arduino acknowledgment
        response = ser.readline().decode('utf-8').strip()
        print(f" Arduino Response: {response}")

        ser.close()
    except serial.SerialException as e:
        print(f" Serial error: {e}")
    except Exception as e:
        print(f" Unexpected error: {e}")

# Example: Set encoder position to -4500
set_encoder_position(-4500)
