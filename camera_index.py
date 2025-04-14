import serial
import numpy as np
import cv2
import os
import threading
import time
from datetime import datetime


# Function to read encoder value from Arduino
def read_encoder_value(ser):
    try:
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8').strip()
            if line and (line.isdigit() or (line.startswith('-') and line[1:].isdigit())):
                return int(line)
    except Exception as e:
        print(f"Error reading encoder: {e}")
    return None


# Function to handle keyboard input for motor control
def keyboard_handler(ser, stop_event):
    import msvcrt  # For Windows
    print("Keyboard control active:")
    print("  'A' - Turn Left")
    print("  'D' - Turn Right")
    print("  'S' - Stop Motor")
    print("  'Q' - Quit Program")
    
    current_command = 'S'  # Start with motor stopped
    
    try:
        while not stop_event.is_set():
            if msvcrt.kbhit():
                key = msvcrt.getch().decode('utf-8', errors='ignore').upper()
                
                if key in ['A', 'D', 'S']:
                    if key != current_command:
                        current_command = key
                        ser.write(current_command.encode())
                        print(f"Sent command: {current_command}")
                elif key == 'Q':
                    stop_event.set()
                    print("Quitting program...")
            
    except Exception as e:
        print(f"Keyboard handler error: {e}")


# Configure the two webcams
webcam1 = cv2.VideoCapture(1)  # Using index 0 as requested
webcam2 = cv2.VideoCapture(2)  # Using index 2 as requested

# webcam2 = cv2.VideoCapture(2, cv2.CAP_DSHOW)  # Explicitly use DirectShow backend

# Check if webcams opened successfully
if not webcam1.isOpened():
    print(f"Failed to open webcam at index 0")
else:
    webcam1.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    webcam1.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print("Webcam 1 (index 0) opened successfully")

if not webcam2.isOpened():
    print(f"Failed to open webcam at index 2")
    # You could try releasing the camera and opening it again
    webcam2.release()
    webcam2 = cv2.VideoCapture(2, cv2.CAP_DSHOW)  # Try DirectShow backend
    
    if not webcam2.isOpened():
        print("Still couldn't open webcam at index 2")
else:
    webcam2.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    webcam2.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print("Webcam 2 (index 2) opened successfully")

# Configure the two webcams
# webcam1 = cv2.VideoCapture(0)  # Replace 0 with the correct device index if needed
# webcam2 = cv2.VideoCapture(2)  # Replace 1 with the correct device index if needed
# webcam1.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# webcam1.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
# webcam2.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# webcam2.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Create output directory for frames
output_dir = r"C:\C++\test"


# Open serial port for encoder readings
port_name = 'COM4'  # Replace with the correct port for your system
try:
    ser = serial.Serial(port_name, 9600, timeout=1)
    print(f"Connected to Arduino on {port_name}")
    
    # Set up keyboard handler thread
    stop_event = threading.Event()
    keyboard_thread = threading.Thread(target=keyboard_handler, args=(ser, stop_event))
    keyboard_thread.daemon = True
    keyboard_thread.start()
    
    frame_count = 0
    current_motor_state = "STOPPED"
    
    while not stop_event.is_set():
        # time.sleep(1)
          # Add this at the beginning of each loop iteration

        
        # Capture frames from the two external webcams
        ret1, webcam1_frame = webcam1.read()
        ret2, webcam2_frame = webcam2.read()
        
        
        if not ret1 and not ret2:
            print("Failed to capture frames from all external webcams.")
            continue

        elif not ret1:
            print("Failed to capture frame from external webcam 1 (index 0).")
            # Create blank frame for webcam1
            webcam1_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(webcam1_frame, "Camera 0 - No Signal", (50, 240), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        elif not ret2:
            print("Failed to capture frame from external webcam 2 (index 2).")
            # Create blank frame for webcam2
            webcam2_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(webcam2_frame, "Camera 2 - No Signal", (50, 240), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
        
        # Combine all frames into a list (RealSense and two external webcams)
        color_images = [webcam1_frame, webcam2_frame]
        

        # Capture frames from the two webcams
        # ret1, webcam1_frame = webcam1.read()
        # ret2, webcam2_frame = webcam2.read()
        # if not ret1 or not ret2:
        #     print("Failed to capture frames from webcams.")
        #     continue

        # # Combine all frames into a list
        # color_images = [color_image, webcam1_frame, webcam2_frame]

        # Get the encoder value
        encoder_value = read_encoder_value(ser)
        if encoder_value is None:
            encoder_value = 0  # Default if no reading available

        # Save frames and encoder value
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        for i, color_image in enumerate(color_images):
            frame_filename = os.path.join(output_dir, f"camera_{i}frame{frame_count:04d}.png")
            cv2.imwrite(frame_filename, color_image)

        # Save encoder value in a metadata file
        metadata_filename = os.path.join(output_dir, f"metadata_{frame_count:04d}.txt")
        with open(metadata_filename, "w") as metadata_file:
            metadata_file.write(f"Encoder Position: {encoder_value}\n")
            metadata_file.write(f"Timestamp: {timestamp}\n")
            metadata_file.write(f"Motor State: {current_motor_state}\n")

        frame_count += 1

        # Display the images from all cameras
        stacked_images = np.hstack(color_images)
        cv2.putText(stacked_images, f"Encoder: {encoder_value}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(stacked_images, f"Motor: {current_motor_state}", (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.namedWindow('RealSense and Webcams', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense and Webcams', stacked_images)

        # Process OpenCV keys for quitting
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            stop_event.set()
            break

except Exception as e:
    print(f"Error: {e}")

finally:
    if 'stop_event' in locals():
        stop_event.set()
    if 'ser' in locals() and ser.is_open:
        ser.write(b'S')
        ser.close()
        print("Serial connection closed")

    webcam1.release()
    webcam2.release()
    
    cv2.destroyAllWindows()
    print("All resources released")