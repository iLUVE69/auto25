# Combined Imports
import serial
import re
import time
import threading
import cv2
import numpy as np
import torch
from ultralytics import YOLO
# import pyfirmata2 # <-- REMOVED
import math

# --- Configuration ---

# Serial configurationsq
GPS_COM_PORT = 'COM20'
MOTOR_CONTROL_COM_PORT = 'COM4' # <<< Consolidated Port for ALL motor commands
GPS_BAUD_RATE = 115200
MOTOR_CONTROL_BAUD_RATE = 9600 # <<< Baud rate for COM4

# PyFirmata Pins (REMOVED)
# PYFIRMATA_DIR_PIN = 'd:4:o'
# PYFIRMATA_PWM_PIN = 'd:5:p'

# Camera
CAMERA_INDEX = 1

# YOLO Model
YOLO_MODEL_PATH = r"C:\C++\model.pt" # Update if needed

# GPS Turn Logic
TARGET_LOCATIONS = [
    (23.2112928, 72.6859224),  # Turn 1
    (23.211306, 72.686403),    # Turn 2
    (23.2133064, 72.6864152)   # Turn 3
]
GPS_NEAR_THRESHOLD = 0.000065
COOLDOWN_TIME = 10
MIN_MOVEMENT_THRESHOLD = 0.00002
# --- Commands for COM4 ---
GPS_TURN_COMMAND_PREFIX = "SET:" # Prefix for absolute position turns
LANE_FOLLOW_COMMAND_PREFIX = "PWM:" # NEW Prefix for PWM-like steering
GPS_TURN_COMMAND_1 = -4500 # Value for SET command
GPS_TURN_COMMAND_2 = 4500 # Value for SET command
GPS_TURN_DELAY = 4
# ---

# GPS pattern
gps_pattern = re.compile(r"LAT=(-?\d+\.\d+),LON=(-?\d+\.\d+)")

# PID Controller
PID_KP = 1600
PID_KI = 5.0
PID_KD = 5000
PID_MAX_OUTPUT = 255 # Corresponds to max value sent with "PWM:" command
PID_MIN_OUTPUT = -255 # Corresponds to min value sent with "PWM:" command

# Visualization and ROI
trap_vertices = np.array([[100, 480], [540, 480], [420, 300], [220, 300]], dtype=np.int32)
# GLASBEY color palette (Ensure this is copied correctly)
GLASBEY = [
    (0, 0, 255), (255, 0, 0), (0, 255, 0), (0, 0, 51), (255, 0, 182),
    (0, 83, 0), (255, 211, 0), (0, 159, 255), (154, 77, 66), (0, 255, 190),
    (120, 63, 193), (31, 150, 152), (255, 172, 253), (177, 204, 113),
    (241, 8, 92), (254, 143, 66), (221, 0, 255), (32, 26, 1), (114, 0, 85),
    (118, 108, 149), (2, 173, 36), (200, 255, 0), (136, 108, 0),
    (255, 183, 159), (133, 133, 103), (161, 3, 0), (20, 249, 255),
    (0, 71, 158), (220, 94, 147), (147, 212, 255), (0, 76, 255),
    (0, 66, 80), (57, 167, 106), (238, 112, 254), (0, 0, 100),
    (171, 245, 204), (161, 146, 255), (164, 255, 115), (255, 206, 113),
    (71, 0, 21), (212, 173, 197), (251, 118, 111), (171, 188, 0),
    (117, 0, 215), (166, 0, 154), (0, 115, 254), (165, 93, 174),
    (98, 132, 2), (0, 121, 168), (0, 255, 131), (86, 53, 0), (159, 0, 63),
    (66, 45, 66), (255, 242, 187), (0, 93, 67), (252, 255, 124),
    (159, 191, 186), (167, 84, 19), (74, 39, 108), (0, 16, 166),
    (145, 78, 109), (207, 149, 0), (195, 187, 255), (253, 68, 64),
    (66, 78, 32), (106, 1, 0), (181, 131, 84), (132, 233, 147),
    (96, 217, 0), (255, 111, 211), (102, 75, 63), (254, 100, 0),
    (228, 3, 127), (17, 199, 174), (210, 129, 139), (91, 118, 124),
    (32, 59, 106), (180, 84, 255), (226, 8, 210), (0, 1, 20),
    (93, 132, 68), (166, 250, 255), (97, 123, 201), (98, 0, 122),
    (126, 190, 58), (0, 60, 183), (255, 253, 0), (7, 197, 226),
    (180, 167, 57), (148, 186, 138), (204, 187, 160), (55, 0, 49),
    (0, 40, 1), (150, 122, 129), (39, 136, 38), (206, 130, 180),
    (150, 164, 196), (180, 32, 128), (110, 86, 180), (147, 0, 185),
    (199, 48, 61), (115, 102, 255), (15, 187, 253), (172, 164, 100),
    (182, 117, 250), (216, 220, 254), (87, 141, 113), (216, 85, 34),
    (0, 196, 103), (243, 165, 105), (216, 255, 182), (1, 24, 219),
    (52, 66, 54), (255, 154, 0), (87, 95, 1), (198, 241, 79),
    (255, 95, 133), (123, 172, 240), (120, 100, 49), (162, 133, 204),
    (105, 255, 220), (198, 82, 100), (121, 26, 64), (0, 238, 70),
    (231, 207, 69), (217, 128, 233), (255, 211, 209), (209, 255, 141),
    (36, 0, 3), (87, 163, 193), (211, 231, 201), (203, 111, 79),
    (62, 24, 0), (0, 117, 223), (112, 176, 88), (209, 24, 0),
    (0, 30, 107), (105, 200, 197), (255, 203, 255), (233, 194, 137),
    (191, 129, 46), (69, 42, 145), (171, 76, 194), (14, 117, 61),
    (0, 30, 25), (118, 73, 127), (255, 169, 200), (94, 55, 217),
    (238, 230, 138), (159, 54, 33), (80, 0, 148), (189, 144, 128),
    (0, 109, 126), (88, 223, 96), (71, 80, 103), (1, 93, 159),
    (99, 48, 60), (2, 206, 148), (139, 83, 37), (171, 0, 255),
    (141, 42, 135), (85, 83, 148), (150, 255, 0), (0, 152, 123),
    (255, 138, 203), (222, 69, 200), (107, 109, 230), (30, 0, 68),
    (173, 76, 138), (255, 134, 161), (0, 35, 60), (138, 205, 0),
    (111, 202, 157), (225, 75, 253), (255, 176, 77), (229, 232, 57),
    (114, 16, 255), (111, 82, 101), (134, 137, 48), (99, 38, 80),
    (105, 38, 32), (200, 110, 0), (209, 164, 255), (198, 210, 86),
    (79, 103, 77), (174, 165, 166), (170, 45, 101), (199, 81, 175),
    (255, 89, 172), (146, 102, 78), (102, 134, 184), (111, 152, 255),
    (92, 255, 159), (172, 137, 178), (210, 34, 98), (199, 207, 147),
    (255, 185, 30), (250, 148, 141), (49, 34, 78), (254, 81, 97),
    (254, 141, 100), (68, 54, 23), (201, 162, 84), (199, 232, 240),
    (68, 152, 0), (147, 172, 58), (22, 75, 28), (8, 84, 121),
    (116, 45, 0), (104, 60, 255), (64, 41, 38), (164, 113, 215),
    (207, 0, 155), (118, 1, 35), (83, 0, 88), (0, 82, 232),
    (43, 92, 87), (160, 217, 146), (176, 26, 229), (29, 3, 36),
    (122, 58, 159), (214, 209, 207), (160, 100, 105), (106, 157, 160),
    (153, 219, 113), (192, 56, 207), (125, 255, 89), (149, 0, 34),
    (213, 162, 223), (22, 131, 204), (166, 249, 69), (109, 105, 97),
    (86, 188, 78), (255, 109, 81), (255, 3, 248), (255, 0, 73),
    (202, 0, 35), (67, 109, 18), (234, 170, 173), (191, 165, 0),
    (38, 44, 51), (85, 185, 2), (121, 182, 158), (254, 236, 212),
    (139, 165, 89), (141, 254, 193), (0, 60, 43), (63, 17, 40),
    (255, 221, 246), (17, 26, 146), (154, 66, 84), (149, 157, 238),
    (126, 130, 72), (58, 6, 101), (189, 117, 101)
]
GLASBEY = torch.tensor(GLASBEY) / 255.0  # Normalize colors to [0,1] range for PyTorch tensors


# --- State and Shared Variables ---
current_state = "LANE_FOLLOWING"
current_lat = None
current_lon = None
previous_lat = None
previous_lon = None
last_trigger_time = 0
visited_targets = set()
gps_turn_in_progress = False

# Threading Locks
gps_lock = threading.Lock()
state_lock = threading.Lock()
motor_serial_lock = threading.Lock() # Lock for accessing the shared motor serial port

# --- Global Serial Object ---
motor_ser = None # Will hold the serial connection for motor control (COM4)

# --- Initialization ---
print("Initializing components...")

# 1. Motor Control Serial Port (COM4)
try:
    motor_ser = serial.Serial(MOTOR_CONTROL_COM_PORT, MOTOR_CONTROL_BAUD_RATE, timeout=1)
    time.sleep(2) # Allow time for potential Arduino reset after connection
    print(f"Motor control serial connected on {MOTOR_CONTROL_COM_PORT}")
    # Send initial stop command
    motor_ser.write(f"{LANE_FOLLOW_COMMAND_PREFIX} 0\n".encode('utf-8'))
    print("Initial 'PWM: 0' command sent.")
except serial.SerialException as e:
    print(f"FATAL: Could not connect motor control serial on {MOTOR_CONTROL_COM_PORT}: {e}")
    exit()
except Exception as e:
      print(f"FATAL: Error initializing motor control serial: {e}")
      exit()


# 2. Camera
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print(f"FATAL: Could not open camera {CAMERA_INDEX}")
    if motor_ser and motor_ser.is_open: motor_ser.close()
    exit()
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Camera {CAMERA_INDEX} opened ({frame_width}x{frame_height})")

# 3. YOLO Model
try:
    model = YOLO(YOLO_MODEL_PATH)
    dummy_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
    model(dummy_frame, verbose=False)
    print(f"YOLO model loaded from {YOLO_MODEL_PATH}")
except Exception as e:
    print(f"FATAL: Could not load YOLO model: {e}")
    cap.release()
    if motor_ser and motor_ser.is_open: motor_ser.close()
    exit()


# 5. ROI Mask
roi_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
if frame_width != 640 or frame_height != 480:
    print("WARNING: Camera resolution differs from ROI vertices assumption (640x480). Adjust 'trap_vertices'.")
cv2.fillPoly(roi_mask, [trap_vertices], 255)
print("ROI Mask created.")

# --- Helper Functions ---

# NEW: Function to send PWM command via Serial (COM4)
def send_serial_pwm(value):
    """Sends PWM command formatted as 'PWM: <value>\n' via the global motor_ser."""
    global motor_ser
    if motor_ser is None or not motor_ser.is_open:
        print("ERROR: Motor serial port not available for PWM.")
        return

    try:
        # Ensure value is integer and clipped
        pwm_val = int(np.clip(value, PID_MIN_OUTPUT, PID_MAX_OUTPUT))
        command = f"{LANE_FOLLOW_COMMAND_PREFIX} {pwm_val}\n"

        with motor_serial_lock: # Ensure only one thread writes at a time
             motor_ser.write(command.encode('utf-8'))
        # print(f"Serial PWM Sent: {command.strip()}") # Debug
    except serial.SerialException as e:
        print(f"ERROR: Serial error sending PWM command: {e}")
        # Attempt to close/reopen or handle error appropriately might be needed here
    except Exception as e:
        print(f"ERROR: Unexpected error sending PWM command: {e}")


# MODIFIED: Function to set encoder position via Serial (COM4) using the global motor_ser
def set_encoder_position(position):
    """Sends SET command formatted as 'SET: <position>\n' via the global motor_ser."""
    global motor_ser
    if motor_ser is None or not motor_ser.is_open:
        print("ERROR: Motor serial port not available for SET command.")
        return False

    try:
        command = f"{GPS_TURN_COMMAND_PREFIX} {position}\n"
        print(f"--> GPS TURN: Sending command: {command.strip()}")
        with motor_serial_lock: # Ensure only one thread writes at a time
            motor_ser.write(command.encode('utf-8'))
            # Optional: Wait for Arduino acknowledgment if implemented
            # response = motor_ser.readline().decode('utf-8', errors='ignore').strip()
            # print(f"--> GPS TURN: Arduino Response: {response}")
        print(f"--> GPS TURN: SET command for {position} sent.")
        return True
    except serial.SerialException as e:
        print(f"ERROR: Serial error sending SET command: {e}")
        return False
    except Exception as e:
        print(f"ERROR: Unexpected error sending SET command: {e}")
        return False

# Function for visualization (from Script 2) - NO CHANGE NEEDED
def fast_show_mask_gpu(annotation):
    # <<< Copy the exact function content from your Script 2 here >>>
    mask_sum = annotation.shape[0]
    height, weight = annotation.shape[1], annotation.shape[2]
    areas = torch.sum(annotation, dim=(1, 2))
    sorted_indices = torch.argsort(areas, descending=False)
    annotation = annotation[sorted_indices]
    index = (annotation != 0).to(torch.long).argmax(dim=0)
    device = annotation.device
    glasbey_tensor = GLASBEY.to(device) # Ensure GLASBEY is defined and tensorized
    color = glasbey_tensor[:mask_sum].reshape(mask_sum, 1, 1, 3)
    transparency = torch.ones((mask_sum, 1, 1, 1)).to(device) * 0.5
    visual = torch.cat([color, transparency], dim=-1)
    mask_image = torch.unsqueeze(annotation, -1) * visual
    show = torch.zeros((height, weight, 4)).to(device)
    h_indices, w_indices = torch.meshgrid(torch.arange(height, device=device), torch.arange(weight, device=device), indexing='ij')
    indices = (index[h_indices, w_indices], h_indices, w_indices, slice(None))
    show[h_indices, w_indices, :] = mask_image[indices]
    show_cpu = show.cpu().numpy()
    return show_cpu


# PID Controller Class (from Script 2) - NO CHANGE NEEDED
class PIDController:
    # <<< Copy the exact class content from your Script 2 here >>>
    def __init__(self, Kp, Ki, Kd, max_output=None, min_output=None):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.max_output = max_output
        self.min_output = min_output
        self.integral = 0
        self.previous_error = 0
        self.last_time = time.time()

    def calculate(self, error):
        current_time = time.time()
        delta_time = current_time - self.last_time
        if delta_time <= 0:
             delta_time = 1e-6

        proportional = self.Kp * error
        self.integral += error * delta_time
        self.integral = np.clip(self.integral, -500, 500)
        integral_term = self.Ki * self.integral
        derivative = self.Kd * (error - self.previous_error) / delta_time
        output = proportional + integral_term + derivative
        self.previous_error = error
        self.last_time = current_time

        if self.max_output is not None:
            output = np.clip(output, self.min_output, self.max_output)
        return output


# 4. PID Controller
pid_controller = PIDController(Kp=PID_KP, Ki=PID_KI, Kd=PID_KD, max_output=PID_MAX_OUTPUT, min_output=PID_MIN_OUTPUT)
print("PID Controller initialized.")


# Function to check proximity (from Script 1) - NO CHANGE NEEDED
def is_near_target(lat, lon, threshold=GPS_NEAR_THRESHOLD):
    if lat is None or lon is None: return None
    for target_coords in TARGET_LOCATIONS:
        if abs(lat - target_coords[0]) < threshold and abs(lon - target_coords[1]) < threshold:
            return target_coords
    return None

# --- GPS Reading Thread --- (NO CHANGE NEEDED internally, uses different COM port)
def read_gps_continuously():
    global current_lat, current_lon, previous_lat, previous_lon
    gps_ser = None
    while True:
        try:
            if gps_ser is None or not gps_ser.is_open:
                print("Attempting to connect GPS...")
                gps_ser = serial.Serial(GPS_COM_PORT, GPS_BAUD_RATE, timeout=2)
                print(f"GPS connected on {GPS_COM_PORT}.")
                time.sleep(1)

            if gps_ser.in_waiting > 0:
                line = gps_ser.readline().decode('utf-8', errors='ignore').strip()
                match = gps_pattern.search(line)

                if match:
                    lat, lon = float(match.group(1)), float(match.group(2))
                    with gps_lock:
                        moved = True
                        if previous_lat is not None and previous_lon is not None:
                             if abs(lat - previous_lat) < MIN_MOVEMENT_THRESHOLD and abs(lon - previous_lon) < MIN_MOVEMENT_THRESHOLD:
                                 moved = False
                        if moved:
                            current_lat = lat
                            current_lon = lon
                            previous_lat = lat
                            previous_lon = lon
                        else:
                            current_lat = lat
                            current_lon = lon
        except serial.SerialException as e:
            print(f"GPS Serial error: {e}")
            if gps_ser and gps_ser.is_open: gps_ser.close()
            gps_ser = None
            with gps_lock: current_lat, current_lon = None, None
            print("GPS waiting 5s to reconnect...")
            time.sleep(5)
        except Exception as e:
            print(f"Unexpected error in GPS thread: {e}")
            time.sleep(2)
        time.sleep(0.05)

# --- GPS Turn Execution Thread --- (Uses MODIFIED set_encoder_position)
def execute_gps_turn(target_coords):
    global gps_turn_in_progress
    print(f"*** Starting GPS Turn Sequence for target {target_coords} ***")

    with state_lock:
        gps_turn_in_progress = True

    # Send first command
    success1 = set_encoder_position(GPS_TURN_COMMAND_1) # Uses modified function
    if success1:
        print(f"GPS Turn: Waiting {GPS_TURN_DELAY} seconds...")
        time.sleep(GPS_TURN_DELAY)
        # Send second command
        success2 = set_encoder_position(GPS_TURN_COMMAND_2) # Uses modified function
    else:
        print("GPS Turn: Skipped second command due to failure on first.")

    print(f"*** GPS Turn Sequence for {target_coords} Finished ***")
    with state_lock:
        gps_turn_in_progress = False


# --- Main Control Loop ---
try:
    # Start GPS reading thread
    gps_thread = threading.Thread(target=read_gps_continuously, daemon=True)
    gps_thread.start()
    print("GPS reader thread started.")
    time.sleep(5) # Wait for initial GPS

    cv2.namedWindow("Lane Detection", cv2.WINDOW_NORMAL)

    while True:
        # Read Camera Frame
        success, frame = cap.read()
        if not success:
            print("Error: Failed to grab camera frame."); time.sleep(0.5); continue

        # Get current state and GPS data
        with state_lock: active_state = current_state
        with gps_lock: local_lat, local_lon = current_lat, current_lon

        # --- State Machine Logic ---
        if active_state == "LANE_FOLLOWING":
            # 1. Check if GPS turn should be triggered
            target_found = is_near_target(local_lat, local_lon)
            should_trigger_turn = False
            if target_found:
                with state_lock:
                    if target_found not in visited_targets and (time.time() - last_trigger_time > COOLDOWN_TIME):
                        should_trigger_turn = True
                        print(f"!!! GPS Target {target_found} detected. Preparing turn. !!!")
                        visited_targets.add(target_found)
                        last_trigger_time = time.time()
                        current_state = "GPS_TURNING"

            if should_trigger_turn:
                 print("Switching to GPS_TURNING. Stopping Serial PWM.")
                 send_serial_pwm(0) # <<< Send PWM: 0 command via Serial
                 time.sleep(0.1) # Small delay to ensure command is sent/processed
                 # Start the turn sequence thread
                 turn_thread = threading.Thread(target=execute_gps_turn, args=(target_found,), daemon=True)
                 turn_thread.start()
                 display_state = "SWITCHING_TO_GPS"

            else:
                # 2. Perform Lane Following
                display_state = "LANE_FOLLOWING"
                try:
                    results = model(frame, verbose=False)
                    if results[0].masks is not None and results[0].masks.data is not None and len(results[0].masks.data) > 0:
                        # --- Lane Processing (Identical to previous version) ---
                        annotated_overlay = fast_show_mask_gpu(results[0].masks.data)
                        annotated_overlay = cv2.resize(annotated_overlay, (frame_width, frame_height))
                        alpha = annotated_overlay[:, :, 3]
                        alpha_3_channel = np.repeat(alpha[:, :, np.newaxis], 3, axis=2)
                        annotated_frame_rgb = annotated_overlay[:, :, :3]
                        frame_display = (annotated_frame_rgb * alpha_3_channel + frame.astype(np.float32)/255.0 * (1 - alpha_3_channel)) * 255
                        frame_display = frame_display.astype(np.uint8)

                        lane_mask_tensor = results[0].masks.data[0]
                        lane_mask = (lane_mask_tensor.cpu().numpy() * 255).astype(np.uint8)
                        lane_mask_resized = cv2.resize(lane_mask, (frame_width, frame_height))
                        combined_mask = cv2.bitwise_and(lane_mask_resized, lane_mask_resized, mask=roi_mask)

                        centerline = []
                        image_center_x = frame_width / 2
                        for y in range(frame_height):
                             x_coords = np.where(combined_mask[y, :] > 0)[0]
                             if len(x_coords) > 0: centerline.append((int(np.mean(x_coords)), y))

                        if centerline:
                            lower_centerline_x = np.array([pt[0] for pt in centerline if pt[1] > frame_height * 0.6])
                            if len(lower_centerline_x) > 0:
                                error_normalised = np.mean(lower_centerline_x - image_center_x) / (frame_width / 2)
                            else:
                                error_normalised = np.mean(np.array([pt[0] for pt in centerline]) - image_center_x) / (frame_width / 2)
                        else:
                             error_normalised = 0

                        # --- PID Calculation ---
                        pid_output_value = pid_controller.calculate(error_normalised)

                        # --- Send PWM command via SERIAL ---
                        send_serial_pwm(pid_output_value) # <<< USE NEW SERIAL PWM FUNCTION

                        # Display info
                        cv2.putText(frame_display, f"Error: {error_normalised:.3f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        cv2.putText(frame_display, f"SerialPWM: {int(pid_output_value)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        # ---------------------------------------------

                    else:
                        # No mask detected
                        print("No lane mask detected.")
                        send_serial_pwm(0) # <<< Send PWM: 0 via Serial
                        frame_display = frame.copy()
                        cv2.putText(frame_display, "No Lane Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                except Exception as e:
                    print(f"Error during lane following processing: {e}")
                    send_serial_pwm(0) # <<< Stop on error via Serial
                    frame_display = frame.copy()
                    cv2.putText(frame_display, "Processing Error", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


        elif active_state == "GPS_TURNING":
            # Wait for turn thread
            display_state = "GPS_TURNING"
            frame_display = frame.copy()
            with state_lock:
                if not gps_turn_in_progress:
                    print("GPS turn finished. Switching back to LANE_FOLLOWING.")
                    current_state = "LANE_FOLLOWING"
                    pid_controller.integral = 0
                    pid_controller.previous_error = 0
                    pid_controller.last_time = time.time()
            cv2.putText(frame_display, "EXECUTING GPS TURN", (frame_width // 2 - 150, frame_height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)


        # --- Display ---
        cv2.putText(frame_display, f"State: {display_state}", (10, frame_height - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if local_lat and local_lon:
             cv2.putText(frame_display, f"GPS: {local_lat:.5f}, {local_lon:.5f}", (10, frame_height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
             cv2.putText(frame_display, "GPS: Waiting for fix...", (10, frame_height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow("Lane Detection", frame_display)

        # --- Loop Control ---
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("'q' pressed. Exiting...")
            break
        # time.sleep(0.01) # Optional small delay

except KeyboardInterrupt:
    print("Ctrl+C detected. Stopping.")

finally:
    # --- Cleanup ---
    print("Cleaning up resources...")
    if 'cap' in locals() and cap.isOpened():
        cap.release()
        print("Camera released.")
    cv2.destroyAllWindows()
    print("OpenCV windows closed.")

    # Close Motor Control Serial Port (COM4)
    if 'motor_ser' in locals() and motor_ser and motor_ser.is_open:
         try:
             # Send a final stop command before closing
             print("Sending final stop command via Serial...")
             send_serial_pwm(0)
             time.sleep(0.2) # Give it time to send/process
             motor_ser.close()
             print(f"Motor control serial port {MOTOR_CONTROL_COM_PORT} closed.")
         except Exception as e:
              print(f"Error closing motor serial port: {e}")

    # GPS serial port is managed by its thread

    print("Cleanup complete. Exiting.")