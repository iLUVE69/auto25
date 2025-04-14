
from ultralytics import YOLO
import os
import cv2  # OpenCV for image processing
import numpy as np  # NumPy for numerical operations
import torch  # PyTorch for tensor operations
import time  # For calculating delta_time
import pyfirmata2
com_channel = 'COM6'
baud_rate = 9600
board = pyfirmata2.Arduino(com_channel)
a_pin = board.get_pin('d:4:o')  # Direction control
b_pin = board.get_pin('d:5:p')  # PWM control

# Function to send PWM values to the motor
def send_pwm(value):
    abs_value = abs(value)
    if value < 0:
        a_pin.write(1)
        b_pin.write(abs_value / 255)
    elif value > 0:
        a_pin.write(0)
        b_pin.write(abs_value / 255)
    else:
        a_pin.write(0)
        b_pin.write(0)

# Define the GLASBEY color palette for mask visualization
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

# Function to display annotated masks using GPU acceleration
def fast_show_mask_gpu(annotation):
    mask_sum = annotation.shape[0]
    height, weight = annotation.shape[1], annotation.shape[2]
    areas = torch.sum(annotation, dim=(1, 2))
    sorted_indices = torch.argsort(areas, descending=False)
    annotation = annotation[sorted_indices]
    index = (annotation != 0).to(torch.long).argmax(dim=0)
    color = GLASBEY[:mask_sum].reshape(mask_sum, 1, 1, 3).to(annotation.device)
    transparency = torch.ones((mask_sum, 1, 1, 1)).to(annotation.device) * 0.5
    visual = torch.cat([color, transparency], dim=-1)
    mask_image = torch.unsqueeze(annotation, -1) * visual
    show = torch.zeros((height, weight, 4)).to(annotation.device)
    h_indices, w_indices = torch.meshgrid(torch.arange(height), torch.arange(weight), indexing='ij')
    indices = (index[h_indices, w_indices], h_indices, w_indices, slice(None))
    show[h_indices, w_indices, :] = mask_image[indices]
    show_cpu = show.cpu().numpy()
    return show_cpu

# Load the YOLO model with the specified weights
model = YOLO('E:/Autonomous_vehicle_project/lane/best.pt')  # Update with actual path

# Define the trapezoidal region of interest (ROI) vertices
trap_vertices = np.array([[100, 480], [540, 480], [420, 300], [220, 300]], dtype=np.int32)
mask = np.zeros((480, 640), dtype=np.uint8)
cv2.fillPoly(mask, [trap_vertices], 255)

class PIDController:
    def __init__(self, Kp, Ki, Kd, max_output=None, min_output=None):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.max_output = max_output
        self.min_output = min_output
        self.integral = 0
        self.previous_error = 0

    def calculate(self, error, delta_time, speed=1.0):
        proportional = self.Kp * error
        self.integral += error * delta_time
        integral = self.Ki * self.integral
        derivative = self.Kd * (error - self.previous_error) / delta_time
        self.previous_error = error
        output = proportional + integral + derivative
        if self.max_output is not None and output > self.max_output:
            output = self.max_output
        elif self.min_output is not None and output < self.min_output:
            output = self.min_output
        return output

pid_controller = PIDController(Kp=1600, Ki=5.0, Kd=5000, max_output=255, min_output=-255)

# Set up the camera for real-time video capture
cv2.namedWindow("Lane Detection", cv2.WINDOW_NORMAL)
cap = cv2.VideoCapture(0)  # Use 0 for the default camera

while cap.isOpened():
    start_time = time.time()
    success, frame = cap.read()
    if not success:
        break

    print("Frame captured. Running YOLO model...")
    results = model(frame)

    if results[0].masks is None or results[0].masks.data is None:
        print("No masks detected in this frame. Continuing to next frame...")
        continue  # Skip the rest of the loop and go to the next frame

    annotated_frame = fast_show_mask_gpu(results[0].masks.data)
    # Ensure `annotated_frame` matches the size of `frame`
    # Ensure `annotated_frame` matches the size of `frame`
    annotated_frame = cv2.resize(annotated_frame, (frame.shape[1], frame.shape[0]))

    # Recompute `alpha` and adjust it to match `frame` dimensions with three channels
    alpha = annotated_frame[:, :, 3]  # Extract the alpha channel
    alpha = cv2.resize(alpha, (frame.shape[1], frame.shape[0]))  # Resize to match `frame`
    alpha_3_channel = np.repeat(alpha[:, :, np.newaxis], 3, axis=2)  # Convert to 3 channels

    # Now combine `annotated_frame` and `frame` using alpha blending
    annotated_frame_rgb = annotated_frame[:, :, :3]  # Take only the RGB part
    annotated_frame = (annotated_frame_rgb * 255 * alpha_3_channel + frame * (1 - alpha_3_channel)) / 255

    lane_mask = results[0].masks.data[0].cpu().numpy()
    lane_mask = (lane_mask * 255).astype(np.uint8)
    mask = cv2.resize(mask, (lane_mask.shape[1], lane_mask.shape[0]))
    combined_mask = cv2.bitwise_and(lane_mask, lane_mask, mask=mask)

    centerline = []
    height, width = combined_mask.shape
    image_center_x = width / 2

    for y in range(height):
        x_coords = np.where(combined_mask[y, :] > 0)[0]
        if len(x_coords) > 0:
            x_mean = np.mean(x_coords)
            centerline.append((int(x_mean), y))

    if centerline:
        centerline_x = np.array([pt[0] for pt in centerline])
        offsets = centerline_x - image_center_x
        average_offset = np.mean(offsets)
        error_normalised = average_offset / (width / 2)
    else:
        error_normalised = 0  # No lane detected, no offset

    delta_time = time.time() - start_time
    pwm_value = pid_controller.calculate(error_normalised, delta_time)
    send_pwm(pwm_value)
    cv2.putText(annotated_frame, f"Error: {error_normalised:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(annotated_frame, f"PWM: {pwm_value:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.imshow("Lane Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()