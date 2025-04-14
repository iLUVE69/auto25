import argparse
from pathlib import Path
import math
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
import threading
import serial
import time
import re
from threading import Lock
from shapely.geometry import Point
import pyproj
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
import pyrealsense2 as rs
import numpy as np
import pyfirmata2
from ultralytics import YOLO
import os
import cv2  


BAUD_RATE=9600
com_port = 'COM4' 

board = pyfirmata2.Arduino(com_port)
osm_file_path = "E:\Autonomous-2024\map(8).osm"
graph = ox.graph_from_xml(osm_file_path)
for i, node in enumerate(graph.nodes()):
    graph.nodes[node]['name'] = f"CustomNode{i+1}"


print('Firmata connection established') 
brake_dir = board.get_pin('d:13:o')  # Example pin setup for 'a'
brake_pwm = board.get_pin('d:6:p')  # Example pin setup for 'b' as PWM
accn_1 = board.get_pin('d:2:o')  #accn motor driver pins
accn_2 = board.get_pin('d:3:o')
accn_pwm = board.get_pin('d:10:p') #en
a_pin = board.get_pin('d:4:o')  # Direction control
b_pin = board.get_pin('d:5:p')

current_lat, current_lon = None, None
selected_nodes = []
current_position_marker = None
destination_coords = None  # Coordinates of the second selected node
gps_pattern = re.compile(r"LAT=([-+]?[0-9]*\.?[0-9]+),LON=([-+]?[0-9]*\.?[0-9]+)")
graph_crs = graph.graph['crs']
projector = pyproj.Transformer.from_crs("epsg:4326", graph_crs, always_xy=True)

# Lists to store error values for plotting
latitude_errors = []
longitude_errors = []

brake_active = 0
maxspeed = 160  
# slowsspeed = 190
currentspeed=0.0
increment = 10.0
go_count = 0
max_go_count = 0

stop_distance = 1
slow_distance = 4 # velocity control

frame = cv2.imread('obstacledetection/yolov7modified/snakeroad.jpg')

def update_error_plot():
    plt.figure("Error Plot")
    plt.clf()
    plt.plot(latitude_errors, label="Latitude Error")
    plt.plot(longitude_errors, label="Longitude Error")
    plt.xlabel("Time Step")
    plt.ylabel("Error")
    plt.legend()
    plt.pause(0.05)


def calculate_position_error(current_lat, current_lon, destination_lat, destination_lon):
    # Calculate the latitude and longitude error
    latitude_error = destination_lat - current_lat
    longitude_error = destination_lon - current_lon
    
    return latitude_error, longitude_error

def update_position_on_map():
    global current_lat, current_lon, ax, canvas, current_position_marker

    try:
        if current_lat is not None and current_lon is not None:
            # Convert GPS coordinates (latitude and longitude) to the graph's CRS
            x, y = projector.transform(current_lon, current_lat)

            print(f"Projected coordinates: x={x}, y={y}")

            # Remove the previous position marker if it exists
            if current_position_marker:
                current_position_marker.remove()

            # Plot the current position without clearing the entire map
            current_position_marker, = ax.plot(
                x, y,
                marker='o', color='blue', markersize=10, label='Current Position'
            )
            

            canvas.draw()
            print(f"Position updated: Lat={current_lat}, Lon={current_lon}")
            
    except Exception as e:
        print(f"Error updating position: {e}")


def calculate_absolute_heading(pointA, pointB):
    """
    Calculate the absolute heading (true bearing) between two latitude/longitude points.
    :param pointA: Tuple of (lat1, lon1) in degrees
    :param pointB: Tuple of (lat2, lon2) in degrees
    :return: Absolute heading in degrees (0° to 360°)
    """
    lat1, lon1 = math.radians(pointA[0]), math.radians(pointA[1])
    lat2, lon2 = math.radians(pointB[0]), math.radians(pointB[1])

    dLon = lon2 - lon1  # Difference in longitude

    # Bearing formula
    y = math.sin(dLon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dLon)
    bearing = math.atan2(y, x)

    # Convert bearing from radians to degrees and normalize to 0-360°
    bearing = math.degrees(bearing)
    bearing = (bearing + 360) % 360
    shared_sensor_data.absolute_heading = bearing

    return bearing

def read_serial():

    ser = serial.Serial('COM3', 38400, timeout=1)  # Replace 'COM3' with your Arduino's serial port
    heading_pattern = re.compile(r'HEADING\s+([\d.]+)')

    time.sleep(2)  # Allow serial connection to initialize

    try:
        while True:
            line = ser.readline().decode('utf-8').strip()
            try:
                match = heading_pattern.search(line)
                if match:
                    heading_value = float(match.group(1))
                    shared_sensor_data.heading = heading_value
                    print(f"Received heading: {heading_value}")
                # Add your logic here (e.g., save to file, send to API, etc.)
            except ValueError:
                pass  # Skip non-numeric lines like "heading"
    except KeyboardInterrupt:
        print("Serial reading stopped.")
    finally:
        ser.close()

def on_click(event):
    global destination_coords
    if len(selected_nodes) < 2:  # Allow selecting only two nodes
        # Find the nearest node to the click
        nearest_node = ox.distance.nearest_nodes(graph, event.xdata, event.ydata)
        selected_nodes.append(nearest_node)

        # Highlight the selected node
        ax.plot(event.xdata, event.ydata, marker='o', color='red', markersize=10)
        canvas.draw()

        if len(selected_nodes) == 2:
            # Once two nodes are selected, compute and display the shortest path using A*
            start_node, end_node = selected_nodes
            shortest_path = nx.astar_path(graph, start_node, end_node, weight='length')

            # Compute total path length (sum of edge lengths)
            total_length = sum(graph.edges[u, v, 0]['length'] for u, v in zip(shortest_path[:-1], shortest_path[1:]))

            # Extract lat/lon values of start and end nodes
            start_lat, start_lon = graph.nodes[start_node]['y'], graph.nodes[start_node]['x']
            end_lat, end_lon = graph.nodes[end_node]['y'], graph.nodes[end_node]['x']

            # Print the shortest path in terms of the custom node names
            print(f"\nShortest path between {graph.nodes[start_node]['name']} ({start_lat}, {start_lon}) "
                  f"and {graph.nodes[end_node]['name']} ({end_lat}, {end_lon}):")
            for node in shortest_path:
                print(graph.nodes[node]['name'])

            # Print total path length
            print(f"\nTotal path length: {total_length:.2f} meters")

            # Plot the shortest path
            ox.plot_graph_route(graph, shortest_path, route_linewidth=6, node_size=0, ax=ax)
            canvas.draw()
            dest_node_data = graph.nodes[end_node]
            destination_coords = (dest_node_data['y'], dest_node_data['x'])
            print(f"Destination coordinates set to: Latitude={destination_coords[0]}, Longitude={destination_coords[1]}")

            # Calculate and print the absolute heading for each segment in the path
            print("\nWaypoints with absolute headings (relative to true north) and lat/long values:")
            previous_point = (start_lat, start_lon)
            for node in shortest_path:
                lat, lon = graph.nodes[node]['y'], graph.nodes[node]['x']
                current_point = (lat, lon)
                if current_point != previous_point:
                    absolute_heading = calculate_absolute_heading(previous_point, current_point)
                    print(f"Node: {graph.nodes[node]['name']}, Lat: {lat:.6f}, Lon: {lon:.6f}, Absolute Heading: {absolute_heading:.2f}°")
                    previous_point = current_point

def on_scroll(event):
    try:
        base_scale = 1.25
        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()

        xdata = event.xdata  # get event x location
        ydata = event.ydata  # get event y location

        if event.button == 'up':
            scale_factor = 1 / base_scale
        elif event.button == 'down':
            scale_factor = base_scale
        else:
            scale_factor = 1.5
            print(event.button)

        new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor

        relx = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0])
        rely = (cur_ylim[1] - ydata) / (cur_ylim[1] - cur_ylim[0])

        ax.set_xlim([xdata - new_width * (1 - relx), xdata + new_width * (relx)])
        ax.set_ylim([ydata - new_height * (1 - rely), ydata + new_height * (rely)])
        ax.figure.canvas.draw()

        print(f"Zoom event: scale_factor={scale_factor}")
    except Exception as e:
        print(f"Error handling zoom event: {e}")

def read_gps_data():
    global current_lat, current_lon, destination_coords, destination_reached
    try:
        print("Attempting to connect to the serial port...")
        ser = serial.Serial(com_port, BAUD_RATE, timeout=0.1)
        print("Connected to the serial port successfully.")

        plt.ion()  # Turn on interactive mode for plotting
        while True:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if line:
                print(f"Received line from serial: {line}")
                # Match the Arduino output format for latitude and longitude
                match = gps_pattern.match(line)
                if match:
                    lat = float(match.group(1))
                    lon = float(match.group(2))
                    current_lat, current_lon = lat, lon
                    print(f"Updating current position to: Latitude={current_lat}, Longitude={current_lon}")
                    update_position_on_map()

                    # Check if the cart has reached the destination and calculate the position error
                    if destination_coords:
                        dest_lat, dest_lon = destination_coords
                        latitude_error, longitude_error = calculate_position_error(current_lat, current_lon, dest_lat, dest_lon)
                        print(f"Latitude Error: {latitude_error}, Longitude Error: {longitude_error}")

                        # Append errors to the lists for plotting
                        latitude_errors.append(latitude_error)
                        longitude_errors.append(longitude_error)

                        # Update the error plot
                        update_error_plot()

                        if 10**-5 < abs(latitude_error) < 10**-4 and 10**-5 < abs(longitude_error) < 10**-4:
                            vehicle_stop()
                            return
                            print("Errors are within range. Vehicle stop condition met.")
            
            time.sleep(0.1)  # Reduce delay for faster updates
    except serial.SerialException as e:
        print(f"Serial connection error: {e}")
    except KeyboardInterrupt:
        ser.close()
        print("\nSerial connection closed.")
    except Exception as e:
        print(f"Unexpected error: {e}")

def start_gui():
    global ax, canvas
    root = tk.Tk()
    root.title("Map Navigation")
    fig, ax = plt.subplots()
    ox.plot_graph(graph, ax=ax, show=False, close=True)
    canvas = FigureCanvasTkAgg(fig, master=root)

    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
    canvas.mpl_connect("button_press_event",on_click)
    canvas.mpl_connect("scroll_event", on_scroll)
    gps_thread = threading.Thread(target=read_gps_data, daemon=True)
    gps_thread.daemon=True
    gps_thread.start()
    root.mainloop()

def vehicle_stop():
    global go_count
    global brake_active
    global currentspeed
    go_count = 0
    print("stopping vehicle")
    brake_dir.write(1)
    brake_pwm.write(1)
    # brake_motor.setSpeed(255)
    brake_active = 1
    accn_pwm.write(0) #zero pwm input
    accn_1.write(1) #dirn
    accn_2.write(0)
    # motor.setSpeed(0)
    # motor.forward()
    currentspeed = 0
    

def vehicle_go():
    global go_count
    global brake_active
    global currentspeed
    go_count+=1
    if go_count > max_go_count:
        if brake_active == 1:
            print("releasing brakes")
            brake_dir.write(0)
            brake_pwm.write(1)
            time.sleep(0.5)
            brake_pwm.write(0)
            brake_active = 0
        
        if currentspeed < maxspeed:
            currentspeed+=increment
            if currentspeed > maxspeed:
                currentspeed = maxspeed
        
        accn_pwm.write(currentspeed/255)
        accn_1.write(1)
        accn_2.write(0)
        print(f'currentspeed = {currentspeed}')


def vehicle_slow():
    global go_count
    global brake_active
    global currentspeed
    go_count+=1
    if go_count > max_go_count:
        if brake_active == 1:
            print("releasing brakes")
            brake_dir.write(0)
            brake_pwm.write(1)
            time.sleep(0.5)
            brake_pwm.write(0)
            brake_active = 0
        
        # if currentspeed > slowsspeed:
        #     currentspeed-=increment 
        #     if currentspeed < slowsspeed:
        #         currentspeed = slowsspeed
        
        accn_pwm.write(currentspeed/255)
        accn_1.write(1)
        accn_2.write(0)
        print(f'currentspeed = {currentspeed}')

def cleanup_resources(pipeline):
    cv2.destroyAllWindows()
    if pipeline:
        pipeline.stop()

def canny(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = 7
    blur = cv2.GaussianBlur(gray, (kernel, kernel), sigmaX=0, sigmaY=0)
    canny = cv2.Canny(blur,80,100)
    # canny = cv2.Canny(blur,100,160)
    return canny


def region_of_interest_trapezium(img):
    height = img.shape[0]
    width = img.shape[1]
    mask = np.zeros_like(img)

    # Define coordinates for the trapezium
    # Adjust the points (x1, y1), (x2, y2), (x3, y3), (x4, y4) as needed
    bottom_left = (0, height)
    top_left = (0, height * 0.5)
    top_right = (width, height * 0.5)
    bottom_right = (width, height)

    # np.array expects points as [[first_point, second_point, third_point, fourth_point]]
    trapezium = np.array([[bottom_left, top_left, top_right, bottom_right]], np.int32)

    # Fill the polygon (trapezium here) with white (255)
    cv2.fillPoly(mask, [trapezium], 255)

    # Apply the mask
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def houghLines(img):
    houghLines = cv2.HoughLinesP(img, 2, np.pi / 180, 10, np.array([]), minLineLength=30, maxLineGap=10)
    return houghLines

def preprocess_frame(frame):
    # new_width = 640
    # new_height = 480
    # aspect_ratio = original_width / original_height
    # # Calculate the new height maintaining the aspect ratio
    # new_height = int(new_width / aspect_ratio)
    # frame = cv2.resize(frame, (new_width, new_height))
    # Convert frame to float
    frame_float = frame.astype(np.float32)
    # Calculate the mean of the pixel values
    mean = np.mean(frame_float)
    # Scale factor for contrast adjustment; values < 1.0 decrease contrast
    # scale_factor = 1.0
    scale_factor = 0.8
    # Adjust the contrast
    # Moving pixel values towards the mean to reduce contrast
    frame_adjusted = (frame_float - mean) * scale_factor + mean
    # Clip values to stay between 0 and 255 and convert back to uint8
    frame = np.clip(frame_adjusted, 0, 255).astype(np.uint8)
    return frame


def display_filled_region(img, lines, init_point):
    img_copy = img.copy()
    mask = np.zeros_like(img)
    if lines is not None:
        left_line = lines[0][0]  # Assuming first line is the left
        right_line = lines[1][0]  # Assuming second line is the right

        pts = np.array([[left_line[0], left_line[1]], [left_line[2], left_line[3]],
                        [right_line[2], right_line[3]], [right_line[0], right_line[1]]], np.int32)

        bottom_left_corner = pts[1]
        bottom_right_corner = pts[2]

        top_left_corner = [bottom_left_corner[0], bottom_left_corner[1] - 300]
        top_right_corner = [bottom_right_corner[0], bottom_right_corner[1] - 300]

        pts2 = np.array([bottom_left_corner, bottom_right_corner, top_right_corner, top_left_corner], np.int32)
        pts2 = pts2.reshape((-1, 1, 2))

        pts = pts.reshape((-1, 1, 2))
        image = cv2.fillPoly(img_copy, [pts], (144, 238, 144))  # Light green color
        image = cv2.fillPoly(img_copy, [pts2], (144, 238, 144))  # Light green color
        
        # image = cv2.polylines(img_copy, [pts], isClosed=True, color=(144, 238, 144), thickness=5)
        # image = cv2.polylines(img_copy, [pts2], isClosed=True, color=(144, 238, 144), thickness=5) 

    return image


def make_points(img, lineSI):
    slope, intercept = lineSI
    height = img.shape[0]
    y1 = int(height)
    y2 = int(y1 * 3 / 5)
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return [[x1, y1, x2, y2]]

def average_slope_intercept(img,lines):
    # lower_value_slope = 0.5
    # higher_value_slope = 3
    lower_value_slope = 0.5
    higher_value_slope = 2.5
    flag_left = True
    flag_right = True
    left_fit = []
    right_fit = []
    for line in lines:
        for x1,y1,x2,y2 in line:
            fit = np.polyfit((x1,x2),(y1,y2),1)
            slope = fit[0]
            intercept = fit[1]
            # print(intercept,slope)
            if slope < -lower_value_slope and slope >= -higher_value_slope:
                left_fit.append((slope, intercept))
            elif slope >= lower_value_slope and slope <= higher_value_slope :
                right_fit.append((slope,intercept))
    if left_fit == []:
        # left_fit = np.array([(0.0001,0.0001)])
        flag_left = False
    if right_fit == []:
        # right_fit = np.array([(0.0001,0.0001)])
        flag_right = False

    if flag_left:
        left_fit_average = np.average(left_fit,axis=0)
        left_line = make_points(img, left_fit_average)
    else:
        left_line = np.array([[None,None,None,None]])

    if flag_right:
        right_fit_average = np.average(right_fit,axis=0)
        right_line = make_points(img, right_fit_average)
    else:
        right_line = np.array([[None,None,None,None]])


    average_lines = [np.array(left_line),np.array(right_line)]
    return average_lines



def average_slope_intercept_with_centre(img, lines):
    lower_value_slope = 0.5
    higher_value_slope = 2.5
    flag_left = True
    flag_right = True
    left_fit = []
    right_fit = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            # Check if any of the points are None
            if any(p is None for p in [x1, y1, x2, y2]):
                continue
            fit = np.polyfit((x1, x2), (y1, y2), 1)
            slope = fit[0]
            intercept = fit[1]
            if slope < -lower_value_slope and slope >= -higher_value_slope:
                left_fit.append((slope, intercept))
            elif slope >= lower_value_slope and slope <= higher_value_slope:
                right_fit.append((slope, intercept))

    if not left_fit:
        flag_left = False
    if not right_fit:
        flag_right = False

    if flag_left:
        left_fit_average = np.average(left_fit, axis=0)
        left_line = make_points(img, left_fit_average)
    else:
        # left_line = np.array([[None, None, None, None]])
        left_line = np.array([])

    if flag_right:
        right_fit_average = np.average(right_fit, axis=0)
        right_line = make_points(img, right_fit_average)
    else:
        # right_line = np.array([[None, None, None, None]])
        right_line = np.array([])

    if flag_left and flag_right:
        center_line = np.array([[0, 0, 0, 0]])
        for i in range(4):
            center_line[0][i] = np.int32((left_line[0][i] + right_line[0][i]) / 2)
    else:
        # center_line = np.array([[None, None, None, None]])
        center_line = np.array([])

    average_lines = [np.array(left_line), np.array(right_line), np.array(center_line)]
    return average_lines


def update_line_history(line_history, new_line, history_length=5):
    if new_line[0].all() == np.array([None,None,None,None]).all() and line_history:
        # Use the most recent valid line if the new line is invalid
        new_line = line_history[-1]
    line_history.append(new_line)
    if len(line_history) > history_length:
        line_history.pop(0)
    # print(line_history)
    return line_history


def average_line_from_history(line_history):
    if not line_history:
        return np.array([0, 0, 0, 0])
    avg_line = np.mean(np.array(line_history), axis=0, dtype=np.int32)
    return avg_line


def lane_detection(frame):
    mask = frame.copy()
    left_line_history = []
    right_line_history = []
    history_length = 15
    init_point = (23, 384)

    try:
        canny_output = canny(frame)
        masked_output = region_of_interest_trapezium(canny_output)
        lines = houghLines(masked_output)
        average_lines = average_slope_intercept(frame,lines)
        average_lines_with_centre_avg = average_slope_intercept_with_centre(frame, lines)

        left_line = average_lines_with_centre_avg[0]
        right_line = average_lines_with_centre_avg[1]

        left_line_history = update_line_history(left_line_history, left_line, history_length)
        right_line_history = update_line_history(right_line_history, right_line, history_length)
        left_line_avg = average_line_from_history(left_line_history)
        right_line_avg = average_line_from_history(right_line_history)
        center_line_avg = np.array([[0, 0, 0, 0]])
        for i in range(4):
            center_line_avg[0][i] = np.int32((left_line_avg[0][i] + right_line_avg[0][i]) / 2)

        average_lines_with_centre_avg = np.array(
            [np.array(left_line_avg), np.array(right_line_avg), np.array(center_line_avg)])

        line_image_2_filled = display_filled_region(frame, average_lines_with_centre_avg, init_point)
        line_mask_filled = display_filled_region(mask, average_lines_with_centre_avg, init_point)
    except Exception as e:
        print("Error:", e)
        line_image_2_filled = frame
        line_mask_filled = frame

    return line_image_2_filled, line_mask_filled

def check_intersection(masked_img1, masked_img2):
    # Resize masked_img2 to match the dimensions of masked_img1
    masked_img2_resized = cv2.resize(masked_img2, (masked_img1.shape[1], masked_img1.shape[0]))
    intersection = cv2.bitwise_and(masked_img1, masked_img2)
    intersects = False
    if np.any(intersection != 0):
        intersects = True
        # print("Obstacle Entered Lane")
    return intersection, intersects


def detection():
    # board = Arduino('COM11')
    pipeline = None
    try:
        
        colorizer = rs.colorizer()
        colorizer.set_option(rs.option.visual_preset, 1)
        colorizer.set_option(rs.option.histogram_equalization_enabled, 1.0)  # disable histogram equalization
        colorizer.set_option(rs.option.color_scheme, 0)  # replace 'float' with your desired color scheme
        colorizer.set_option(rs.option.min_distance, 0.2)  # replace 'float' with your desired min distance
        colorizer.set_option(rs.option.max_distance, 6)  # replace 'float' with your desired max distance 

        source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
        # Directories
        save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Initialize
        set_logging()
        device = select_device(opt.device)
        half = device.type != 'cpu'  # half precision only supported on CUDA  
        # Load model
        print(device)
        model = attempt_load(weights, map_location=device)  # load FP32 model
        # model = attempt_load(weights, map_location=torch.device('cpu')) 
        #problem^^^^^^^^^^^^^^^^^^^^
        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(imgsz, s=stride)  # check img_size
        # if trace:
        #     model = TracedModel(model, device, opt.img_size)
        # if half:
        #     model.half()  # to FP16
        model.half()  # to FP16

        # Second-stage classifier
        classify = False
        if classify:
            modelc = load_classifier(name='resnet101', n=2)  # initialize
            modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

        # Run inference
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        old_img_w = old_img_h = imgsz
        old_img_b = 1

        config = rs.config()

        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    

        pipeline = rs.pipeline()


        try:
            profile = pipeline.start(config)
            print("Pipeline successfull.")
        except RuntimeError as e:
            print("Pipeline start failed:", e)

    
        while True:
            

            aligned_frames=pipeline.wait_for_frames()
            color_frame = aligned_frames.get_color_frame()
            lane_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
           


            img = np.asanyarray(color_frame.get_data())
            lane = np.asanyarray(lane_frame.get_data())
            # depth_image = np.asanyarray(depth_frame.get_data())
            # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.5), cv2.COLORMAP_JET)
            colorized_frame = colorizer.colorize(depth_frame)

            # Get lane masking for road
            frames = pipeline.wait_for_frames()
            # if color_frame is not None:
            #     color_image = color_image.reshape((480, 640, 3))

            img = preprocess_frame(img)
            lane_image, lane_masked_image = lane_detection(img)   #for live frame
           
            depth_colormap = np.asanyarray(colorized_frame.get_data())
            # Letterbox
            im0 = img.copy()
            img = img[np.newaxis, :, :, :]

            # Create a zero mask image
            im0_masked = np.zeros_like(im0)

            # Stack
            img = np.stack(img, 0)

            # Convert
            img = img[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
            img = np.ascontiguousarray(img)


            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Warmup
            if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
                old_img_b = img.shape[0]
                old_img_h = img.shape[2]
                old_img_w = img.shape[3]
                for i in range(3):
                    model(img, augment=opt.augment)[0]

            # Inference
            t1 = time_synchronized()
            with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
                pred = model(img, augment=opt.augment)[0]
            t2 = time_synchronized()

            # Apply NMS
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
            t3 = time_synchronized()

            # print(len(pred[0]))

            if len(pred[0])==0:
                print("Path is clear!! (No obstacles)")
                vehicle_go()
                # ser.write(b'stoop\n')

            # Process detections
            for i, det in enumerate(pred):  # detections per image

                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        #s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write resul
                    for *xyxy, conf, cls in reversed(det):
                        c = int(cls)  # integer class
                        label = f'{names[c]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=2)
                        plot_one_box(xyxy, depth_colormap, label=label, color=colors[int(cls)], line_thickness=2)

                    # Initialize a list to store distances of all detected objects
                    object_distances = []

                    for *xyxy, _, _ in det:
                        indv_mask = np.zeros_like(im0_masked)
                        # cv2.imshow("Masks", indv_mask)

                        # Draw bounding boxes on the zero mask image
                        cv2.rectangle(indv_mask, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (255, 255, 255), -1)
                        
                        # Check for intersection only if both images are non-empty
                        # if lane_masked_image.size != 0 and indv_mask.size != 0:
                        intersection, intersects = check_intersection(lane_masked_image, indv_mask)

                        
                        if intersects:
                            c = int(cls)  # integer class
                            label = f'{names[c]} {conf:.2f}'
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=2)

                            depth_data = np.array(depth_frame.get_data())

                            # Extract bounding box coordinates
                            x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])

                            # Get depth data within the bounding box
                            depth_region = depth_data[y1:y2, x1:x2]
                            
                            if np.count_nonzero(depth_region) > 0:
                                # Calculate the minimum distance within the bounding box
                                object_min_distance = np.min(depth_region[depth_region != 0]) * 0.001
                                object_distances.append(object_min_distance)
                            
                            else:
                                object_min_distance = float('inf')

                                # Add the distance to the list

                        else:  #there are no objects detected
                            print("Path is clear!!, no objects detected")
                            vehicle_go()
                            # ser.write(b'stoop\n')

                    # After iterating through all detected objects, find the minimum distance

                    if object_distances:
                        min_distance = min(object_distances)
                    # Check if the minimum distance is less than a threshold
                        if min_distance < stop_distance:
                            print(f"Obstacle detected within {min_distance:.2f} meters!")
                            vehicle_stop()
                           
                        else:
                            print("Path is clear!!")
                            vehicle_go()
                            # ser.write(b'stoop\n')
                        # print(f"The minimum distance among all detected objects is: {min_distance:.2f} meters")
                    else:
                        print("No objects detected.")
                        vehicle_go()
                       

                # Stream results
                cv2.imshow("Recognition result", im0)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Display images for debugging
            cv2.imshow("Lane Masked Image", lane_image)
            # cv2.imshow("Obstacle Masked Image", im0_masked)

            # Check if any of the images are empty
            if lane_masked_image.size == 0:
                print("Error: Lane Masked Image is empty")
            if im0_masked.size == 0:
                print("Error: Obstacle Masked Image is empty")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # cleanup_resources(pipeline)
        cv2.destroyAllWindows()
    
    # board.exit()


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

# -------------------- Visualization Setup --------------------
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
GLASBEY = torch.tensor(GLASBEY) / 255.0  # Normalize colors to [0,1] range

# Function to display annotated masks using GPU acceleration
def fast_show_mask_gpu(annotation):
    mask_sum = annotation.shape[0]
    height, width = annotation.shape[1], annotation.shape[2]
    areas = torch.sum(annotation, dim=(1, 2))
    sorted_indices = torch.argsort(areas, descending=False)
    annotation = annotation[sorted_indices]
    index = (annotation != 0).to(torch.long).argmax(dim=0)
    color = GLASBEY[:mask_sum].reshape(mask_sum, 1, 1, 3).to(annotation.device)
    transparency = torch.ones((mask_sum, 1, 1, 1), device=annotation.device) * 0.5
    visual = torch.cat([color, transparency], dim=-1)
    mask_image = torch.unsqueeze(annotation, -1) * visual
    show = torch.zeros((height, width, 4), device=annotation.device)
    h_indices, w_indices = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')
    indices = (index[h_indices, w_indices], h_indices, w_indices, slice(None))
    show[h_indices, w_indices, :] = mask_image[indices]
    show_cpu = show.cpu().numpy()
    return show_cpu

# -------------------- Model and ROI Setup --------------------
# Load the YOLO model with the specified weights
model = YOLO('E:/Autonomous_vehicle_project/lane/best.pt')  # Update with actual path

# Define the trapezoidal region of interest (ROI) vertices
trap_vertices = np.array([[100, 480], [540, 480], [420, 300], [220, 300]], dtype=np.int32)
roi_mask = np.zeros((480, 640), dtype=np.uint8)
cv2.fillPoly(roi_mask, [trap_vertices], 255)

# -------------------- PID Controller --------------------
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

# -------------------- Threaded Lane Detection --------------------
class LaneDetectionThread(threading.Thread):
    def __init__(self, model, pid_controller, roi_mask,sensor_data, camera_index=0):
        super().__init__()
        self.sensor_data = sensor_data
        self.model = model
        self.pid_controller = pid_controller
        self.roi_mask = roi_mask
        self.cap = cv2.VideoCapture(camera_index)
        self._stop_event = threading.Event()

    def run(self):
        cv2.namedWindow("Lane Detection", cv2.WINDOW_NORMAL)
        while not self._stop_event.is_set():
            start_time = time.time()
            success, frame = self.cap.read()
            if not success:
                print("Camera frame not received. Exiting lane detection loop.")
                break

            print("Frame captured. Running YOLO model...")
            results = self.model(frame)

            # Check if any masks were detected
            if results[0].masks is None or results[0].masks.data is None:
                print("No masks detected in this frame. Continuing to next frame...")
                continue

            annotated_frame = fast_show_mask_gpu(results[0].masks.data)
            # Resize annotated frame to match the original frame dimensions
            annotated_frame = cv2.resize(annotated_frame, (frame.shape[1], frame.shape[0]))

            # Extract the alpha channel and prepare for alpha blending
            alpha = annotated_frame[:, :, 3]
            alpha = cv2.resize(alpha, (frame.shape[1], frame.shape[0]))
            alpha_3_channel = np.repeat(alpha[:, :, np.newaxis], 3, axis=2)
            annotated_frame_rgb = annotated_frame[:, :, :3]
            annotated_frame = (annotated_frame_rgb * 255 * alpha_3_channel + frame * (1 - alpha_3_channel)) / 255
            current_heading = self.sensor_data.heading
            current_abs_heading = self.sensor_data.absolute_heading
            heading_error = (current_heading - current_abs_heading) / 360
            # Process lane mask for PID control
            lane_mask = results[0].masks.data[0].cpu().numpy()
            lane_mask = (lane_mask * 255).astype(np.uint8)
            resized_roi = cv2.resize(self.roi_mask, (lane_mask.shape[1], lane_mask.shape[0]))
            combined_mask = cv2.bitwise_and(lane_mask, lane_mask, mask=resized_roi)

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
                error_normalised = average_offset / (width / 2) + 0.5 * heading_error
            else:
                error_normalised = 0  # No lane detected

            delta_time = time.time() - start_time
            pwm_value = self.pid_controller.calculate(error_normalised, delta_time)
            send_pwm(pwm_value)

            # Display error and PWM on the frame
            cv2.putText(annotated_frame, f"Error: {error_normalised:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(annotated_frame, f"PWM: {pwm_value:.2f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imshow("Lane Detection", annotated_frame)

            # Check for the quit signal (press "q" to stop)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                self.stop()

        # Cleanup resources
        self.cap.release()
        cv2.destroyAllWindows()

    def stop(self):
        self._stop_event.set()

class SensorData:
    def __init__(self):
        self.lock = Lock()
        self._heading = 0.0
        self._absolute_heading = 0.0

    @property
    def heading(self):
        with self.lock:
            return self._heading

    @heading.setter
    def heading(self, value):
        with self.lock:
            self._heading = value

    @property
    def absolute_heading(self):
        with self.lock:
            return self._absolute_heading

    @absolute_heading.setter
    def absolute_heading(self, value):
        with self.lock:
            self._absolute_heading = value

shared_sensor_data = SensorData()

if __name__ == '__main__':
    # Parse arguments for YOLOv7
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.45, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='device to run on: cuda or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()

    # Set specific classes to detect
    classes_to_detect = [0, 1, 2, 3]  # Modify this list to include desired classes
    opt.classes = classes_to_detect
    lane_detection_thread = LaneDetectionThread(model=model,
                                                pid_controller=pid_controller,
                                                roi_mask=roi_mask,
                                                sensor_data=shared_sensor_data,
                                                camera_index=0)

    # Start Tkinter GUI for GPS navigation
    gui_thread = threading.Thread(target=start_gui, daemon=True)
    serial_thread = threading.Thread(target=read_serial,daemon = True)
    detection_thread = threading.Thread(target=detection, daemon=True)

    # Launch threads
    lane_detection_thread.start()
    gui_thread.start()
    detection_thread.start()
    serial_thread.start()


    # Wait for threads to complete
    gui_thread.join()
    lane_detection_thread.join()
    detection_thread.join()
    serial_thread.join()
