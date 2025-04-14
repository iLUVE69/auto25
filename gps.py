import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
import threading
import serial
import time
import re
from shapely.geometry import Point
import pyproj

# Serial port configuration for Arduino
SERIAL_PORT = 'COM8'  # Replace with your port (e.g., COM3 for Windows, /dev/ttyUSB0 for Linux)
BAUD_RATE = 9600

# Load your graph from the .osm file
osm_file_path = "C:\C++\iit2.osm"
graph = ox.graph_from_xml(osm_file_path)

# Assign custom names to each node
# Ensure node 'name' attributes from the OSM file are preserved (if they exist)
for node, data in graph.nodes(data=True):
    if 'name' in data:
        print(f"Node ID: {node}, Name: {data['name']}")



# Global variables to store GPS position, selected nodes, and the current position marker
current_lat, current_lon = None, None
selected_nodes = []
current_position_marker = None
destination_coords = None  # Coordinates of the second selected node

destination_reached = False

# Regular expression pattern to match the output from Arduino
gps_pattern = re.compile(r"Latitude=\s([-+]?[0-9]*\.?[0-9]+)\sLongitude=\s([-+]?[0-9]*\.?[0-9]+)")

# Get the projection used by the graph
graph_crs = graph.graph['crs']
projector = pyproj.Transformer.from_crs("epsg:4326", graph_crs, always_xy=True)

# Function to calculate the error between current GPS location and the destination node
def calculate_position_error(current_lat, current_lon, destination_lat, destination_lon):
    # Calculate the latitude and longitude error
    latitude_error = abs(destination_lat - current_lat)
    longitude_error = abs(destination_lon - current_lon)
    
    return latitude_error, longitude_error

# Function to read GPS data from the Arduino
def read_gps_data():
    global current_lat, current_lon, destination_coords, destination_reached
    try:
        print("Attempting to connect to the serial port...")
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
        print("Connected to the serial port successfully.")

        while True:
            if destination_reached:
                print("Vehicle already stopped. GPS updates paused.")
                break

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

                        # Check if the error is within the specified bounds
                        if 1e-5 < latitude_error < 1e-4 and 1e-5 < longitude_error < 1e-4:
                            #vehicle_stop()  # Call the test function
                            break

            time.sleep(0.1)  # Reduce delay for faster updates
    except serial.SerialException as e:
        print(f"Serial connection error: {e}")
    except KeyboardInterrupt:
        ser.close()
        print("\nSerial connection closed.")
    except Exception as e:
        print(f"Unexpected error: {e}")

# Function to update the position marker on the map
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

            # Update the canvas without losing the current zoom level
            canvas.draw()
            print(f"Position updated on the map: Latitude={current_lat}, Longitude={current_lon} (Projected to x={x}, y={y})")
    except Exception as e:
        print(f"Error updating position on map: {e}")

# Function to handle mouse clicks on the plot
def on_click(event):
    global destination_coords
    if len(selected_nodes) < 2:  # Allow selecting only two nodes
        try:
            # Find the nearest node to the click
            nearest_node = ox.distance.nearest_nodes(graph, event.xdata, event.ydata)
            selected_nodes.append(nearest_node)

            # Highlight the selected node
            ax.plot(event.xdata, event.ydata, marker='o', color='red', markersize=10)
            canvas.draw()

            print(f"Selected nodes: {selected_nodes}")

            if len(selected_nodes) == 2:
                # Once two nodes are selected, compute and display the shortest path using A*
                start_node, end_node = selected_nodes
                shortest_path = nx.astar_path(graph, start_node, end_node, weight='length')

                # Print the shortest path in terms of the custom node names
                print(f"\nShortest path between {graph.nodes[start_node]['name']} and {graph.nodes[end_node]['name']}:")
                for node in shortest_path:
                    print(graph.nodes[node]['name'])

                # Plot the shortest path
                ox.plot_graph_route(graph, shortest_path, route_linewidth=6, node_size=0, ax=ax)
                canvas.draw()

                # Set the destination coordinates to the latitude and longitude of the second selected node
                dest_node_data = graph.nodes[end_node]
                destination_coords = (dest_node_data['y'], dest_node_data['x'])
                print(f"Destination coordinates set to: Latitude={destination_coords[0]}, Longitude={destination_coords[1]}")
        except Exception as e:
            print(f"Error handling click event: {e}")

# Function to handle zooming with the mouse scroll
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

# Create the main window
root = tk.Tk()
root.title("Node Selector")

# Create a Matplotlib figure and plot the graph
fig, ax = plt.subplots()
ox.plot_graph(graph, ax=ax, show=False, close=True)

# Embed the Matplotlib figure in the Tkinter window
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

# Bind the click event to the on_click function
canvas.mpl_connect("button_press_event", on_click)

# Bind the scroll event to the on_scroll function
canvas.mpl_connect("scroll_event", on_scroll)

# Start a thread to read GPS data
gps_thread = threading.Thread(target=read_gps_data)
gps_thread.daemon = True
gps_thread.start()

# Start the Tkinter main loop
root.mainloop()
