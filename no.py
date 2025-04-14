import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
import math
import serial
import re
import threading
import time
from threading import Lock

# Load your graph from the .osm file
osm_file_path = r"C:\Users\ASUS\Downloads\iits.osm"
graph = ox.graph_from_xml(osm_file_path)

# Assign custom names to each node
for i, node in enumerate(graph.nodes()):
    graph.nodes[node]['name'] = f"CustomNode{i+1}"

# Global variables
selected_nodes = []
current_position_marker = None

class SensorData:
    def __init__(self):
        self.lock = Lock()
        self.heading = 0.0

shared_data = SensorData()

def read_serial():
    ser = serial.Serial('COM3', 38400, timeout=1)
    heading_pattern = re.compile(r'HEADING=([-\d.]+)')
    
    while True:
        try:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            print(f"Raw Serial Data: {line}")  # Debugging print
            
            if match := heading_pattern.search(line):
                heading = float(match.group(1)) % 360
                print(f"Extracted Heading: {heading}")  # Debugging print
                
                with shared_data.lock:
                    shared_data.heading = heading
                    
                # Update heading label in the main GUI thread
                root.after(0, update_heading_label, heading)  # Use root.after() to update GUI safely
        except Exception as e:
            print(f"Serial error: {e}")
        time.sleep(0.1)

def update_heading_label(heading):
    heading_label.config(text=f"Heading: {heading:.2f}°")

def on_click(event):
    global selected_nodes, current_position_marker
    if event.inaxes != ax:
        return
    
    node = ox.distance.nearest_nodes(graph, event.xdata, event.ydata)
    selected_nodes.append(node)
    
    # Update position marker
    if current_position_marker:
        current_position_marker[0].remove()
    x, y = graph.nodes[node]['x'], graph.nodes[node]['y']
    current_position_marker = ax.plot(x, y, 'bo', markersize=10)
    
    if len(selected_nodes) == 2:
        try:
            path = nx.shortest_path(graph, *selected_nodes, weight='length')
            ox.plot_graph_route(graph, path, route_linewidth=6, ax=ax)
            print(f"Path calculated between {selected_nodes}")
        except nx.NetworkXNoPath:
            print("No path found!")
        selected_nodes = []
    
    canvas.draw()

def on_scroll(event):
    base_scale = 1.1
    scale_factor = 1/base_scale if event.button == 'up' else base_scale
    
    x = event.xdata
    y = event.ydata
    
    cur_xlim = ax.get_xlim()
    cur_ylim = ax.get_ylim()
    
    ax.set_xlim([x - (x - cur_xlim[0])*scale_factor, 
                x + (cur_xlim[1] - x)*scale_factor])
    ax.set_ylim([y - (y - cur_ylim[0])*scale_factor, 
                y + (cur_ylim[1] - y)*scale_factor])
    
    canvas.draw()

# Initialize GUI
root = tk.Tk()
root.title("Navigation Map")

# Create heading display
heading_label = tk.Label(root, text="Heading: --.--°", font=('Arial', 14))
heading_label.pack(side=tk.TOP, fill=tk.X)

# Create map display
fig, ax = plt.subplots()
ox.plot_graph(graph, ax=ax, show=False, close=False, node_size=0)
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# Start serial thread
serial_thread = threading.Thread(target=read_serial, daemon=True)
serial_thread.start()

# Bind events
canvas.mpl_connect('button_press_event', on_click)
canvas.mpl_connect('scroll_event', on_scroll)

root.mainloop()