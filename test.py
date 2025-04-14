import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
import math

# Load your graph from the .osm file
osm_file_path = r"C:\Users\ASUS\Downloads\iits.osm"
graph = ox.graph_from_xml(osm_file_path)

# Assign custom names to each node
for i, node in enumerate(graph.nodes()):
    graph.nodes[node]['name'] = f"CustomNode{i+1}"

# Global variables to store the selected nodes
selected_nodes = []

# Function to calculate the absolute heading (true bearing) between two points
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

    return bearing

# Function to handle mouse clicks on the plot
def on_click(event):
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
            shortest_path = nx.shortest_path(graph, start_node, end_node, weight='length')

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

# Function to handle zooming with the mouse scroll
def on_scroll(event):
    base_scale = 1.1
    cur_xlim = ax.get_xlim()
    cur_ylim = ax.get_ylim()

    xdata = event.xdata  # get event x location
    ydata = event.ydata  # get event y location

    if event.button == 'up':
        scale_factor = 1 / base_scale
    elif event.button == 'down':
        scale_factor = base_scale
    else:
        scale_factor = 1
        print(event.button)

    new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
    new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor

    relx = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0])
    rely = (cur_ylim[1] - ydata) / (cur_ylim[1] - cur_ylim[0])

    ax.set_xlim([xdata - new_width * (1 - relx), xdata + new_width * (relx)])
    ax.set_ylim([ydata - new_height * (1 - rely), ydata + new_height * (rely)])
    ax.figure.canvas.draw()

# Create the main window
root = tk.Tk()
root.title("Node Selector")

# Create a Matplotlib figure and plot the graph
fig, ax = plt.subplots()
ox.plot_graph(graph, ax=ax, show=False, close=False)

# Embed the Matplotlib figure in the Tkinter window
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

# Bind the click event to the on_click function
canvas.mpl_connect("button_press_event", on_click)

# Bind the scroll event to the on_scroll function
canvas.mpl_connect("scroll_event", on_scroll)

# Run the Tkinter event loop
root.mainloop()