import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import messagebox, filedialog
import os

class OSMShortestPathFinder:
    def __init__(self, root):
        self.root = root
        self.root.title("OSM Shortest Path Finder")
        self.root.geometry("1000x800")
        
        # Initialize variables
        self.G = None
        self.nodes = None
        self.start_node = None
        self.end_node = None
        self.selected_nodes = []
        self.path = None
        
        # Create GUI elements
        self.create_widgets()
        
    def create_widgets(self):
        # Frame for controls
        control_frame = tk.Frame(self.root)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)
        
        # Place entry
        tk.Label(control_frame, text="Place:").pack(side=tk.LEFT)
        self.place_entry = tk.Entry(control_frame, width=30)
        self.place_entry.pack(side=tk.LEFT, padx=5)
        self.place_entry.insert(0, "Berkeley, California, USA")
        
        # Load online map button
        load_online_button = tk.Button(control_frame, text="Load Online Map", command=self.load_online_map)
        load_online_button.pack(side=tk.LEFT, padx=5)
        
        # Load local file button
        load_file_button = tk.Button(control_frame, text="Load OSM File", command=self.load_osm_file)
        load_file_button.pack(side=tk.LEFT, padx=5)
        
        # Reset button
        reset_button = tk.Button(control_frame, text="Reset", command=self.reset)
        reset_button.pack(side=tk.LEFT, padx=5)
        
        # Status label
        self.status_label = tk.Label(control_frame, text="Enter a place name or load an OSM file")
        self.status_label.pack(side=tk.LEFT, padx=10)
        
        # Create matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        
        # Create canvas to display figure
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Connect click event
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
    
    def load_online_map(self):
        place = self.place_entry.get()
        if not place:
            messagebox.showerror("Error", "Please enter a place name")
            return
        
        try:
            self.status_label.config(text=f"Loading map for {place}...")
            self.root.update()
            
            # Get street network graph
            self.G = ox.graph_from_place(place, network_type='drive')
            
            # Project graph to use distance in meters
            self.G = ox.project_graph(self.G)
            
            # Get nodes and edges
            self.nodes, edges = ox.graph_to_gdfs(self.G)
            
            # Plot the graph
            self.reset_plot()
            
            self.status_label.config(text=f"Map loaded! Click to select start and end nodes")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load map: {str(e)}")
    
    def load_osm_file(self):
        # Open file dialog to select OSM file
        file_path = filedialog.askopenfilename(
            title="Select OSM File",
            filetypes=[("OSM Files", "*.osm"), ("XML Files", "*.xml"), ("All Files", "*.*")]
        )
        
        if not file_path:
            return  # User canceled
        
        try:
            self.status_label.config(text=f"Loading OSM file: {os.path.basename(file_path)}...")
            self.root.update()
            
            # Load graph from OSM file
            self.G = ox.graph_from_xml(file_path, simplify=True, retain_all=False)
            
            # Project graph to use distance in meters
            self.G = ox.project_graph(self.G)
            
            # Get nodes and edges
            self.nodes, edges = ox.graph_to_gdfs(self.G)
            
            # Plot the graph
            self.reset_plot()
            
            self.status_label.config(text=f"OSM file loaded! Click to select start and end nodes")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load OSM file: {str(e)}")
    
    def reset_plot(self):
        self.ax.clear()
        ox.plot_graph(self.G, ax=self.ax, node_size=5, node_color='blue', edge_color='gray')
        self.canvas.draw()
    
    def on_click(self, event):
        if event.inaxes != self.ax or self.G is None:
            return
        
        # Get the nearest node to the click
        x, y = event.xdata, event.ydata
        nearest_node = ox.distance.nearest_nodes(self.G, X=x, Y=y)
        
        if len(self.selected_nodes) < 2:
            # Add to selected nodes
            self.selected_nodes.append(nearest_node)
            node_type = "start" if len(self.selected_nodes) == 1 else "end"
            
            # Update UI
            node_coords = (self.nodes.loc[nearest_node]['x'], self.nodes.loc[nearest_node]['y'])
            self.ax.scatter(node_coords[0], node_coords[1], c='red' if node_type == "start" else "green", s=50, zorder=3)
            self.canvas.draw()
            
            self.status_label.config(text=f"Selected {node_type} node. {2-len(self.selected_nodes)} more to select.")
            
            # If both nodes are selected, find the path
            if len(self.selected_nodes) == 2:
                self.find_path()
        else:
            # If already have start and end, reset selections
            self.selected_nodes = [nearest_node]
            self.reset_plot()
            
            # Show the new start node
            node_coords = (self.nodes.loc[nearest_node]['x'], self.nodes.loc[nearest_node]['y'])
            self.ax.scatter(node_coords[0], node_coords[1], c='red', s=50, zorder=3)
            self.canvas.draw()
            
            self.status_label.config(text=f"Selected new start node. Select end node.")
    
    def find_path(self):
        start_node, end_node = self.selected_nodes
        
        try:
            # Find the shortest path
            path = nx.shortest_path(self.G, start_node, end_node, weight='length')
            
            # Plot the path
            path_edges = list(zip(path[:-1], path[1:]))
            ox.plot_graph_route(self.G, path, ax=self.ax, route_color='red', route_linewidth=4, node_size=5)
            self.canvas.draw()
            
            # Calculate distance
            path_length = int(sum(ox.utils_graph.get_route_edge_attributes(self.G, path, 'length')))
            self.status_label.config(text=f"Shortest path found! Length: {path_length} meters. Click to select new nodes.")
        except nx.NetworkXNoPath:
            messagebox.showerror("Error", "No path exists between the selected nodes")
            self.status_label.config(text="No path found. Select new nodes.")
    
    def reset(self):
        if self.G is not None:
            self.selected_nodes = []
            self.reset_plot()
            self.status_label.config(text="Map reset. Click to select nodes.")

if __name__ == "__main__":
    root = tk.Tk()
    app = OSMShortestPathFinder(root)
    root.mainloop()