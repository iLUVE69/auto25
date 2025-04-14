import osmnx as ox
from geopy.distance import great_circle

def build_undirected_graph(osm_path):
    G = ox.graph_from_xml(osm_path)
    
    node_coords = {n: (data['y'], data['x']) for n, data in G.nodes(data=True)}
    adj_list = {}
    
    for u, v, data in G.edges(data=True):
        distance = data.get('length', great_circle(node_coords[u], node_coords[v]).meters)
        
        adj_list.setdefault(u, {})[v] = distance
        adj_list.setdefault(v, {})[u] = distance
    
    return adj_list, node_coords

osm_file = r"C:\Users\ASUS\Downloads\iits.osm"
adjacency_list, coordinates = build_undirected_graph(osm_file)

#prints the nodes and their connections with the distance
'''
for i, node in enumerate(adjacency_list):
    print(f"\nNode {node}  connects to:")
    for neighbor, dist in list(adjacency_list[node].items())[:]:  # First 3 connections per node
        print(f"  - Node {neighbor}: {dist:.5f}m")
'''

print(adjacency_list)