#!/usr/bin/env python3
"""
Generate benchmark dataset from OCTA500 vessel network dataset.

Dataset characteristics:
- 500 images at 400×400 resolution
- Split: 300 train, 100 validation, 100 test (by image)
- Extract overlapping 256×256 patches from each image
- Extract graphs from segmentation masks using skeletonization
"""

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import cv2
from tqdm import tqdm
import networkx as nx
from skimage.morphology import skeletonize
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from functools import partial


def _merge_close_nodes(key_nodes: List[int], skeleton_graph: nx.Graph, min_distance: float = 8.0) -> Tuple[List[int], Dict[int, int]]:
    """
    Merge key nodes that are too close together spatially.
    
    Args:
        key_nodes: List of key node indices in skeleton_graph
        skeleton_graph: The skeleton graph
        min_distance: Minimum distance between nodes (pixels)
    
    Returns:
        Tuple of (merged_key_nodes, node_mapping) where node_mapping maps old node indices to new merged indices
    """
    if len(key_nodes) < 2:
        return key_nodes, {node: 0 for node in key_nodes}
    
    # Get coordinates for all key nodes
    node_coords = {}
    for node in key_nodes:
        node_coords[node] = (skeleton_graph.nodes[node]['row'], skeleton_graph.nodes[node]['col'])
    
    # Build distance matrix and merge close nodes
    merged_nodes = []
    node_mapping = {}  # Maps original node -> merged node index
    used = set()
    
    for i, node1 in enumerate(key_nodes):
        if node1 in used:
            continue
        
        # Start a new merged group with this node
        merged_group = [node1]
        used.add(node1)
        node_mapping[node1] = len(merged_nodes)
        
        # Find all nodes close to this one
        row1, col1 = node_coords[node1]
        for node2 in key_nodes:
            if node2 in used or node2 == node1:
                continue
            
            row2, col2 = node_coords[node2]
            dist = np.sqrt((row1 - row2)**2 + (col1 - col2)**2)
            
            if dist < min_distance:
                merged_group.append(node2)
                used.add(node2)
                node_mapping[node2] = len(merged_nodes)
        
        # Use the first node in the group as the representative
        merged_nodes.append(merged_group[0])
    
    return merged_nodes, node_mapping


def _extract_graph_from_mask_standalone(mask: np.ndarray, curvature_threshold: float = 160.0, min_node_distance: float = 8.0, min_edge_length: float = 10.0) -> nx.Graph:
    """
    Standalone function to extract tree graph from mask (for multiprocessing).
    This is a copy of the class method but at module level for pickling.
    """
    # Binarize mask
    binary_mask = (mask > 256).astype(np.uint8)
    
    # Skeletonize
    skel = skeletonize(binary_mask).astype(np.uint8)
    
    # Build full skeleton graph for path finding
    h, w = skel.shape
    idx_map = -np.ones_like(skel, dtype=np.int32)
    coords = np.column_stack(np.nonzero(skel))  # (y, x) = (row, col)
    
    if len(coords) == 0:
        return nx.Graph()
    
    for i, (y, x) in enumerate(coords):
        idx_map[y, x] = i
    
    # Build full skeleton graph
    skeleton_graph = nx.Graph()
    for i, (y, x) in enumerate(coords):
        skeleton_graph.add_node(i, row=int(y), col=int(x))
    
    # 8-connectivity neighbors
    nbrs = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),          (0, 1),
        (1, -1),  (1, 0), (1, 1),
    ]
    
    for i, (y, x) in enumerate(coords):
        for dy, dx in nbrs:
            ny, nx_ = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx_ < w and idx_map[ny, nx_] >= 0:
                j = int(idx_map[ny, nx_])
                if i != j:
                    skeleton_graph.add_edge(i, j)
    
    # Find key nodes: endpoints (degree 1) and junctions (degree >= 3)
    key_nodes_raw = []
    for node in skeleton_graph.nodes():
        degree = skeleton_graph.degree(node)
        if degree == 1 or degree >= 3:
            key_nodes_raw.append(node)
    
    if len(key_nodes_raw) < 2:
        if len(key_nodes_raw) == 1:
            G = nx.Graph()
            node = key_nodes_raw[0]
            G.add_node(0, row=skeleton_graph.nodes[node]['row'], 
                      col=skeleton_graph.nodes[node]['col'])
            return G
        return nx.Graph()
    
    # Merge nodes that are too close together
    key_nodes, node_mapping = _merge_close_nodes(key_nodes_raw, skeleton_graph, min_node_distance)
    
    if len(key_nodes) < 2:
        if len(key_nodes) == 1:
            G = nx.Graph()
            node = key_nodes[0]
            G.add_node(0, row=skeleton_graph.nodes[node]['row'], 
                      col=skeleton_graph.nodes[node]['col'])
            return G
        return nx.Graph()
    
    # Build tree graph by connecting merged key nodes
    tree_graph = nx.Graph()
    
    # Add all merged key nodes to tree graph
    for i, node in enumerate(key_nodes):
        tree_graph.add_node(i, row=skeleton_graph.nodes[node]['row'],
                           col=skeleton_graph.nodes[node]['col'])
    
    # Connect merged key nodes that have paths in the skeleton
    # For each pair, check if any of the original nodes in their groups have a DIRECT path
    # (without going through any other key nodes)
    for i, merged_node1 in enumerate(key_nodes):
        for j, merged_node2 in enumerate(key_nodes):
            if i >= j:
                continue
            
            # Get all original nodes in each merged group
            group1_nodes = [n for n, mapped_idx in node_mapping.items() if mapped_idx == i]
            group2_nodes = [n for n, mapped_idx in node_mapping.items() if mapped_idx == j]
            
            # Check if any node in group1 has a DIRECT path to any node in group2
            # Direct path means: path exists AND doesn't go through ANY other key nodes
            has_direct_path = False
            best_path = None
            best_path_length = float('inf')
            
            for n1 in group1_nodes:
                for n2 in group2_nodes:
                    try:
                        if nx.has_path(skeleton_graph, n1, n2):
                            path = nx.shortest_path(skeleton_graph, n1, n2)
                            
                            # Check if path doesn't go through ANY other key nodes (not just merged ones)
                            # This ensures we only connect nodes that are directly connected
                            intermediate_key_nodes = [n for n in path[1:-1] if n in key_nodes_raw]
                            
                            if len(intermediate_key_nodes) == 0:
                                # Valid direct path found
                                path_length = len(path)
                                if path_length < best_path_length:
                                    best_path = path
                                    best_path_length = path_length
                                    has_direct_path = True
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        continue
            
            # Only add edge if we found a valid direct path
            if has_direct_path:
                tree_graph.add_edge(i, j)
    
    # Ensure it's a tree (remove cycles if any)
    # Convert each connected component to a minimum spanning tree
    for u, v in list(tree_graph.edges()):
        row1, col1 = tree_graph.nodes[u]['row'], tree_graph.nodes[u]['col']
        row2, col2 = tree_graph.nodes[v]['row'], tree_graph.nodes[v]['col']
        dist = np.sqrt((row1 - row2)**2 + (col1 - col2)**2)
        tree_graph[u][v]['weight'] = dist
    
    # Get MST for each connected component
    mst_components = []
    for component in nx.connected_components(tree_graph):
        subgraph = tree_graph.subgraph(component).copy()
        if len(subgraph.nodes()) > 1:
            for u, v in list(subgraph.edges()):
                if 'weight' not in subgraph[u][v]:
                    row1, col1 = subgraph.nodes[u]['row'], subgraph.nodes[u]['col']
                    row2, col2 = subgraph.nodes[v]['row'], subgraph.nodes[v]['col']
                    dist = np.sqrt((row1 - row2)**2 + (col1 - col2)**2)
                    subgraph[u][v]['weight'] = dist
            mst = nx.minimum_spanning_tree(subgraph)
            mst_components.append(mst)
        elif len(subgraph.nodes()) == 1:
            mst_components.append(subgraph)
    
    # Combine all MST components into one graph
    tree_graph = nx.Graph()
    node_counter = 0
    for mst in mst_components:
        comp_node_mapping = {}
        for old_node in mst.nodes():
            new_node = node_counter
            comp_node_mapping[old_node] = new_node
            tree_graph.add_node(new_node, **mst.nodes[old_node])
            node_counter += 1
        for u, v in mst.edges():
            tree_graph.add_edge(comp_node_mapping[u], comp_node_mapping[v])
    
    # Filter out edges that are too short
    edges_to_remove = []
    for u, v in tree_graph.edges():
        row1, col1 = tree_graph.nodes[u]['row'], tree_graph.nodes[u]['col']
        row2, col2 = tree_graph.nodes[v]['row'], tree_graph.nodes[v]['col']
        dist = np.sqrt((row1 - row2)**2 + (col1 - col2)**2)
        if dist < min_edge_length:
            edges_to_remove.append((u, v))
    
    for u, v in edges_to_remove:
        tree_graph.remove_edge(u, v)
    
    # Remove isolated nodes (nodes with no edges)
    isolated_nodes = [n for n in tree_graph.nodes() if tree_graph.degree(n) == 0]
    tree_graph.remove_nodes_from(isolated_nodes)
    
    # Relabel nodes to 0..N-1 for consistency
    nodes = sorted(tree_graph.nodes())
    if len(nodes) == 0:
        return nx.Graph()
    
    mapping = {old: i for i, old in enumerate(nodes)}
    tree_graph_relabeled = nx.Graph()
    
    for old_node in nodes:
        new_node = mapping[old_node]
        tree_graph_relabeled.add_node(new_node, **tree_graph.nodes[old_node])
    
    for u, v in tree_graph.edges():
        tree_graph_relabeled.add_edge(mapping[u], mapping[v])
    
    return tree_graph_relabeled


def _extract_patches_from_image(args):
    """
    Standalone function for multiprocessing to extract patches from a single image.
    This function must be at module level to be picklable.
    """
    image_id, image_path, mask_path, patch_size, output_size, stride, curvature_threshold, min_node_distance, min_edge_length = args
    
    # Load image
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        return []
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    # Load mask
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return []
    
    patches = []
    h, w = image_rgb.shape[:2]
    
    # Generate patch coordinates
    for y_start in range(0, h - patch_size + 1, stride):
        for x_start in range(0, w - patch_size + 1, stride):
            # Extract image patch (64x64)
            patch_image = image_rgb[y_start:y_start+patch_size, x_start:x_start+patch_size]
            
            # Extract mask patch (64x64)
            patch_mask = mask[y_start:y_start+patch_size, x_start:x_start+patch_size]
            
            # Extract graph directly from mask patch using standalone function
            # Use min_node_distance based on patch size (smaller patches need smaller distance)
            min_node_distance = max(5.0, patch_size / 32.0)  # e.g., 5px for 64x64
            patch_graph = _extract_graph_from_mask_standalone(patch_mask, curvature_threshold, min_node_distance)
            
            if patch_graph is None or len(patch_graph.nodes()) < 2:
                continue  # Skip patches with insufficient graph data
            
            # Resize image and mask to output_size (256x256)
            patch_image_resized = cv2.resize(patch_image, (output_size, output_size), interpolation=cv2.INTER_LINEAR)
            patch_mask_resized = cv2.resize(patch_mask, (output_size, output_size), interpolation=cv2.INTER_NEAREST)
            
            # Scale graph coordinates to match resized image
            scale_factor = output_size / patch_size  # e.g., 256/64 = 2.0
            patch_graph_scaled = patch_graph.copy()
            for node in patch_graph_scaled.nodes():
                row = patch_graph_scaled.nodes[node]['row']
                col = patch_graph_scaled.nodes[node]['col']
                patch_graph_scaled.nodes[node]['row'] = int(row * scale_factor)
                patch_graph_scaled.nodes[node]['col'] = int(col * scale_factor)
            
            patches.append({
                'image_id': image_id,
                'patch_coords': (x_start, y_start),
                'image': patch_image_resized,  # Already resized to 256x256
                'mask': patch_mask_resized,  # Already resized to 256x256
                'graph': patch_graph_scaled  # Coordinates scaled to 256x256
            })
    
    return patches


class OCTA500BenchmarkDatasetGenerator:
    def __init__(
        self,
        raw_data_path: str,
        output_path: str,
        patch_size: int = 64,  # Extract patches at this size
        output_size: int = 256,  # Resize patches to this size
        overlap: int = 16,  # 25% overlap (64 * 0.25 = 16)
        max_samples_per_split: Optional[int] = None,
        max_nodes: Optional[int] = None,  # Filter out samples with more nodes than this
    ):
        self.raw_data_path = Path(raw_data_path)
        self.output_path = Path(output_path)
        self.patch_size = patch_size  # Extraction size (64x64)
        self.output_size = output_size  # Output size after resize (256x256)
        self.overlap = overlap
        self.stride = patch_size - overlap
        self.max_samples_per_split = max_samples_per_split
        self.max_nodes = max_nodes
        self.curvature_threshold = 160.0  # degrees - prune degree-2 nodes with curvature >= this
        self.min_edge_length = 10.0  # pixels - minimum edge length to keep
        
        # Create output directories
        self.output_path.mkdir(parents=True, exist_ok=True)
    
    def extract_graph_from_mask(self, mask: np.ndarray) -> nx.Graph:
        """
        Extract tree graph from binary segmentation mask using skeletonization.
        Key nodes (endpoints and junctions) are extracted and connected with straight edges.
        
        Args:
            mask: Binary mask (0 = background, 255 = vessel)
        
        Returns:
            NetworkX tree graph with key nodes only, connected by straight edges
        """
        # Binarize mask
        binary_mask = (mask > 256).astype(np.uint8)
        
        # Skeletonize
        skel = skeletonize(binary_mask).astype(np.uint8)
        
        # Build full skeleton graph for path finding
        h, w = skel.shape
        idx_map = -np.ones_like(skel, dtype=np.int32)
        coords = np.column_stack(np.nonzero(skel))  # (y, x) = (row, col)
        
        if len(coords) == 0:
            return nx.Graph()
        
        for i, (y, x) in enumerate(coords):
            idx_map[y, x] = i
        
        # Build full skeleton graph
        skeleton_graph = nx.Graph()
        for i, (y, x) in enumerate(coords):
            skeleton_graph.add_node(i, row=int(y), col=int(x))
        
        # 8-connectivity neighbors
        nbrs = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),          (0, 1),
            (1, -1),  (1, 0), (1, 1),
        ]
        
        for i, (y, x) in enumerate(coords):
            for dy, dx in nbrs:
                ny, nx_ = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx_ < w and idx_map[ny, nx_] >= 0:
                    j = int(idx_map[ny, nx_])
                    if i != j:
                        skeleton_graph.add_edge(i, j)
        
        # Find key nodes: endpoints (degree 1) and junctions (degree >= 3)
        key_nodes = []
        for node in skeleton_graph.nodes():
            degree = skeleton_graph.degree(node)
            if degree == 1 or degree >= 3:
                key_nodes.append(node)
        
        if len(key_nodes) < 2:
            # Not enough key nodes, return empty or minimal graph
            if len(key_nodes) == 1:
                G = nx.Graph()
                node = key_nodes[0]
                G.add_node(0, row=skeleton_graph.nodes[node]['row'], 
                          col=skeleton_graph.nodes[node]['col'])
                return G
            return nx.Graph()
        
        # Build tree graph by connecting key nodes
        # For each pair of key nodes, check if there's a path in the skeleton
        tree_graph = nx.Graph()
        
        # Add all key nodes to tree graph
        for i, node in enumerate(key_nodes):
            tree_graph.add_node(i, row=skeleton_graph.nodes[node]['row'],
                               col=skeleton_graph.nodes[node]['col'])
        
        # Connect key nodes that have DIRECT paths in the skeleton
        # Direct path means: path exists AND doesn't go through ANY other key nodes
        for i, node1 in enumerate(key_nodes):
            for j, node2 in enumerate(key_nodes):
                if i >= j:
                    continue
                
                # Check if there's a DIRECT path between these key nodes
                try:
                    if nx.has_path(skeleton_graph, node1, node2):
                        path = nx.shortest_path(skeleton_graph, node1, node2)
                        # Check if path doesn't go through ANY other key nodes
                        # This ensures we only connect nodes that are directly connected
                        # Exclude the endpoints (node1 and node2) from the check
                        intermediate_key_nodes = [n for n in path[1:-1] if n in key_nodes]
                        if len(intermediate_key_nodes) == 0:
                            # Valid direct path found, add edge
                            tree_graph.add_edge(i, j)
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    continue
        
        # Filter out edges that are too short
        edges_to_remove = []
        for u, v in list(tree_graph.edges()):
            row1, col1 = tree_graph.nodes[u]['row'], tree_graph.nodes[u]['col']
            row2, col2 = tree_graph.nodes[v]['row'], tree_graph.nodes[v]['col']
            dist = np.sqrt((row1 - row2)**2 + (col1 - col2)**2)
            if dist < self.min_edge_length:
                edges_to_remove.append((u, v))
        
        for u, v in edges_to_remove:
            tree_graph.remove_edge(u, v)
        
        # Remove isolated nodes (nodes with no edges)
        isolated_nodes = [n for n in tree_graph.nodes() if tree_graph.degree(n) == 0]
        tree_graph.remove_nodes_from(isolated_nodes)
        
        # Ensure it's a tree (remove cycles if any)
        # Convert each connected component to a minimum spanning tree
        # Compute edge weights (Euclidean distance)
        for u, v in list(tree_graph.edges()):
            row1, col1 = tree_graph.nodes[u]['row'], tree_graph.nodes[u]['col']
            row2, col2 = tree_graph.nodes[v]['row'], tree_graph.nodes[v]['col']
            dist = np.sqrt((row1 - row2)**2 + (col1 - col2)**2)
            tree_graph[u][v]['weight'] = dist
        
        # Get MST for each connected component
        mst_components = []
        for component in nx.connected_components(tree_graph):
            subgraph = tree_graph.subgraph(component).copy()
            if len(subgraph.nodes()) > 1:
                # Ensure weights are set
                for u, v in subgraph.edges():
                    if 'weight' not in subgraph[u][v]:
                        row1, col1 = subgraph.nodes[u]['row'], subgraph.nodes[u]['col']
                        row2, col2 = subgraph.nodes[v]['row'], subgraph.nodes[v]['col']
                        dist = np.sqrt((row1 - row2)**2 + (col1 - col2)**2)
                        subgraph[u][v]['weight'] = dist
                mst = nx.minimum_spanning_tree(subgraph)
                mst_components.append(mst)
            elif len(subgraph.nodes()) == 1:
                # Single node component, keep it
                mst_components.append(subgraph)
        
        # Combine all MST components into one graph
        # Use fresh node indices to avoid collisions
        tree_graph = nx.Graph()
        node_counter = 0
        node_mapping = {}  # Maps (component_idx, old_node) -> new_node
        
        for comp_idx, mst in enumerate(mst_components):
            comp_node_mapping = {}
            for old_node in mst.nodes():
                new_node = node_counter
                comp_node_mapping[old_node] = new_node
                tree_graph.add_node(new_node, **mst.nodes[old_node])
                node_counter += 1
            
            for u, v in mst.edges():
                tree_graph.add_edge(comp_node_mapping[u], comp_node_mapping[v])
        
        # Relabel nodes to 0..N-1 for consistency
        nodes = sorted(tree_graph.nodes())
        mapping = {old: i for i, old in enumerate(nodes)}
        tree_graph_relabeled = nx.Graph()
        
        for old_node in nodes:
            new_node = mapping[old_node]
            tree_graph_relabeled.add_node(new_node, **tree_graph.nodes[old_node])
        
        for u, v in tree_graph.edges():
            tree_graph_relabeled.add_edge(mapping[u], mapping[v])
        
        return tree_graph_relabeled
    
    def compute_angle(self, p1: Tuple[int, int], p2: Tuple[int, int], p3: Tuple[int, int]) -> float:
        """Compute angle at p2 formed by p1-p2-p3 in degrees."""
        # Vectors
        v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
        
        # Compute angle using dot product
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        
        return np.degrees(angle)
    
    def simplify_graph_by_curvature(self, G: nx.Graph) -> nx.Graph:
        """
        Simplify graph by removing degree-2 nodes with curvature >= threshold.
        Following Belli et al., we compute the angle between segments at degree-2 nodes.
        Only keep nodes if curvature < threshold (i.e., remove straight paths).
        
        Args:
            G: NetworkX graph
        
        Returns:
            Simplified NetworkX graph
        """
        if len(G.nodes()) < 3:
            return G  # Need at least 3 nodes to compute angles
        
        # Identify degree-2 nodes to potentially remove
        nodes_to_remove = set()
        for node in G.nodes():
            if G.degree(node) == 2:
                neighbors = list(G.neighbors(node))
                if len(neighbors) == 2:
                    # Get coordinates
                    p1 = (G.nodes[neighbors[0]]['row'], G.nodes[neighbors[0]]['col'])
                    p2 = (G.nodes[node]['row'], G.nodes[node]['col'])
                    p3 = (G.nodes[neighbors[1]]['row'], G.nodes[neighbors[1]]['col'])
                    
                    # Compute angle at this node
                    angle = self.compute_angle(p1, p2, p3)
                    
                    # If angle >= threshold (straight line), remove this node
                    if angle >= self.curvature_threshold:
                        nodes_to_remove.add(node)
        
        if not nodes_to_remove:
            return G  # No nodes to remove
        
        # Build simplified graph - iterate until no more nodes can be removed
        H = G.copy()
        changed = True
        
        while changed:
            changed = False
            # Find degree-2 nodes to remove in current iteration
            current_nodes_to_remove = []
            for node in H.nodes():
                if H.degree(node) == 2:
                    neighbors = list(H.neighbors(node))
                    if len(neighbors) == 2:
                        # Get coordinates
                        p1 = (H.nodes[neighbors[0]]['row'], H.nodes[neighbors[0]]['col'])
                        p2 = (H.nodes[node]['row'], H.nodes[node]['col'])
                        p3 = (H.nodes[neighbors[1]]['row'], H.nodes[neighbors[1]]['col'])
                        
                        # Compute angle at this node
                        angle = self.compute_angle(p1, p2, p3)
                        
                        # If angle >= threshold (straight line), mark for removal
                        if angle >= self.curvature_threshold:
                            current_nodes_to_remove.append(node)
            
            # Remove nodes and reconnect neighbors
            for node in current_nodes_to_remove:
                if node in H and H.degree(node) == 2:  # Check still exists and is degree-2
                    neighbors = list(H.neighbors(node))
                    if len(neighbors) == 2:
                        # Connect neighbors directly
                        n1, n2 = neighbors[0], neighbors[1]
                        if not H.has_edge(n1, n2):
                            H.add_edge(n1, n2)
                        H.remove_node(node)
                        changed = True
        
        # Relabel nodes to 0..N-1 for consistency
        nodes = sorted(H.nodes())
        mapping = {old: i for i, old in enumerate(nodes)}
        H_relabeled = nx.Graph()
        
        for old_node in nodes:
            new_node = mapping[old_node]
            H_relabeled.add_node(new_node, **H.nodes[old_node])
        
        for u, v in H.edges():
            H_relabeled.add_edge(mapping[u], mapping[v])
        
        return H_relabeled
    
    def extract_patch_graph(self, full_graph: nx.Graph, x_start: int, y_start: int, patch_size: int) -> Optional[nx.Graph]:
        """
        Extract graph nodes within patch boundaries.
        
        Args:
            full_graph: Full graph from entire image
            x_start, y_start: Patch top-left corner in image coordinates
            patch_size: Size of patch
        
        Returns:
            Subgraph containing only nodes within patch, or None if insufficient nodes
        """
        x_end = x_start + patch_size
        y_end = y_start + patch_size
        
        patch_nodes = []
        for node in full_graph.nodes():
            # Graph stores (row, col) = (y, x)
            row, col = full_graph.nodes[node].get('row', 0), full_graph.nodes[node].get('col', 0)
            
            # Check if node is within patch bounds
            if y_start <= row < y_end and x_start <= col < x_end:
                patch_nodes.append(node)
        
        if len(patch_nodes) < 2:
            return None
        
        # Create subgraph
        patch_graph = full_graph.subgraph(patch_nodes).copy()
        
        # Adjust coordinates to patch-local coordinates
        for node in patch_graph.nodes():
            row = patch_graph.nodes[node].get('row', 0)
            col = patch_graph.nodes[node].get('col', 0)
            # Convert to patch-local coordinates
            patch_graph.nodes[node]['row'] = row - y_start
            patch_graph.nodes[node]['col'] = col - x_start
        
        return patch_graph
    
    def networkx_to_adjacency_matrix(self, G: nx.Graph) -> Tuple[np.ndarray, Dict[int, Tuple]]:
        """Convert NetworkX graph to adjacency matrix and node mapping.
        
        Returns:
            adj_matrix: Adjacency matrix
            idx_to_coords: Mapping from node index to coordinates (row, col) format
        """
        nodes = sorted(G.nodes())
        n = len(nodes)
        
        # Create node index mapping
        node_to_idx = {node: idx for idx, node in enumerate(nodes)}
        idx_to_coords = {}
        
        for idx, node in enumerate(nodes):
            row = G.nodes[node].get('row', 0)
            col = G.nodes[node].get('col', 0)
            idx_to_coords[idx] = (row, col)
        
        # Create adjacency matrix
        adj_matrix = np.zeros((n, n), dtype=np.float32)
        for u, v in G.edges():
            i, j = node_to_idx[u], node_to_idx[v]
            adj_matrix[i, j] = 1.0
            adj_matrix[j, i] = 1.0
        
        return adj_matrix, idx_to_coords
    
    def generate_masks_from_graph(self, G: nx.Graph, idx_to_coords: Dict, image_size: Tuple[int, int], gt_mask_patch: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate keypoint, edge, and skeleton masks.
        
        Args:
            G: NetworkX graph (full graph from mask, no simplification)
            idx_to_coords: Mapping from node index to (row, col) coordinates
            image_size: (height, width) of the patch
            gt_mask_patch: GT mask patch - use it directly as edge mask (original mask, no simplification)
        
        Note: Graph coordinates are (row, col) = (y, x), but cv2 functions expect (x, y) = (col, row).
        """
        h, w = image_size
        
        # Initialize masks
        keypoint_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Adaptive node radius based on image size
        if min(w, h) <= 64:
            node_radius = 2
        elif min(w, h) <= 256:
            node_radius = 3
        elif min(w, h) <= 256:
            node_radius = 5
        else:
            node_radius = 8
        
        # Draw ALL nodes from idx_to_coords to match adjacency matrix
        for idx, (row, col) in idx_to_coords.items():
            # Graph stores (row, col) = (y, x), cv2 expects (x, y) = (col, row)
            x, y = col, row  # Swap to (x, y) for cv2
            x, y = int(x), int(y)
            x = max(node_radius, min(w - 1 - node_radius, x))
            y = max(node_radius, min(h - 1 - node_radius, y))
            cv2.circle(keypoint_mask, (x, y), node_radius, 255, -1)
        
        # Use GT mask patch directly as edge mask (no simplification)
        if gt_mask_patch is not None:
            # Use original GT mask patch directly
            edge_mask = (gt_mask_patch > 256).astype(np.uint8) * 255
        else:
            # Fallback: generate from graph (shouldn't happen in normal flow)
            edge_mask = np.zeros((h, w), dtype=np.uint8)
            if min(w, h) <= 64:
                edge_width = 1
            elif min(w, h) <= 256:
                edge_width = 2
            elif min(w, h) <= 256:
                edge_width = 3
            else:
                edge_width = 4
            
            # Draw edges from graph
            for edge in G.edges():
                node1, node2 = edge
                row1, col1 = G.nodes[node1]['row'], G.nodes[node1]['col']
                row2, col2 = G.nodes[node2]['row'], G.nodes[node2]['col']
                x1, y1 = int(col1), int(row1)
                x2, y2 = int(col2), int(row2)
                cv2.line(edge_mask, (x1, y1), (x2, y2), 255, edge_width)
        
        # Generate skeleton from edge mask (original GT mask)
        skeleton_bool = skeletonize(edge_mask > 0)
        skeleton = (skeleton_bool * 255).astype(np.uint8)
        
        return keypoint_mask, edge_mask, skeleton
    
    def generate_topology_data(self, adj_matrix: np.ndarray, idx_to_coords: Dict, image_size: Tuple[int, int], max_pairs: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate topology data: graph_points, pairs, valid, connected.
        
        Note: idx_to_coords has graph format (row, col), but we store as (x, y) = (col, row) for training.
        """
        n_nodes = adj_matrix.shape[0]
        
        # Graph points - convert from (row, col) to (x, y) = (col, row)
        graph_points = np.zeros((n_nodes, 2), dtype=np.float32)
        for idx, (row, col) in idx_to_coords.items():
            graph_points[idx] = [col, row]  # Store as (x, y) = (col, row)
        
        # Determine max_pairs dynamically if not provided
        if max_pairs is None:
            max_neighbors = max([int(np.sum(adj_matrix[i])) for i in range(n_nodes)], default=0)
            max_pairs = max(32, max_neighbors + 10)
        
        # Pairs, valid, connected
        pairs = np.zeros((n_nodes, max_pairs, 2), dtype=np.int64)
        valid = np.zeros((n_nodes, max_pairs), dtype=np.float32)
        connected = np.zeros((n_nodes, max_pairs), dtype=np.float32)
        
        for i in range(n_nodes):
            neighbors = np.where(adj_matrix[i] > 0)[0]
            n_neighbors = len(neighbors)
            
            for j in range(min(n_neighbors, max_pairs)):
                neighbor_idx = neighbors[j]
                pairs[i, j] = [i, neighbor_idx]
                valid[i, j] = 1.0
                connected[i, j] = 1.0
            
            # Fill remaining with invalid pairs (pointing to self)
            for j in range(n_neighbors, max_pairs):
                pairs[i, j] = [i, i]
                valid[i, j] = 0.0
                connected[i, j] = 0.0
        
        return graph_points, pairs, valid, connected
    
    def extract_patches(self, image: np.ndarray, mask: np.ndarray, image_id: str) -> List[Dict]:
        """
        Extract overlapping patches from an image and extract graph from each mask patch.
        Patches are extracted at patch_size (64x64) and resized to output_size (256x256).
        This is much faster than extracting from the full image first.
        """
        patches = []
        h, w = image.shape[:2]
        
        # Generate patch coordinates
        for y_start in range(0, h - self.patch_size + 1, self.stride):
            for x_start in range(0, w - self.patch_size + 1, self.stride):
                # Extract image patch (64x64)
                patch_image = image[y_start:y_start+self.patch_size, x_start:x_start+self.patch_size]
                
                # Extract mask patch (64x64)
                patch_mask = mask[y_start:y_start+self.patch_size, x_start:x_start+self.patch_size]
                
                # Extract graph directly from mask patch (much faster!)
                patch_graph = self.extract_graph_from_mask(patch_mask)
                
                if patch_graph is None or len(patch_graph.nodes()) < 2:
                    continue  # Skip patches with insufficient graph data
                
                # Resize image and mask to output_size (256x256)
                patch_image_resized = cv2.resize(patch_image, (self.output_size, self.output_size), interpolation=cv2.INTER_LINEAR)
                patch_mask_resized = cv2.resize(patch_mask, (self.output_size, self.output_size), interpolation=cv2.INTER_NEAREST)
                
                # Scale graph coordinates to match resized image
                scale_factor = self.output_size / self.patch_size  # e.g., 256/64 = 2.0
                patch_graph_scaled = patch_graph.copy()
                for node in patch_graph_scaled.nodes():
                    row = patch_graph_scaled.nodes[node]['row']
                    col = patch_graph_scaled.nodes[node]['col']
                    patch_graph_scaled.nodes[node]['row'] = int(row * scale_factor)
                    patch_graph_scaled.nodes[node]['col'] = int(col * scale_factor)
                
                # Graph is already a tree with key nodes only
                
                patches.append({
                    'image_id': image_id,
                    'patch_coords': (x_start, y_start),
                    'image': patch_image_resized,  # Already resized to 256x256
                    'mask': patch_mask_resized,  # Already resized to 256x256
                    'graph': patch_graph_scaled  # Coordinates scaled to 256x256
                })
        
        return patches
    
    def _save_overlay_visualization(
        self, 
        image: np.ndarray, 
        node_mask: np.ndarray, 
        edge_mask: np.ndarray, 
        graph_points: np.ndarray,
        adj_matrix: np.ndarray,
        save_path: Path
    ):
        """Save overlay visualization showing image, masks, and graph structure."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        
        # Original image
        axes[0, 0].imshow(image)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Image + Edge mask
        axes[0, 1].imshow(image, alpha=0.7)
        axes[0, 1].imshow(edge_mask, cmap='gray', alpha=0.3)
        axes[0, 1].set_title('Image + Edge Mask')
        axes[0, 1].axis('off')
        
        # Image + Node mask
        axes[1, 0].imshow(image, alpha=0.7)
        axes[1, 0].imshow(node_mask, cmap='hot', alpha=0.5)
        axes[1, 0].set_title('Image + Node Mask')
        axes[1, 0].axis('off')
        
        # Full overlay: image + masks + graph structure
        axes[1, 1].imshow(image, alpha=0.7)
        axes[1, 1].imshow(edge_mask, cmap='gray', alpha=0.3)
        
        # Draw edges from adjacency matrix
        for i in range(len(graph_points)):
            for j in range(i+1, len(graph_points)):
                if adj_matrix[i, j] > 0:
                    x1, y1 = graph_points[i]
                    x2, y2 = graph_points[j]
                    axes[1, 1].plot([x1, x2], [y1, y2], 'b-', linewidth=2, alpha=0.7)
        
        # Draw nodes
        axes[1, 1].scatter(graph_points[:, 0], graph_points[:, 1], 
                       c='red', s=100, marker='o', edgecolors='yellow', 
                       linewidths=2, zorder=5)
        
        # Label nodes
        for i, pt in enumerate(graph_points):
            axes[1, 1].text(pt[0], pt[1], str(i), color='white', fontsize=8, 
                        ha='center', va='center', fontweight='bold', zorder=6)
        
        axes[1, 1].set_title(f'Full Overlay ({len(graph_points)} nodes, {int(np.sum(adj_matrix)/2)} edges)')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
    
    def generate_benchmark_dataset(self):
        """Generate the complete benchmark dataset."""
        # Get all image IDs (10001 to 10500)
        octa_dir = self.raw_data_path / 'OCTA'
        mask_dir = self.raw_data_path / 'GT_LargeVessel'
        
        all_image_ids = sorted([int(p.stem) for p in octa_dir.glob('*.bmp')])
        
        print(f"Found {len(all_image_ids)} images")
        
        # Split: 350 train, 50 val, 100 test
        train_ids = all_image_ids[:350]
        val_ids = all_image_ids[350:400]
        test_ids = all_image_ids[400:500]
        
        print(f"Train images: {len(train_ids)}")
        print(f"Val images: {len(val_ids)}")
        print(f"Test images: {len(test_ids)}")
        
        splits = {
            'train': train_ids,
            'val': val_ids,
            'test': test_ids
        }
        
        dataset_stats = {}
        
        for split_name, image_ids in splits.items():
            print(f"\n{'='*80}")
            print(f"Processing {split_name} split ({len(image_ids)} images)")
            print(f"{'='*80}")
            
            # Create split directory
            split_dir = self.output_path / split_name
            split_dir.mkdir(parents=True, exist_ok=True)
            (split_dir / 'images').mkdir(exist_ok=True)
            (split_dir / 'adjacency_matrices').mkdir(exist_ok=True)
            (split_dir / 'masks' / 'nodes').mkdir(parents=True, exist_ok=True)
            (split_dir / 'masks' / 'edges').mkdir(parents=True, exist_ok=True)
            (split_dir / 'masks' / 'skeleton').mkdir(parents=True, exist_ok=True)
            (split_dir / 'points').mkdir(exist_ok=True)
            (split_dir / 'visualizations').mkdir(exist_ok=True)
            
            all_patches = []
            
            # Prepare arguments for multiprocessing
            process_args = []
            for image_id in image_ids:
                image_path = octa_dir / f"{image_id}.bmp"
                mask_path = mask_dir / f"{image_id}.bmp"
                if image_path.exists() and mask_path.exists():
                    process_args.append((
                        image_id,
                        image_path,
                        mask_path,
                        self.patch_size,  # Extract at this size (64)
                        self.output_size,  # Resize to this size (256)
                        self.stride,
                        self.curvature_threshold,
                        None,  # min_node_distance will be computed per patch
                        None    # min_edge_length will be computed per patch
                    ))
            
            # Extract patches from all images in this split using multiprocessing
            num_workers = max(1, min(cpu_count() - 1, len(process_args)))  # Use available cores
            print(f"Using {num_workers} CPU cores for patch extraction...")
            
            with Pool(processes=num_workers) as pool:
                # Use imap for progress bar support
                patch_lists = list(tqdm(
                    pool.imap(_extract_patches_from_image, process_args),
                    total=len(process_args),
                    desc=f"Extracting patches from {split_name} images"
                ))
            
            # Flatten the list of patch lists
            for patches in patch_lists:
                all_patches.extend(patches)
            
            print(f"Total patches extracted: {len(all_patches)}")
            
            # Limit samples if requested
            if self.max_samples_per_split is not None:
                all_patches = all_patches[:self.max_samples_per_split]
                print(f"Limited to {len(all_patches)} patches")
            
            # Process and save patches
            metadata = []
            filtered_count = 0
            
            for idx, patch in enumerate(tqdm(all_patches, desc=f"Processing {split_name} patches")):
                # Generate adjacency matrix
                adj_matrix, idx_to_coords = self.networkx_to_adjacency_matrix(patch['graph'])
                
                # Filter: Skip samples with num_nodes > max_nodes
                num_nodes = adj_matrix.shape[0]
                if self.max_nodes is not None and num_nodes > self.max_nodes:
                    filtered_count += 1
                    continue
                
                # Generate masks (use output_size since patches are resized)
                image_size = (self.output_size, self.output_size)
                # Use original GT mask patch directly for edge mask (no simplification)
                # Note: patch['mask'] is already resized to output_size
                keypoint_mask, edge_mask, skeleton = self.generate_masks_from_graph(
                    patch['graph'], idx_to_coords, image_size, gt_mask_patch=patch['mask']
                )
                
                # Generate topology data
                graph_points, pairs, valid, connected = self.generate_topology_data(
                    adj_matrix, idx_to_coords, image_size
                )
                
                # Save sample
                sample_id = f"{idx:06d}"
                
                # Save image
                image_path = split_dir / 'images' / f"{sample_id}.png"
                cv2.imwrite(str(image_path), cv2.cvtColor(patch['image'], cv2.COLOR_RGB2BGR))
                
                # Save masks
                cv2.imwrite(str(split_dir / 'masks' / 'nodes' / f"{sample_id}.png"), keypoint_mask)
                cv2.imwrite(str(split_dir / 'masks' / 'edges' / f"{sample_id}.png"), edge_mask)
                cv2.imwrite(str(split_dir / 'masks' / 'skeleton' / f"{sample_id}.png"), skeleton)
                
                # Save adjacency matrix
                adj_path = split_dir / 'adjacency_matrices' / f"{sample_id}.npy"
                np.save(adj_path, adj_matrix)
                
                # Save topology data
                points_data = {
                    'coords': graph_points,
                    'pairs': pairs,
                    'valid': valid,
                    'connected': connected
                }
                points_path = split_dir / 'points' / f"{sample_id}_points.pkl"
                with open(points_path, 'wb') as f:
                    pickle.dump(points_data, f)
                
                # Save overlay visualization
                viz_path = split_dir / 'visualizations' / f"{sample_id}_overlay.png"
                self._save_overlay_visualization(
                    patch['image'], keypoint_mask, edge_mask, graph_points, adj_matrix, viz_path
                )
                
                # Add to metadata
                metadata.append({
                    'sample_id': sample_id,
                    'image_filename': f"{sample_id}.png",
                    'image_id': patch['image_id'],
                    'patch_coords': patch['patch_coords'],
                    'adjacency_matrix_filename': f"{sample_id}.npy",
                    'node_mask_filename': f"{sample_id}.png",
                    'edge_mask_filename': f"{sample_id}.png",
                    'skeleton_mask_filename': f"{sample_id}.png",
                    'points_filename': f"{sample_id}_points.pkl",
                    'num_nodes': int(num_nodes),
                    'num_edges': int(np.sum(adj_matrix) / 2),
                })
            
            # Save metadata
            metadata_path = split_dir / 'metadata.json'
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Print statistics
            if metadata:
                num_nodes_list = [m['num_nodes'] for m in metadata]
                num_edges_list = [m['num_edges'] for m in metadata]
                print(f"\n{split_name.upper()} Split Statistics:")
                print(f"  Total samples: {len(metadata)}")
                print(f"  Filtered samples (num_nodes > max_nodes): {filtered_count}")
                print(f"  Nodes: {min(num_nodes_list)}-{max(num_nodes_list)} (mean: {np.mean(num_nodes_list):.2f}, median: {np.median(num_nodes_list):.2f})")
                print(f"  Edges: {min(num_edges_list)}-{max(num_edges_list)} (mean: {np.mean(num_edges_list):.2f}, median: {np.median(num_edges_list):.2f})")
            
            dataset_stats[split_name] = {
                'num_samples': len(metadata),
                'filtered': filtered_count,
                'num_nodes': {
                    'min': int(min(num_nodes_list)) if metadata else 0,
                    'max': int(max(num_nodes_list)) if metadata else 0,
                    'mean': float(np.mean(num_nodes_list)) if metadata else 0.0,
                    'median': float(np.median(num_nodes_list)) if metadata else 0.0,
                },
                'num_edges': {
                    'min': int(min(num_edges_list)) if metadata else 0,
                    'max': int(max(num_edges_list)) if metadata else 0,
                    'mean': float(np.mean(num_edges_list)) if metadata else 0.0,
                    'median': float(np.median(num_edges_list)) if metadata else 0.0,
                }
            }
        
        # Save dataset info
        dataset_info = {
            'dataset_name': 'OCTA500',
            'patch_size': self.patch_size,  # Extraction size (64)
            'output_size': self.output_size,  # Output size after resize (256)
            'overlap': self.overlap,
            'stride': self.stride,
            'max_nodes': self.max_nodes,
            'splits': dataset_stats
        }
        info_path = self.output_path / 'dataset_info.json'
        with open(info_path, 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        print(f"\n{'='*80}")
        print("Dataset generation complete!")
        print(f"Output directory: {self.output_path}")
        print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(description='Generate OCTA500 benchmark dataset')
    parser.add_argument('--raw_data_path', type=str, required=True, help='Path to raw OCTA500 dataset')
    parser.add_argument('--output_path', type=str, required=True, help='Output path for benchmark dataset')
    parser.add_argument('--patch_size', type=int, default=64, help='Patch extraction size (default: 64)')
    parser.add_argument('--output_size', type=int, default=256, help='Output size after resize (default: 256)')
    parser.add_argument('--overlap', type=int, default=16, help='Overlap between patches (default: 16, 25% overlap for 64x64)')
    parser.add_argument('--max_samples_per_split', type=int, default=None, help='Maximum samples per split (for testing)')
    parser.add_argument('--max_nodes', type=int, default=None, help='Filter out samples with more nodes than this')
    parser.add_argument('--curvature-threshold', type=float, default=160.0,
                       dest='curvature_threshold',
                       help='Curvature threshold for node simplification (degrees, default: 160.0)')
    parser.add_argument('--min-edge-length', type=float, default=10.0,
                       dest='min_edge_length',
                       help='Minimum edge length in pixels to keep (default: 10.0)')
    
    args = parser.parse_args()
    
    generator = OCTA500BenchmarkDatasetGenerator(
        raw_data_path=args.raw_data_path,
        output_path=args.output_path,
        patch_size=args.patch_size,
        output_size=args.output_size,
        overlap=args.overlap,
        max_samples_per_split=args.max_samples_per_split,
        max_nodes=args.max_nodes,
    )
    
    # Set curvature threshold and min edge length after initialization
    generator.curvature_threshold = args.curvature_threshold
    generator.min_edge_length = args.min_edge_length
    
    generator.generate_benchmark_dataset()


if __name__ == '__main__':
    main()

