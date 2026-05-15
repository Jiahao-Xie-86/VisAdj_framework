#!/usr/bin/env python3
"""
Benchmark Dataset Generation Pipeline

This script creates a comprehensive benchmark dataset for the node-link image to 
adjacency matrix conversion task. It generates node-link visualizations from 
adjacency matrices and stores both the images and ground truth data.

Key features:
- Generates multiple visualization styles for each graph
- Handles permutation invariance through graph isomorphism evaluation
- Creates train/validation/test splits
- Stores metadata for evaluation
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path
import json
import pickle
from typing import Dict, List, Tuple, Optional
import argparse
from tqdm import tqdm
import random
from sklearn.model_selection import train_test_split
import cv2
from skimage.morphology import skeletonize

class BenchmarkDatasetGenerator:
    """Generates benchmark datasets for node-link to adjacency matrix conversion."""
    
    def __init__(self, 
                 raw_data_path: str,
                 output_path: str,
                 image_size: Tuple[int, int] = (512, 512),
                 num_visualizations_per_graph: int = 1):
        """
        Initialize the benchmark dataset generator.
        
        Args:
            raw_data_path: Path to raw adjacency matrix .npy files
            output_path: Path where benchmark dataset will be saved
            image_size: Size of generated images (width, height)
            num_visualizations_per_graph: Number of different visualizations per graph
        """
        self.raw_data_path = Path(raw_data_path)
        self.output_path = Path(output_path)
        self.image_size = image_size
        self.num_visualizations_per_graph = num_visualizations_per_graph
        
        # Create output directories with separate folders for each split
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Create split-specific directories
        for split in ["train", "val", "test"]:
            (self.output_path / split).mkdir(exist_ok=True)
            (self.output_path / split / "images").mkdir(exist_ok=True)
            (self.output_path / split / "adjacency_matrices").mkdir(exist_ok=True)
            (self.output_path / split / "masks").mkdir(exist_ok=True)
            (self.output_path / split / "points").mkdir(exist_ok=True)
        
        # Create metadata directory
        (self.output_path / "metadata").mkdir(exist_ok=True)
        
        # Visualization styles
        self.visualization_styles = [
            "spring_layout",
            "kamada_kawai_layout"
        ]
        
    def load_adjacency_matrices(self) -> Dict[str, List[np.ndarray]]:
        """Load all adjacency matrices from the raw dataset."""
        print("Loading adjacency matrices...")
        
        graphs = {"planar": [], "nonplanar": []}
        
        # Load planar graphs (try medium_planar first, fallback to small_planar)
        planar_dir = self.raw_data_path / "medium_planar"
        if not planar_dir.exists():
        planar_dir = self.raw_data_path / "small_planar"
        if planar_dir.exists():
            planar_files = list(planar_dir.glob("*.npy"))
            for file_path in tqdm(planar_files, desc="Loading planar graphs"):
                adj_matrix = np.load(file_path)
                graphs["planar"].append({
                    "adjacency_matrix": adj_matrix,
                    "filename": file_path.name,
                    "graph_type": "planar"
                })
        
        # Load non-planar graphs (try medium_nonplanar first, fallback to small_nonplanar)
        nonplanar_dir = self.raw_data_path / "medium_nonplanar"
        if not nonplanar_dir.exists():
        nonplanar_dir = self.raw_data_path / "small_nonplanar"
        if nonplanar_dir.exists():
            nonplanar_files = list(nonplanar_dir.glob("*.npy"))
            for file_path in tqdm(nonplanar_files, desc="Loading non-planar graphs"):
                adj_matrix = np.load(file_path)
                graphs["nonplanar"].append({
                    "adjacency_matrix": adj_matrix,
                    "filename": file_path.name,
                    "graph_type": "nonplanar"
                })
        
        print(f"Loaded {len(graphs['planar'])} planar graphs")
        print(f"Loaded {len(graphs['nonplanar'])} non-planar graphs")
        
        return graphs
    
    def adjacency_matrix_to_graph(self, adj_matrix: np.ndarray) -> nx.Graph:
        """Convert adjacency matrix to NetworkX graph."""
        G = nx.from_numpy_array(adj_matrix.astype(float))
        return G
    
    def generate_node_link_visualization(self, 
                                       adj_matrix: np.ndarray, 
                                       style: str,
                                       seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """
        Generate node-link visualization from adjacency matrix using OpenCV.
        This ensures perfect consistency with mask generation.
        
        Args:
            adj_matrix: Adjacency matrix
            style: Visualization style ('spring_layout', 'kamada_kawai_layout')
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of (image as numpy array, node positions dictionary)
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        G = self.adjacency_matrix_to_graph(adj_matrix)
        
        # Skip if graph is empty
        if G.number_of_nodes() == 0:
            # Return a blank white image
            blank_image = np.ones((self.image_size[1], self.image_size[0], 3), dtype=np.uint8) * 255
            return blank_image, {}
        
        # Generate layout with parameters optimized for maximum space usage
        if style == "spring_layout":
            pos = nx.spring_layout(G, k=3.0, iterations=150, seed=seed, scale=4.0)
        elif style == "kamada_kawai_layout":
            pos = nx.kamada_kawai_layout(G, scale=4.0)
        else:
            pos = nx.spring_layout(G, k=3.0, iterations=150, seed=seed, scale=4.0)
        
        # Normalize coordinates to fit within [-1, 1] range with better distribution
        if pos:
            x_coords = [pos[node][0] for node in pos]
            y_coords = [pos[node][1] for node in pos]
            
            if x_coords and y_coords:
                # Find current bounds
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
                
                # Calculate ranges
                x_range = x_max - x_min if x_max != x_min else 1
                y_range = y_max - y_min if y_max != y_min else 1
                
                # Normalize to [-1, 1] range with minimal padding for maximum space usage
                padding = 0.01  # 1% padding for maximum use of space
                scale_x = (2 - 2*padding) / x_range
                scale_y = (2 - 2*padding) / y_range
                scale = min(scale_x, scale_y)
                
                # Center and scale coordinates
                center_x = (x_min + x_max) / 2
                center_y = (y_min + y_max) / 2
                
                for node in pos:
                    new_x = (pos[node][0] - center_x) * scale
                    new_y = (pos[node][1] - center_y) * scale
                    pos[node] = (new_x, new_y)
        
        # Create white background image using OpenCV
        h, w = self.image_size[1], self.image_size[0]
        image = np.ones((h, w, 3), dtype=np.uint8) * 255
        
        # Convert normalized [-1, 1] coordinates to pixel coordinates
        # OpenCV uses top-left origin, so no y-flipping needed
        pixel_positions = {}
        node_radius = 8  # Same radius used for drawing circles
        for node, (x, y) in pos.items():
            # Convert from [-1, 1] to [0, image_size]
            pixel_x = int((x + 1) * w / 2)
            pixel_y = int((y + 1) * h / 2)
            
            # Clamp coordinates to ensure circles stay within bounds
            pixel_x = max(node_radius, min(w - 1 - node_radius, pixel_x))
            pixel_y = max(node_radius, min(h - 1 - node_radius, pixel_y))
            
            pixel_positions[node] = (pixel_x, pixel_y)
        
        # Draw edges first (so they appear behind nodes)
        edge_color = (255, 0, 0)  # Blue in BGR
        edge_width = 3
        for edge in G.edges():
            node1, node2 = edge
            x1, y1 = pixel_positions[node1]
            x2, y2 = pixel_positions[node2]
            cv2.line(image, (x1, y1), (x2, y2), edge_color, edge_width)
        
        # Draw nodes
        node_color = (0, 0, 255)  # Red in BGR
        node_radius = 8
        for node, (x, y) in pixel_positions.items():
            cv2.circle(image, (x, y), node_radius, node_color, -1)
        
        # Convert BGR to RGB for consistency with the rest of the pipeline
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return image_rgb, pos
    
    def generate_perfect_masks(self, adj_matrix: np.ndarray, pos: Dict, 
                              image_size: Tuple[int, int] = (512, 512)) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate perfect masks from ground truth adjacency matrix and node positions.
        This is much more accurate than image-based mask generation.
        
        Returns:
        - keypoint_mask: Binary mask for nodes only
        - edge_mask: Binary mask for edges only (no nodes)
        - skeleton: Skeletonized version of edges for topology
        """
        h, w = image_size
        keypoint_mask = np.zeros((h, w), dtype=np.uint8)
        edge_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Convert NetworkX positions to pixel coordinates
        # IMPORTANT: Use the EXACT same normalization as in image generation
        if pos:
            x_coords = [pos[node][0] for node in pos]
            y_coords = [pos[node][1] for node in pos]
            
            if x_coords and y_coords:
                # Find current bounds
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
                
                # Calculate ranges
                x_range = x_max - x_min if x_max != x_min else 1
                y_range = y_max - y_min if y_max != y_min else 1
                
                # Normalize to [-1, 1] range with minimal padding for maximum space usage
                padding = 0.01  # 1% padding for maximum use of space
                scale_x = (2 - 2*padding) / x_range
                scale_y = (2 - 2*padding) / y_range
                scale = min(scale_x, scale_y)
                
                # Center and scale coordinates (SAME as image generation)
                center_x = (x_min + x_max) / 2
                center_y = (y_min + y_max) / 2
                
                for node in pos:
                    new_x = (pos[node][0] - center_x) * scale
                    new_y = (pos[node][1] - center_y) * scale
                    pos[node] = (new_x, new_y)
        
        # Now convert normalized [-1, 1] coordinates to pixel coordinates
        # Both image and mask generation use OpenCV, so no y-flipping needed
        pixel_positions = {}
        node_radius = 8  # Same radius used for drawing circles
        for node, (x, y) in pos.items():
            # Convert from [-1, 1] to [0, image_size]
            pixel_x = int((x + 1) * w / 2)
            pixel_y = int((y + 1) * h / 2)
            
            # Clamp coordinates to ensure circles stay within bounds
            pixel_x = max(node_radius, min(w - 1 - node_radius, pixel_x))
            pixel_y = max(node_radius, min(h - 1 - node_radius, pixel_y))
            
            pixel_positions[node] = (pixel_x, pixel_y)
        
        # Draw nodes on keypoint mask (nodes only)
        node_radius = 8  # Same radius as used in image generation
        for node, (x, y) in pixel_positions.items():
            cv2.circle(keypoint_mask, (x, y), node_radius, 255, -1)
        
        # Draw edges on edge mask (edges can overlap with nodes)
        edge_width = 3
        G = self.adjacency_matrix_to_graph(adj_matrix)
        for edge in G.edges():
            node1, node2 = edge
            x1, y1 = pixel_positions[node1]
            x2, y2 = pixel_positions[node2]
            cv2.line(edge_mask, (x1, y1), (x2, y2), 255, edge_width)
        
        # NO overlap removal - edges and nodes can overlap as requested
        
        # Create skeleton from FULL graph (ALLOWING overlap) to ensure continuous paths
        # The skeleton needs to connect through nodes for proper topology
        full_graph_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Draw nodes
        for node, (x, y) in pixel_positions.items():
            cv2.circle(full_graph_mask, (x, y), node_radius, 255, -1)
        
        # Draw edges (this will overlap with nodes - that's OK for skeleton)
        for edge in G.edges():
            node1, node2 = edge
            x1, y1 = pixel_positions[node1]
            x2, y2 = pixel_positions[node2]
            cv2.line(full_graph_mask, (x1, y1), (x2, y2), 255, edge_width)
        
        # Generate skeleton from full graph (overlap allowed for continuous paths)
        skeleton = skeletonize((full_graph_mask > 0).astype(np.uint8)).astype(np.uint8)
        
        return keypoint_mask, edge_mask, skeleton
    
    def generate_perfect_topology(self, adj_matrix: np.ndarray, pos: Dict,
                                image_size: Tuple[int, int] = (512, 512),
                                max_pairs: int = 32) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate perfect topology data from ground truth adjacency matrix.
        This is much more accurate than image-based topology generation.
        """
        num_nodes = adj_matrix.shape[0]
        
        # Convert NetworkX positions to pixel coordinates
        # IMPORTANT: Use the EXACT same normalization as in image generation
        if pos:
            x_coords = [pos[node][0] for node in pos]
            y_coords = [pos[node][1] for node in pos]
            
            if x_coords and y_coords:
                # Find current bounds
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
                
                # Calculate ranges
                x_range = x_max - x_min if x_max != x_min else 1
                y_range = y_max - y_min if y_max != y_min else 1
                
                # Normalize to [-1, 1] range with minimal padding for maximum space usage
                padding = 0.01  # 1% padding for maximum use of space
                scale_x = (2 - 2*padding) / x_range
                scale_y = (2 - 2*padding) / y_range
                scale = min(scale_x, scale_y)
                
                # Center and scale coordinates (SAME as image generation)
                center_x = (x_min + x_max) / 2
                center_y = (y_min + y_max) / 2
                
                for node in pos:
                    new_x = (pos[node][0] - center_x) * scale
                    new_y = (pos[node][1] - center_y) * scale
                    pos[node] = (new_x, new_y)
        
        # Now convert normalized [-1, 1] coordinates to pixel coordinates
        # Both image and topology generation use OpenCV, so no y-flipping needed
        pixel_positions = []
        node_radius = 8  # Same radius used for drawing circles
        for node in range(num_nodes):
            if node in pos:
                x, y = pos[node]
                # Convert from [-1, 1] to [0, image_size]
                pixel_x = (x + 1) * image_size[0] / 2
                pixel_y = (y + 1) * image_size[1] / 2
                
                # Clamp coordinates to ensure circles stay within bounds
                pixel_x = max(node_radius, min(image_size[0] - 1 - node_radius, pixel_x))
                pixel_y = max(node_radius, min(image_size[1] - 1 - node_radius, pixel_y))
                
                pixel_positions.append([pixel_x, pixel_y])
            else:
                # Fallback position if node not in pos
                pixel_positions.append([image_size[0]//2, image_size[1]//2])
        
        pixel_positions = np.array(pixel_positions, dtype=np.float32)
        
        # Generate pairs: each node connects to its neighbors in the adjacency matrix
        pairs = []
        valid = []
        connected = []
        
        for i in range(num_nodes):
            node_pairs = []
            node_valid = []
            node_connected = []
            
            # Find neighbors of node i
            neighbors = np.where(adj_matrix[i] > 0)[0]
            
            # Add actual neighbors
            for j in neighbors:
                if len(node_pairs) < max_pairs:
                    node_pairs.append([i, j])
                    node_valid.append(True)
                    node_connected.append(1.0)  # Connected in ground truth
            
            # Pad with dummy pairs if needed
            while len(node_pairs) < max_pairs:
                node_pairs.append([i, i])  # Self-loop (will be marked as invalid)
                node_valid.append(False)
                node_connected.append(0.0)
            
            pairs.append(node_pairs)
            valid.append(node_valid)
            connected.append(node_connected)
        
        # Convert to numpy arrays
        pairs = np.array(pairs, dtype=np.int32)
        valid = np.array(valid, dtype=bool)
        connected = np.array(connected, dtype=np.float32)
        
        return pixel_positions, pairs, valid, connected
    
    def create_dataset_split(self, graphs: Dict[str, List[Dict]], 
                           test_size: float = 0.15, 
                           val_size: float = 0.15) -> Dict[str, List[Dict]]:
        """Create train/validation/test splits with 0.7:0.15:0.15 ratio."""
        
        # Combine all graphs first
        all_graphs = graphs["planar"] + graphs["nonplanar"]
        
        # Split by graph type to maintain balance
        planar_indices = list(range(len(graphs["planar"])))
        nonplanar_indices = list(range(len(graphs["nonplanar"])))
        
        # Split planar graphs: 70% train, 15% val, 15% test
        planar_train, planar_temp = train_test_split(
            planar_indices, test_size=test_size + val_size, random_state=42
        )
        planar_val, planar_test = train_test_split(
            planar_temp, test_size=test_size/(test_size + val_size), random_state=42
        )
        
        # Split non-planar graphs: 70% train, 15% val, 15% test
        nonplanar_train, nonplanar_temp = train_test_split(
            nonplanar_indices, test_size=test_size + val_size, random_state=42
        )
        nonplanar_val, nonplanar_test = train_test_split(
            nonplanar_temp, test_size=test_size/(test_size + val_size), random_state=42
        )
        
        # Create splits
        splits = {
            "train": [],
            "val": [],
            "test": []
        }
        
        for split_name, planar_idx, nonplanar_idx in [
            ("train", planar_train, nonplanar_train),
            ("val", planar_val, nonplanar_val),
            ("test", planar_test, nonplanar_test)
        ]:
            split_graphs = []
            for idx in planar_idx:
                split_graphs.append(graphs["planar"][idx])
            for idx in nonplanar_idx:
                split_graphs.append(graphs["nonplanar"][idx])
            splits[split_name] = split_graphs
        
        return splits
    
    def generate_benchmark_dataset(self, max_graphs_per_type: Optional[int] = None):
        """Generate the complete benchmark dataset."""
        
        print("Starting benchmark dataset generation...")
        
        # Load adjacency matrices
        graphs = self.load_adjacency_matrices()
        
        # Limit dataset size if specified
        if max_graphs_per_type:
            graphs["planar"] = graphs["planar"][:max_graphs_per_type]
            graphs["nonplanar"] = graphs["nonplanar"][:max_graphs_per_type]
        
        # Create splits
        splits = self.create_dataset_split(graphs)
        
        # Generate visualizations for each split
        for split_name, split_graphs in splits.items():
            print(f"\nGenerating {split_name} split ({len(split_graphs)} graphs)...")
            
            split_metadata = []
            
            for i, graph_data in enumerate(tqdm(split_graphs, desc=f"Processing {split_name}")):
                adj_matrix = graph_data["adjacency_matrix"]
                filename = graph_data["filename"]
                graph_type = graph_data["graph_type"]
                
                # Generate single random visualization
                style = random.choice(self.visualization_styles)
                
                # Generate image and get node positions (using the EXACT same positions)
                image, pos = self.generate_node_link_visualization(
                    adj_matrix, style, seed=i
                )
                
                # Generate perfect masks from ground truth
                keypoint_mask, edge_mask, skeleton = self.generate_perfect_masks(
                    adj_matrix, pos, self.image_size
                )
                
                # Generate perfect topology data from ground truth
                graph_points, pairs, valid, connected = self.generate_perfect_topology(
                    adj_matrix, pos, self.image_size
                )
                
                # Save image in split-specific folder
                image_filename = f"{i:06d}_{style}.png"
                image_path = self.output_path / split_name / "images" / image_filename
                plt.imsave(image_path, image)
                
                # Save adjacency matrix in split-specific folder
                adj_filename = f"{i:06d}_adjacency.npy"
                adj_path = self.output_path / split_name / "adjacency_matrices" / adj_filename
                np.save(adj_path, adj_matrix)
                
                # Save perfect masks as separate binary files in organized folders
                # Create mask subdirectories
                masks_dir = self.output_path / split_name / "masks"
                nodes_dir = masks_dir / "nodes"
                edges_dir = masks_dir / "edges"
                skeleton_dir = masks_dir / "skeleton"
                
                nodes_dir.mkdir(parents=True, exist_ok=True)
                edges_dir.mkdir(parents=True, exist_ok=True)
                skeleton_dir.mkdir(parents=True, exist_ok=True)
                
                # Node mask (keypoints)
                node_mask_filename = f"{i:06d}_{style}_nodes.png"
                node_mask_path = nodes_dir / node_mask_filename
                cv2.imwrite(str(node_mask_path), keypoint_mask)
                
                # Edge mask (edges can overlap with nodes)
                edge_mask_filename = f"{i:06d}_{style}_edges.png"
                edge_mask_path = edges_dir / edge_mask_filename
                cv2.imwrite(str(edge_mask_path), edge_mask)
                
                # Skeleton mask (for topology)
                skeleton_mask_filename = f"{i:06d}_{style}_skeleton.png"
                skeleton_mask_path = skeleton_dir / skeleton_mask_filename
                cv2.imwrite(str(skeleton_mask_path), skeleton)
                
                # Save perfect topology data
                points_filename = f"{i:06d}_{style}_points.pkl"
                points_path = self.output_path / split_name / "points" / points_filename
                
                with open(points_path, 'wb') as f:
                    pickle.dump({
                        'graph_points': graph_points,
                        'pairs': pairs,
                        'valid': valid,
                        'connected': connected,
                        'num_points': len(graph_points),
                        'num_pairs': pairs.shape[1]
                    }, f)
                
                # Store metadata
                metadata = {
                    "image_filename": image_filename,
                    "adjacency_filename": adj_filename,
                    "node_mask_filename": node_mask_filename,
                    "edge_mask_filename": edge_mask_filename,
                    "skeleton_mask_filename": skeleton_mask_filename,
                    "points_filename": points_filename,
                    "original_filename": filename,
                    "graph_type": graph_type,
                    "visualization_style": style,
                    "split": split_name,
                    "graph_index": i,
                    "num_nodes": adj_matrix.shape[0],
                    "num_edges": int(np.sum(adj_matrix) / 2),  # Undirected graph
                    "adjacency_matrix_shape": adj_matrix.shape,
                    "num_points": len(graph_points),
                    "num_pairs": pairs.shape[1]
                }
                split_metadata.append(metadata)
            
            # Save split metadata in split-specific folder
            metadata_path = self.output_path / split_name / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(split_metadata, f, indent=2)
        
        # Create overall dataset info
        dataset_info = {
            "total_graphs": len(graphs["planar"]) + len(graphs["nonplanar"]),
            "planar_graphs": len(graphs["planar"]),
            "nonplanar_graphs": len(graphs["nonplanar"]),
            "visualizations_per_graph": 1,  # Single random visualization per graph
            "total_samples": sum(len(split_graphs) for split_graphs in splits.values()),
            "image_size": self.image_size,
            "visualization_styles": self.visualization_styles,
            "splits": {
                split_name: len(split_graphs) 
                for split_name, split_graphs in splits.items()
            }
        }
        
        info_path = self.output_path / "dataset_info.json"
        with open(info_path, 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        print(f"\nBenchmark dataset generation complete!")
        print(f"Dataset saved to: {self.output_path}")
        print(f"Total samples: {dataset_info['total_samples']}")
        print(f"Train samples: {len(splits['train'])} (70%)")
        print(f"Validation samples: {len(splits['val'])} (15%)")
        print(f"Test samples: {len(splits['test'])} (15%)")


def main():
    parser = argparse.ArgumentParser(description="Generate benchmark dataset for node-link to adjacency matrix conversion")
    parser.add_argument("--raw_data_path", 
                       default="/usa/jiahaox/Image2matrix_baselines/dataset/raw/21-50_planar_graphs_dataset",
                       help="Path to raw adjacency matrix files")
    parser.add_argument("--output_path",
                       default="/usa/jiahaox/Image2matrix_baselines/dataset/processed/benchmark_dataset_21-50",
                       help="Path where benchmark dataset will be saved")
    parser.add_argument("--image_size", nargs=2, type=int, default=[512, 512],
                       help="Image size (width height)")
    parser.add_argument("--num_visualizations", type=int, default=3,
                       help="Number of visualizations per graph")
    parser.add_argument("--max_graphs_per_type", type=int, default=None,
                       help="Maximum number of graphs per type (for testing)")
    
    args = parser.parse_args()
    
    # Create generator
    generator = BenchmarkDatasetGenerator(
        raw_data_path=args.raw_data_path,
        output_path=args.output_path,
        image_size=tuple(args.image_size),
        num_visualizations_per_graph=args.num_visualizations
    )
    
    # Generate dataset
    generator.generate_benchmark_dataset(max_graphs_per_type=args.max_graphs_per_type)


if __name__ == "__main__":
    main()
