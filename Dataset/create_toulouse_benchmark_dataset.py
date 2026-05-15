#!/usr/bin/env python3
"""
Toulouse Road Network Benchmark Dataset Generation Pipeline

This script creates a benchmark dataset from Toulouse road network data for the 
image-to-graph conversion task. It uses existing node-link images and ground truth 
road network graphs.

Key features:
- Uses existing node-link images from Toulouse dataset
- Converts graph data to adjacency matrices
- Generates masks from ground truth graphs
- Creates train/validation/test splits (uses existing splits)
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
from PIL import Image


class ToulouseBenchmarkDatasetGenerator:
    """Generates benchmark datasets from Toulouse road network data."""
    
    def __init__(self, 
                 raw_data_path: str,
                 output_path: str,
                 image_size: Optional[Tuple[int, int]] = None,
                 use_existing_splits: bool = True):
        """
        Initialize the Toulouse benchmark dataset generator.
        
        Args:
            raw_data_path: Path to Toulouse dataset directory (0.001_toulouse)
            output_path: Path where benchmark dataset will be saved
            image_size: Target image size (width, height) for resizing. If None, keeps original 64×64 resolution.
            use_existing_splits: If True, uses existing train/val/test splits. Otherwise creates new splits.
        """
        self.raw_data_path = Path(raw_data_path)
        self.output_path = Path(output_path)
        self.image_size = image_size
        self.use_existing_splits = use_existing_splits
        
        # Create output directories
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
    
    def load_toulouse_data(self) -> Dict[str, List[Dict]]:
        """
        Load all data from Toulouse pickle files.
        
        Returns:
            Dictionary with keys 'train', 'val', 'test', each containing list of graph data dicts
        """
        print("Loading Toulouse dataset...")
        
        splits = {}
        
        for split_name in ['train', 'valid', 'test']:
            # Load graph data
            graph_pickle_path = self.raw_data_path / f"{split_name}.pickle"
            images_pickle_path = self.raw_data_path / f"{split_name}_images.pickle"
            
            if not graph_pickle_path.exists():
                print(f"Warning: {graph_pickle_path} not found, skipping {split_name}")
                continue
            
            with open(graph_pickle_path, 'rb') as f:
                graph_data = pickle.load(f)
            
            # Load images
            images_data = {}
            if images_pickle_path.exists():
                with open(images_pickle_path, 'rb') as f:
                    images_data = pickle.load(f)
            
            # Convert to list format
            samples = []
            for graph_id, graph_info in tqdm(graph_data.items(), desc=f"Loading {split_name}"):
                # Get image if available
                graph_id_val = graph_info.get('id', graph_id)
                img_id_str = str(graph_id_val)
                image = None
                if img_id_str in images_data:
                    image = images_data[img_id_str]
                else:
                    # Try with zero-padded format
                    try:
                        img_id_padded = f"{int(graph_id_val):07d}"
                        if img_id_padded in images_data:
                            image = images_data[img_id_padded]
                    except (ValueError, TypeError):
                        pass
                
                sample = {
                    'graph_id': graph_id,
                    'graph_info': graph_info,
                    'image': image,
                }
                samples.append(sample)
            
            # Map 'valid' to 'val' for consistency
            split_key = 'val' if split_name == 'valid' else split_name
            splits[split_key] = samples
            print(f"Loaded {len(samples)} samples from {split_name} split")
        
        return splits
    
    def graph_info_to_networkx(self, graph_info: Dict) -> nx.Graph:
        """
        Convert Toulouse graph info to NetworkX graph.
        
        Args:
            graph_info: Dictionary with 'nodes', 'edges', 'graph' keys
            
        Returns:
            NetworkX graph
        """
        G = nx.Graph()
        
        # Add nodes with their coordinates
        nodes = graph_info.get('nodes', [])
        for i, node_coord in enumerate(nodes):
            G.add_node(i, pos=node_coord)
        
        # Add edges from the 'graph' defaultdict or 'edges' list
        graph_dict = graph_info.get('graph', {})
        if graph_dict:
            for node_id, neighbors in graph_dict.items():
                for neighbor_id in neighbors:
                    if neighbor_id > node_id:  # Avoid duplicate edges
                        G.add_edge(node_id, neighbor_id)
        else:
            # Fallback to edges list
            edges = graph_info.get('edges', [])
            for edge in edges:
                if len(edge) >= 2:
                    G.add_edge(edge[0], edge[1])
        
        return G
    
    def networkx_to_adjacency_matrix(self, G: nx.Graph, sort_nodes: bool = True) -> Tuple[np.ndarray, Dict[int, Tuple]]:
        """
        Convert NetworkX graph to adjacency matrix and node index mapping.
        
        Args:
            G: NetworkX graph
            sort_nodes: If True, sorts nodes by index for deterministic ordering
            
        Returns:
            Tuple of (adjacency_matrix, idx_to_coords) where idx_to_coords maps node index to coordinates
        """
        if G.number_of_nodes() == 0:
            return np.zeros((0, 0), dtype=np.float32), {}
        
        nodes = sorted(G.nodes()) if sort_nodes else list(G.nodes())
        num_nodes = len(nodes)
        
        # Create adjacency matrix
        adj_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        
        # Map node index to coordinates
        idx_to_coords = {}
        
        for i, node in enumerate(nodes):
            # Get coordinates from node attributes
            pos = G.nodes[node].get('pos', (0.0, 0.0))
            if isinstance(pos, (list, tuple)) and len(pos) >= 2:
                idx_to_coords[i] = (float(pos[0]), float(pos[1]))
            else:
                idx_to_coords[i] = (0.0, 0.0)
            
            # Fill adjacency matrix
            for neighbor in G.neighbors(node):
                if neighbor in nodes:
                    j = nodes.index(neighbor)
                    adj_matrix[i, j] = 1.0
        
        # Make symmetric
        adj_matrix = (adj_matrix + adj_matrix.T) / 2.0
        adj_matrix = (adj_matrix > 0.5).astype(np.float32)
        
        return adj_matrix, idx_to_coords
    
    def load_and_resize_image(self, image: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Load and resize image to target size.
        
        Args:
            image: Input image array (can be shape (1, H, W) or (H, W))
            
        Returns:
            Tuple of (resized_image, original_size)
        """
        # Handle different input shapes
        if image.ndim == 3:
            if image.shape[0] == 1:
                image = image[0]  # Remove channel dimension
            else:
                image = image.transpose(1, 2, 0)  # (H, W, C)
        
        # Normalize from [-1, 1] to [0, 255] if needed
        if image.min() < 0:
            image = (image + 1.0) / 2.0  # Normalize to [0, 1]
        
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
        
        # Convert to RGB if grayscale
        if image.ndim == 2:
            image = np.stack([image, image, image], axis=-1)
        elif image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)
        
        original_size = (image.shape[1], image.shape[0])  # (width, height)
        
        # Resize to target size if specified, otherwise keep original size
        if self.image_size is not None and self.image_size != original_size:
            image_pil = Image.fromarray(image)
            image_pil = image_pil.resize(self.image_size, Image.Resampling.LANCZOS)
            image = np.array(image_pil)
            final_size = self.image_size
        else:
            final_size = original_size
        
        return image, final_size
    
    def generate_masks_from_graph(self, 
                                  G: nx.Graph,
                                  idx_to_coords: Dict[int, Tuple],
                                  image_size: Tuple[int, int],
                                  original_image_size: Optional[Tuple[int, int]] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate masks from graph data.
        
        Args:
            G: NetworkX graph
            idx_to_coords: Mapping from node index to coordinates
            image_size: Target image size (width, height)
            original_image_size: Original image size for coordinate scaling
            
        Returns:
            Tuple of (keypoint_mask, edge_mask, skeleton)
        """
        h, w = image_size[1], image_size[0]
        keypoint_mask = np.zeros((h, w), dtype=np.uint8)
        edge_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Calculate scaling factors
        orig_w, orig_h = original_image_size if original_image_size else (w, h)
        scale_x = w / orig_w
        scale_y = h / orig_h
        
        # Toulouse coordinates are already normalized to [-1, 1]
        # Convert to pixel coordinates with y-axis flip
        # (image coordinates have y increasing downward, normalized coords have y increasing upward)
        def to_pixel_coords(coord: Tuple[float, float]) -> Tuple[int, int]:
            x_norm, y_norm = float(coord[0]), float(coord[1])
            # Convert from [-1, 1] to [0, image_size]
            # X: direct mapping
            x_pixel = int((x_norm + 1) * w / 2)
            # Y: flip axis (1 - y_norm) to convert from bottom-up to top-down
            y_pixel = int((1 - y_norm) * h / 2)
            
            # Clamp to image bounds
            x_pixel = max(0, min(w - 1, x_pixel))
            y_pixel = max(0, min(h - 1, y_pixel))
            return x_pixel, y_pixel
        
        # Draw nodes - scale radius based on image size
        # For 64×64: radius=2, for 128×128: radius=3, for 256×256: radius=5, for 512×512: radius=8
        scale_factor = min(w, h) / 64  # Scale relative to 64×64
        if min(w, h) <= 64:
            node_radius = 2
        elif min(w, h) <= 128:
            node_radius = 5
        elif min(w, h) <= 256:
            node_radius = 5
        else:
            node_radius = 8
        
        for node_idx, coord in idx_to_coords.items():
            x, y = to_pixel_coords(coord)
            # Ensure radius doesn't go out of bounds
            x = max(node_radius, min(w - 1 - node_radius, x))
            y = max(node_radius, min(h - 1 - node_radius, y))
            cv2.circle(keypoint_mask, (x, y), node_radius, 255, -1)
        
        # Draw edges - scale width based on image size
        # For 64×64: width=1, for 128×128: width=2, for 256×256: width=3, for 512×512: width=4
        if min(w, h) <= 64:
            edge_width = 1
        elif min(w, h) <= 128:
            edge_width = 3
        elif min(w, h) <= 256:
            edge_width = 3
        else:
            edge_width = 4
        for edge in G.edges():
            node1, node2 = edge
            if node1 in idx_to_coords and node2 in idx_to_coords:
                x1, y1 = to_pixel_coords(idx_to_coords[node1])
                x2, y2 = to_pixel_coords(idx_to_coords[node2])
                cv2.line(edge_mask, (x1, y1), (x2, y2), 255, edge_width)
        
        # Create skeleton from edge mask
        skeleton = skeletonize((edge_mask > 0).astype(np.uint8)).astype(np.uint8)
        
        return keypoint_mask, edge_mask, skeleton
    
    def generate_topology_data(self, 
                              adj_matrix: np.ndarray, 
                              idx_to_coords: Dict[int, Tuple],
                              image_size: Tuple[int, int],
                              max_pairs: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate topology data (graph_points, pairs, valid, connected).
        
        Args:
            adj_matrix: Adjacency matrix
            idx_to_coords: Mapping from node index to coordinates
            image_size: Image size (width, height)
            max_pairs: Maximum number of pairs per node. If None, calculates dynamically.
            
        Returns:
            Tuple of (graph_points, pairs, valid, connected)
        """
        num_nodes = adj_matrix.shape[0]
        h, w = image_size[1], image_size[0]
        
        # Convert normalized coordinates to pixel positions
        # Y-axis needs to be flipped to match image coordinate system
        pixel_positions = []
        for i in range(num_nodes):
            if i in idx_to_coords:
                coord = idx_to_coords[i]
                x_norm, y_norm = float(coord[0]), float(coord[1])
                # Convert from [-1, 1] to [0, image_size]
                # X: direct mapping
                x_pixel = (x_norm + 1) * w / 2
                # Y: flip axis to convert from bottom-up to top-down
                y_pixel = (1 - y_norm) * h / 2
                pixel_positions.append([x_pixel, y_pixel])
            else:
                pixel_positions.append([w / 2, h / 2])  # Default center
        
        pixel_positions = np.array(pixel_positions, dtype=np.float32)
        
        # Calculate max_pairs dynamically if not provided
        if max_pairs is None:
            max_neighbors = 0
            for i in range(num_nodes):
                neighbors = np.where(adj_matrix[i] > 0)[0]
                max_neighbors = max(max_neighbors, len(neighbors))
            max_pairs = max(32, max_neighbors + 10)
        
        # Generate pairs
        pairs = []
        valid = []
        connected = []
        
        for i in range(num_nodes):
            node_pairs = []
            node_valid = []
            node_connected = []
            
            # Find neighbors
            neighbors = np.where(adj_matrix[i] > 0)[0]
            
            # Add actual neighbors
            for j in neighbors:
                if len(node_pairs) < max_pairs:
                    node_pairs.append([i, j])
                    node_valid.append(True)
                    node_connected.append(1.0)
            
            # Pad with dummy pairs
            while len(node_pairs) < max_pairs:
                node_pairs.append([i, i])  # Self-loop (invalid)
                node_valid.append(False)
                node_connected.append(0.0)
            
            pairs.append(node_pairs)
            valid.append(node_valid)
            connected.append(node_connected)
        
        pairs = np.array(pairs, dtype=np.int32)
        valid = np.array(valid, dtype=bool)
        connected = np.array(connected, dtype=np.float32)
        
        return pixel_positions, pairs, valid, connected
    
    def generate_benchmark_dataset(self, max_samples: Optional[int] = None):
        """Generate the complete benchmark dataset."""
        
        print("Starting Toulouse benchmark dataset generation...")
        
        # Load data
        splits = self.load_toulouse_data()
        
        # Generate dataset for each split
        all_metadata = {}
        
        for split_name, samples in splits.items():
            print(f"\nGenerating {split_name} split ({len(samples)} samples)...")
            
            # Limit samples if specified
            if max_samples:
                samples = samples[:max_samples]
            
            split_metadata = []
            
            for i, sample in enumerate(tqdm(samples, desc=f"Processing {split_name}")):
                graph_info = sample['graph_info']
                image = sample['image']
                graph_id = sample['graph_id']
                
                # Convert to NetworkX graph
                G = self.graph_info_to_networkx(graph_info)
                
                if G.number_of_nodes() == 0:
                    print(f"Warning: Skipping empty graph {graph_id}")
                    continue
                
                # Convert to adjacency matrix
                adj_matrix, idx_to_coords = self.networkx_to_adjacency_matrix(G)
                
                # Load and resize image
                if image is not None:
                    resized_image, final_image_size = self.load_and_resize_image(image)
                else:
                    # Generate blank image if not available
                    target_size = self.image_size if self.image_size is not None else (64, 64)
                    resized_image = np.ones((target_size[1], target_size[0], 3), dtype=np.uint8) * 255
                    final_image_size = target_size
                
                # Generate masks
                keypoint_mask, edge_mask, skeleton = self.generate_masks_from_graph(
                    G, idx_to_coords, final_image_size, final_image_size
                )
                
                # Generate topology data
                graph_points, pairs, valid, connected = self.generate_topology_data(
                    adj_matrix, idx_to_coords, final_image_size
                )
                
                # Save image
                image_filename = f"{i:06d}.png"
                image_path = self.output_path / split_name / "images" / image_filename
                plt.imsave(image_path, resized_image)
                
                # Save adjacency matrix
                adj_filename = f"{i:06d}_adj.npy"
                adj_path = self.output_path / split_name / "adjacency_matrices" / adj_filename
                np.save(adj_path, adj_matrix)
                
                # Save masks
                masks_dir = self.output_path / split_name / "masks"
                nodes_dir = masks_dir / "nodes"
                edges_dir = masks_dir / "edges"
                skeleton_dir = masks_dir / "skeleton"
                
                nodes_dir.mkdir(parents=True, exist_ok=True)
                edges_dir.mkdir(parents=True, exist_ok=True)
                skeleton_dir.mkdir(parents=True, exist_ok=True)
                
                node_mask_filename = f"{i:06d}_nodes.png"
                cv2.imwrite(str(nodes_dir / node_mask_filename), keypoint_mask)
                
                edge_mask_filename = f"{i:06d}_edges.png"
                cv2.imwrite(str(edges_dir / edge_mask_filename), edge_mask)
                
                skeleton_mask_filename = f"{i:06d}_skeleton.png"
                cv2.imwrite(str(skeleton_dir / skeleton_mask_filename), skeleton)
                
                # Save topology data
                points_filename = f"{i:06d}_points.pkl"
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
                    "original_graph_id": int(graph_id),
                    "split": split_name,
                    "graph_index": i,
                    "num_nodes": adj_matrix.shape[0],
                    "num_edges": int(np.sum(adj_matrix) / 2),
                    "adjacency_matrix_shape": list(adj_matrix.shape),
                    "num_points": len(graph_points),
                    "num_pairs": pairs.shape[1]
                }
                split_metadata.append(metadata)
            
            # Save split metadata
            metadata_path = self.output_path / split_name / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(split_metadata, f, indent=2)
            
            all_metadata[split_name] = split_metadata
        
        # Create overall dataset info
        # Determine actual image size from first sample or default
        actual_image_size = self.image_size if self.image_size is not None else (128, 128)
        
        dataset_info = {
            "total_samples": sum(len(meta) for meta in all_metadata.values()),
            "image_size": actual_image_size,
            "splits": {
                split_name: len(meta) 
                for split_name, meta in all_metadata.items()
            }
        }
        
        info_path = self.output_path / "dataset_info.json"
        with open(info_path, 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        print(f"\nToulouse benchmark dataset generation complete!")
        print(f"Dataset saved to: {self.output_path}")
        print(f"Total samples: {dataset_info['total_samples']}")
        for split_name, count in dataset_info['splits'].items():
            print(f"  {split_name}: {count} samples")


def main():
    parser = argparse.ArgumentParser(description="Generate benchmark dataset from Toulouse road network data")
    parser.add_argument("--raw_data_path", 
                       default="/usa/jiahaox/Image2matrix_baselines/dataset/raw/0.001_toulouse",
                       help="Path to Toulouse dataset directory")
    parser.add_argument("--output_path",
                       default="/usa/jiahaox/Image2matrix_baselines/dataset/processed/toulouse_benchmark_dataset",
                       help="Path where benchmark dataset will be saved")
    parser.add_argument("--image_size", nargs=2, type=int, default=[128, 128],
                       help="Image size (width height). Default is 128×128.")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum number of samples per split (for testing)")
    
    args = parser.parse_args()
    
    # Create generator
    image_size = tuple(args.image_size) if args.image_size is not None else None
    generator = ToulouseBenchmarkDatasetGenerator(
        raw_data_path=args.raw_data_path,
        output_path=args.output_path,
        image_size=image_size,
        use_existing_splits=True
    )
    
    # Generate dataset
    generator.generate_benchmark_dataset(max_samples=args.max_samples)


if __name__ == "__main__":
    main()

