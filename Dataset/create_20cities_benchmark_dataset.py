#!/usr/bin/env python3
"""
Generate benchmark dataset from 20 U.S. Cities satellite imagery dataset.

Dataset characteristics:
- 180 regions at 2048×2048 resolution
- Split: 144 train, 9 validation, 27 test (regions)
- Extract overlapping 128×128 patches from each region
- Node simplification: prune degree-2 nodes with curvature < 160°
"""

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
import networkx as nx
import math
from skimage.morphology import skeletonize
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt


class TwentyCitiesBenchmarkDatasetGenerator:
    def __init__(
        self,
        raw_data_path: str,
        output_path: str,
        patch_size: int = 128,
        overlap: int = 64,  # 50% overlap
        max_samples_per_split: Optional[int] = None,
        curvature_threshold: float = 160.0,  # degrees
        max_nodes: Optional[int] = None,  # Filter out samples with more nodes than this
    ):
        self.raw_data_path = Path(raw_data_path)
        self.output_path = Path(output_path)
        self.patch_size = patch_size
        self.overlap = overlap
        self.stride = patch_size - overlap
        self.max_samples_per_split = max_samples_per_split
        self.curvature_threshold = curvature_threshold
        self.max_nodes = max_nodes
        
        # Create output directories
        self.output_path.mkdir(parents=True, exist_ok=True)
        
    def load_region_data(self, region_id: int) -> Dict:
        """Load satellite image and graph for a region."""
        # Load satellite image
        sat_path = self.raw_data_path / f"region_{region_id}_sat.png"
        sat_img = Image.open(sat_path)
        sat_array = np.array(sat_img)
        
        # Load ground truth mask
        gt_path = self.raw_data_path / f"region_{region_id}_gt.png"
        gt_mask = np.array(Image.open(gt_path))
        
        # Load graph
        graph_path = self.raw_data_path / f"region_{region_id}_graph_gt.pickle"
        with open(graph_path, 'rb') as f:
            graph_dict = pickle.load(f)
        
        return {
            'region_id': region_id,
            'image': sat_array,
            'gt_mask': gt_mask,
            'graph': graph_dict
        }
    
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
    
    def simplify_graph(self, graph_dict: Dict[Tuple[int, int], List[Tuple[int, int]]]) -> Dict[Tuple[int, int], List[Tuple[int, int]]]:
        """
        Simplify graph by removing degree-2 nodes with curvature < threshold.
        Following Belli et al., we compute the angle between road segments at degree-2 nodes.
        """
        simplified = {}
        
        # First, identify degree-2 nodes to potentially remove
        degree_2_nodes = {node for node, neighbors in graph_dict.items() if len(neighbors) == 2}
        
        # Check curvature for each degree-2 node
        nodes_to_remove = set()
        for node in degree_2_nodes:
            neighbors = graph_dict[node]
            if len(neighbors) == 2:
                # Compute angle
                angle = self.compute_angle(neighbors[0], node, neighbors[1])
                
                # If angle is close to 180° (straight line), remove this node
                if angle >= self.curvature_threshold:
                    nodes_to_remove.add(node)
        
        # Build simplified graph
        # First, copy all nodes that we're keeping
        for node, neighbors in graph_dict.items():
            if node not in nodes_to_remove:
                simplified[node] = []
        
        # Now rebuild edges, bypassing removed nodes
        for node in simplified.keys():
            original_neighbors = graph_dict[node]
            
            for neighbor in original_neighbors:
                if neighbor not in nodes_to_remove:
                    # Direct neighbor, keep it
                    if neighbor not in simplified[node]:
                        simplified[node].append(neighbor)
                else:
                    # Neighbor is removed, find the next non-removed node
                    # by following the path
                    current = neighbor
                    visited = {node}
                    
                    while current in nodes_to_remove:
                        visited.add(current)
                        # Get neighbors of current
                        next_nodes = [n for n in graph_dict[current] if n not in visited]
                        if not next_nodes:
                            break
                        current = next_nodes[0]
                    
                    # current is now a non-removed node
                    if current not in nodes_to_remove and current != node:
                        if current not in simplified[node]:
                            simplified[node].append(current)
        
        return simplified
    
    def extract_patch_graph(self, graph_dict: Dict, x_start: int, y_start: int, patch_size: int) -> Optional[Dict]:
        """Extract graph nodes within patch boundaries.
        
        This function ensures perfect correspondence between image patches and graph patches:
        - Image patch: image[y_start:y_end, x_start:x_end]
        - Graph nodes: (row, col) where y_start <= row < y_end AND x_start <= col < x_end
        - Both use the same coordinate system: (row, col) = (y, x)
        
        Args:
            graph_dict: Dictionary with nodes as (row, col) tuples in full image coordinates
            x_start: Starting column of patch (in image coordinate system)
            y_start: Starting row of patch (in image coordinate system)
            patch_size: Size of patch
            
        Returns:
            Dictionary with patch-relative coordinates (row, col)
            Coordinates are translated: (row - y_start, col - x_start)
        """
        x_end = x_start + patch_size
        y_end = y_start + patch_size
        
        # Find nodes within patch
        # CRITICAL: Coordinate system consistency
        # - Image indexing: image[row, col] = image[y, x]
        # - Graph format: (row, col) = (y, x)
        # - Patch extraction: image[y_start:y_end, x_start:x_end]
        # - Graph check: y_start <= row < y_end AND x_start <= col < x_end
        # This ensures perfect alignment between image and graph patches
        patch_nodes = {}
        for node, neighbors in graph_dict.items():
            row, col = node  # Graph stores (row, col) = (y, x)
            if y_start <= row < y_end and x_start <= col < x_end:
                # Translate to patch coordinates (still as row, col)
                # This maintains coordinate system consistency
                new_node = (row - y_start, col - x_start)
                new_neighbors = []
                
                for neighbor in neighbors:
                    n_row, n_col = neighbor
                    if y_start <= n_row < y_end and x_start <= n_col < x_end:
                        new_neighbors.append((n_row - y_start, n_col - x_start))
                
                if new_neighbors:  # Only keep nodes with at least one neighbor in patch
                    patch_nodes[new_node] = new_neighbors
        
        # Filter out isolated nodes (no neighbors)
        patch_nodes = {n: nbrs for n, nbrs in patch_nodes.items() if nbrs}
        
        return patch_nodes if len(patch_nodes) >= 2 else None  # Need at least 2 nodes
    
    def graph_dict_to_networkx(self, graph_dict: Dict) -> nx.Graph:
        """Convert graph dictionary to NetworkX graph."""
        G = nx.Graph()
        for node, neighbors in graph_dict.items():
            G.add_node(node, pos=node)
            for neighbor in neighbors:
                G.add_edge(node, neighbor)
        return G
    
    def networkx_to_adjacency_matrix(self, G: nx.Graph) -> Tuple[np.ndarray, Dict[int, Tuple]]:
        """Convert NetworkX graph to adjacency matrix and node mapping.
        
        Returns:
            adj_matrix: Adjacency matrix
            idx_to_coords: Mapping from node index to coordinates (row, col) format
        """
        nodes = sorted(G.nodes())
        n = len(nodes)
        
        # Create node index mapping (preserves (row, col) format)
        node_to_idx = {node: idx for idx, node in enumerate(nodes)}
        idx_to_coords = {idx: node for idx, node in enumerate(nodes)}
        
        # Create adjacency matrix
        adj_matrix = np.zeros((n, n), dtype=np.float32)
        for u, v in G.edges():
            i, j = node_to_idx[u], node_to_idx[v]
            adj_matrix[i, j] = 1.0
            adj_matrix[j, i] = 1.0
        
        return adj_matrix, idx_to_coords
    
    def generate_masks_from_graph(self, G: nx.Graph, idx_to_coords: Dict, image_size: Tuple[int, int], gt_mask_patch: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate keypoint, edge, and skeleton masks.
        
        IMPORTANT: Edge mask is generated from the simplified graph (G) to match the adjacency matrix.
        This ensures consistency between the edge mask and the training target (adjacency matrix).
        
        Args:
            G: NetworkX graph (simplified graph, matches adjacency matrix)
            idx_to_coords: Mapping from node index to (row, col) coordinates
            image_size: (height, width) of the patch
            gt_mask_patch: Optional GT mask patch (currently not used - we generate from graph for consistency)
        
        Note: Graph coordinates are (row, col) = (y, x), but cv2 functions expect (x, y) = (col, row).
        We need to swap coordinates when drawing.
        """
        h, w = image_size
        
        # Initialize masks
        keypoint_mask = np.zeros((h, w), dtype=np.uint8)
        edge_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Draw nodes - scale based on image size
        if min(w, h) <= 64:
            node_radius = 2
        elif min(w, h) <= 128:
            node_radius = 3
        elif min(w, h) <= 256:
            node_radius = 5
        else:
            node_radius = 8
        
        # Draw ALL nodes from idx_to_coords to match adjacency matrix
        # After simplification, all nodes in the graph are key nodes
        for idx, (row, col) in idx_to_coords.items():
            # Graph stores (row, col) = (y, x), cv2 expects (x, y) = (col, row)
            x, y = col, row  # Swap to (x, y) for cv2
            x, y = int(x), int(y)
            x = max(node_radius, min(w - 1 - node_radius, x))
            y = max(node_radius, min(h - 1 - node_radius, y))
            cv2.circle(keypoint_mask, (x, y), node_radius, 255, -1)
        
        # Generate edge mask from simplified graph to match adjacency matrix
        # This ensures the edge mask represents the same structure as the adjacency matrix
        if min(w, h) <= 64:
            edge_width = 1
        elif min(w, h) <= 128:
            edge_width = 2
        elif min(w, h) <= 256:
            edge_width = 3
        else:
            edge_width = 4
        
        for edge in G.edges():
            node1, node2 = edge
            # Graph stores (row, col), cv2 expects (x, y) = (col, row)
            row1, col1 = node1
            row2, col2 = node2
            x1, y1 = int(col1), int(row1)  # Swap
            x2, y2 = int(col2), int(row2)  # Swap
            cv2.line(edge_mask, (x1, y1), (x2, y2), 255, edge_width)
        
        # Generate skeleton from edge mask
        skeleton_bool = skeletonize(edge_mask > 0)
        skeleton = (skeleton_bool * 255).astype(np.uint8)
        
        return keypoint_mask, edge_mask, skeleton
    
    def _save_overlay_visualization(
        self, 
        image: np.ndarray, 
        node_mask: np.ndarray, 
        edge_mask: np.ndarray, 
        graph_points: np.ndarray, 
        adj_matrix: np.ndarray,
        save_path: Path
    ):
        """Generate and save overlay visualization: image + masks + points."""
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        # Original image
        axes[0].imshow(image)
        axes[0].set_title('Satellite Image')
        axes[0].axis('off')
        
        # Image + edge mask
        axes[1].imshow(image)
        axes[1].imshow(edge_mask, cmap='hot', alpha=0.4)
        axes[1].set_title('Image + Edge Mask')
        axes[1].axis('off')
        
        # Image + node mask
        axes[2].imshow(image)
        axes[2].imshow(node_mask, cmap='hot', alpha=0.5)
        axes[2].set_title('Image + Node Mask')
        axes[2].axis('off')
        
        # Full overlay: image + masks + graph structure
        axes[3].imshow(image, alpha=0.7)
        axes[3].imshow(edge_mask, cmap='gray', alpha=0.3)
        
        # Draw edges from adjacency matrix
        for i in range(len(graph_points)):
            for j in range(i+1, len(graph_points)):
                if adj_matrix[i, j] > 0:
                    x1, y1 = graph_points[i]
                    x2, y2 = graph_points[j]
                    axes[3].plot([x1, x2], [y1, y2], 'b-', linewidth=2, alpha=0.7)
        
        # Draw nodes
        axes[3].scatter(graph_points[:, 0], graph_points[:, 1], 
                       c='red', s=100, marker='o', edgecolors='yellow', 
                       linewidths=2, zorder=5)
        
        # Label nodes
        for i, pt in enumerate(graph_points):
            axes[3].text(pt[0], pt[1], str(i), color='white', fontsize=8, 
                        ha='center', va='center', fontweight='bold', zorder=6)
        
        axes[3].set_title(f'Full Overlay ({len(graph_points)} nodes, {int(np.sum(adj_matrix)/2)} edges)')
        axes[3].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
    
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
    
    def extract_patches(self, region_data: Dict) -> List[Dict]:
        """Extract overlapping patches from a region."""
        patches = []
        image = region_data['image']
        gt_mask = region_data['gt_mask']
        graph_dict = region_data['graph']
        region_id = region_data['region_id']
        
        h, w = image.shape[:2]
        
        # Generate patch coordinates
        # CRITICAL: Use same (x_start, y_start) for image, GT mask, and graph extraction
        # This ensures perfect correspondence between all patch types
        for y_start in range(0, h - self.patch_size + 1, self.stride):
            for x_start in range(0, w - self.patch_size + 1, self.stride):
                # Extract image patch: image[y_start:y_end, x_start:x_end]
                patch_image = image[y_start:y_start+self.patch_size, x_start:x_start+self.patch_size]
                
                # Extract GT mask patch: gt_mask[y_start:y_end, x_start:x_end]
                # Uses same coordinates as image patch for perfect alignment
                patch_gt_mask = gt_mask[y_start:y_start+self.patch_size, x_start:x_start+self.patch_size]
                
                # Extract graph patch: nodes where y_start <= row < y_end AND x_start <= col < x_end
                # Uses same (x_start, y_start) as image/GT mask for perfect alignment
                patch_graph = self.extract_patch_graph(graph_dict, x_start, y_start, self.patch_size)
                
                if patch_graph is None:
                    continue  # Skip patches with insufficient graph data
                
                # Simplify graph (remove degree-2 nodes with low curvature)
                simplified_graph = self.simplify_graph(patch_graph)
                
                if len(simplified_graph) < 2:
                    continue  # Need at least 2 nodes after simplification
                
                patches.append({
                    'region_id': region_id,
                    'patch_coords': (x_start, y_start),
                    'image': patch_image,
                    'gt_mask': patch_gt_mask,  # Add GT mask patch
                    'graph': simplified_graph
                })
        
        return patches
    
    def generate_benchmark_dataset(self):
        """Generate the complete benchmark dataset."""
        # Get all regions
        all_regions = sorted([int(p.stem.split('_')[1]) for p in self.raw_data_path.glob('region_*_sat.png')])
        
        print(f"Found {len(all_regions)} regions")
        
        # Split: 144 train, 9 val, 27 test
        train_regions = all_regions[:144]
        val_regions = all_regions[144:153]
        test_regions = all_regions[153:180]
        
        print(f"Train regions: {len(train_regions)}")
        print(f"Val regions: {len(val_regions)}")
        print(f"Test regions: {len(test_regions)}")
        
        splits = {
            'train': train_regions,
            'val': val_regions,
            'test': test_regions
        }
        
        dataset_stats = {}
        
        for split_name, region_ids in splits.items():
            print(f"\n{'='*80}")
            print(f"Processing {split_name} split ({len(region_ids)} regions)")
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
            (split_dir / 'visualizations').mkdir(exist_ok=True)  # For overlay visualizations
            
            all_patches = []
            
            # Extract patches from all regions in this split
            for region_id in tqdm(region_ids, desc=f"Extracting patches from {split_name} regions"):
                region_data = self.load_region_data(region_id)
                patches = self.extract_patches(region_data)
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
                # Convert graph to NetworkX
                G = self.graph_dict_to_networkx(patch['graph'])
                
                if len(G.nodes()) < 2:
                    continue
                
                # Generate adjacency matrix
                adj_matrix, idx_to_coords = self.networkx_to_adjacency_matrix(G)
                
                # Filter: Skip samples with num_nodes > max_nodes
                num_nodes = adj_matrix.shape[0]
                if self.max_nodes is not None and num_nodes > self.max_nodes:
                    filtered_count += 1
                    continue
                
                # Generate masks
                image_size = (self.patch_size, self.patch_size)
                # Generate masks from simplified graph to match adjacency matrix
                # Note: We don't use gt_mask_patch here to ensure edge mask matches adjacency matrix
                keypoint_mask, edge_mask, skeleton = self.generate_masks_from_graph(G, idx_to_coords, image_size)
                
                # Generate topology data
                graph_points, pairs, valid, connected = self.generate_topology_data(adj_matrix, idx_to_coords, image_size)
                
                # Save files
                # Use sequential index after filtering
                sample_id = f"{len(metadata):06d}"
                
                # Save image
                Image.fromarray(patch['image']).save(split_dir / 'images' / f"{sample_id}.png")
                
                # Save adjacency matrix
                np.save(split_dir / 'adjacency_matrices' / f"{sample_id}_adj.npy", adj_matrix)
                
                # Save masks
                cv2.imwrite(str(split_dir / 'masks' / 'nodes' / f"{sample_id}_nodes.png"), keypoint_mask)
                cv2.imwrite(str(split_dir / 'masks' / 'edges' / f"{sample_id}_edges.png"), edge_mask)
                cv2.imwrite(str(split_dir / 'masks' / 'skeleton' / f"{sample_id}_skeleton.png"), skeleton)
                
                # Save points data
                points_data = {
                    'graph_points': graph_points,
                    'pairs': pairs,
                    'valid': valid,
                    'connected': connected
                }
                with open(split_dir / 'points' / f"{sample_id}_points.pkl", 'wb') as f:
                    pickle.dump(points_data, f)
                
                # Generate and save overlay visualization
                self._save_overlay_visualization(
                    patch['image'], 
                    keypoint_mask, 
                    edge_mask, 
                    graph_points, 
                    adj_matrix,
                    split_dir / 'visualizations' / f"{sample_id}_overlay.png"
                )
                
                # Metadata
                metadata.append({
                    'sample_id': sample_id,
                    'image_filename': f"{sample_id}.png",
                    'adjacency_matrix_filename': f"{sample_id}_adj.npy",
                    'node_mask_filename': f"{sample_id}_nodes.png",
                    'edge_mask_filename': f"{sample_id}_edges.png",
                    'skeleton_mask_filename': f"{sample_id}_skeleton.png",
                    'points_filename': f"{sample_id}_points.pkl",
                    'num_nodes': int(adj_matrix.shape[0]),
                    'num_edges': int(np.sum(adj_matrix) / 2),
                    'region_id': int(patch['region_id']),
                    'patch_coords': [int(patch['patch_coords'][0]), int(patch['patch_coords'][1])]
                })
            
            # Print filtering statistics
            if self.max_nodes is not None:
                print(f"Filtered out {filtered_count} samples with num_nodes > {self.max_nodes}")
            
            # Save metadata
            with open(split_dir / 'metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
            
            dataset_stats[split_name] = len(metadata)
            print(f"{split_name}: {len(metadata)} samples saved")
        
        # Save dataset info
        dataset_info = {
            'total_samples': sum(dataset_stats.values()),
            'image_size': [self.patch_size, self.patch_size],
            'splits': dataset_stats,
            'source': '20 U.S. Cities',
            'patch_extraction': {
                'patch_size': self.patch_size,
                'overlap': self.overlap,
                'stride': self.stride
            },
            'preprocessing': {
                'node_simplification': True,
                'curvature_threshold_degrees': self.curvature_threshold
            }
        }
        
        with open(self.output_path / 'dataset_info.json', 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        print(f"\n{'='*80}")
        print("Dataset generation complete!")
        print(f"{'='*80}")
        print(f"Total samples: {dataset_info['total_samples']}")
        for split, count in dataset_stats.items():
            print(f"  {split}: {count}")
        print(f"Saved to: {self.output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate benchmark dataset from 20 U.S. Cities dataset")
    parser.add_argument("--raw_data_path", type=str, 
                       default="dataset/raw/20cities",
                       help="Path to raw 20cities data")
    parser.add_argument("--output_path", type=str,
                       default="dataset/processed/20cities_benchmark_128x128",
                       help="Path to save benchmark dataset")
    parser.add_argument("--patch_size", type=int, default=128,
                       help="Size of patches to extract")
    parser.add_argument("--overlap", type=int, default=64,
                       help="Overlap between patches (pixels)")
    parser.add_argument("--curvature_threshold", type=float, default=160.0,
                       help="Curvature threshold for node simplification (degrees)")
    parser.add_argument("--max_samples_per_split", type=int, default=None,
                       help="Maximum samples per split (for testing)")
    parser.add_argument("--max-nodes", "--max_nodes", type=int, default=None,
                       dest='max_nodes',
                       help="Filter out samples with more nodes than this (default: no filtering)")
    
    args = parser.parse_args()
    
    generator = TwentyCitiesBenchmarkDatasetGenerator(
        raw_data_path=args.raw_data_path,
        output_path=args.output_path,
        patch_size=args.patch_size,
        overlap=args.overlap,
        max_samples_per_split=args.max_samples_per_split,
        curvature_threshold=args.curvature_threshold,
        max_nodes=args.max_nodes
    )
    
    generator.generate_benchmark_dataset()


if __name__ == "__main__":
    main()

