"""
Graph Structure Metrics

Additional metrics for evaluating graph predictions that capture structural properties
beyond isomorphism and edit distance.
"""

import numpy as np
import networkx as nx
from typing import Dict, Tuple, Optional
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon


class GraphStructureMetrics:
    """Compute structural metrics for graph comparison."""
    
    def __init__(self):
        """Initialize the metrics computer."""
        pass
    
    def compute_degree_statistics(self, 
                                  pred_adj: np.ndarray, 
                                  gt_adj: np.ndarray) -> Dict[str, float]:
        """
        Compute degree distribution statistics.
        
        Returns:
            Dictionary with degree statistics
        """
        pred_G = nx.from_numpy_array(pred_adj)
        gt_G = nx.from_numpy_array(gt_adj)
        
        pred_degrees = [d for n, d in pred_G.degree()]
        gt_degrees = [d for n, d in gt_G.degree()]
        
        if len(pred_degrees) == 0:
            pred_degrees = [0]
        if len(gt_degrees) == 0:
            gt_degrees = [0]
        
        # Basic statistics
        metrics = {
            'mean_degree_pred': float(np.mean(pred_degrees)),
            'mean_degree_gt': float(np.mean(gt_degrees)),
            'mean_degree_error': float(np.abs(np.mean(pred_degrees) - np.mean(gt_degrees))),
            'std_degree_pred': float(np.std(pred_degrees)),
            'std_degree_gt': float(np.std(gt_degrees)),
            'max_degree_pred': int(np.max(pred_degrees)) if len(pred_degrees) > 0 else 0,
            'max_degree_gt': int(np.max(gt_degrees)) if len(gt_degrees) > 0 else 0,
        }
        
        # Degree distribution similarity (using histogram)
        max_deg = max(np.max(pred_degrees) if len(pred_degrees) > 0 else 0,
                     np.max(gt_degrees) if len(gt_degrees) > 0 else 0)
        
        if max_deg > 0:
            pred_hist, _ = np.histogram(pred_degrees, bins=min(max_deg + 1, 20), range=(0, max_deg + 1))
            gt_hist, _ = np.histogram(gt_degrees, bins=min(max_deg + 1, 20), range=(0, max_deg + 1))
            
            # Normalize to probabilities
            pred_hist = pred_hist / (np.sum(pred_hist) + 1e-10)
            gt_hist = gt_hist / (np.sum(gt_hist) + 1e-10)
            
            # Jensen-Shannon divergence (symmetric KL divergence)
            js_div = jensenshannon(pred_hist, gt_hist)
            metrics['degree_distribution_js_divergence'] = float(js_div)
            
            # Correlation
            if len(pred_degrees) == len(gt_degrees) and len(pred_degrees) > 1:
                corr = np.corrcoef(pred_degrees, gt_degrees)[0, 1]
                metrics['degree_correlation'] = float(corr) if not np.isnan(corr) else 0.0
            else:
                metrics['degree_correlation'] = 0.0
        else:
            metrics['degree_distribution_js_divergence'] = 0.0
            metrics['degree_correlation'] = 0.0
        
        return metrics
    
    def compute_components_analysis(self, 
                                    pred_adj: np.ndarray, 
                                    gt_adj: np.ndarray) -> Dict[str, float]:
        """
        Analyze connected components.
        
        Returns:
            Dictionary with component statistics
        """
        pred_G = nx.from_numpy_array(pred_adj)
        gt_G = nx.from_numpy_array(gt_adj)
        
        pred_components = list(nx.connected_components(pred_G))
        gt_components = list(nx.connected_components(gt_G))
        
        pred_component_sizes = [len(c) for c in pred_components]
        gt_component_sizes = [len(c) for c in gt_components]
        
        if len(pred_component_sizes) == 0:
            pred_component_sizes = [0]
        if len(gt_component_sizes) == 0:
            gt_component_sizes = [0]
        
        metrics = {
            'num_components_pred': len(pred_components),
            'num_components_gt': len(gt_components),
            'num_components_error': abs(len(pred_components) - len(gt_components)),
            'largest_component_size_pred': int(np.max(pred_component_sizes)) if len(pred_component_sizes) > 0 else 0,
            'largest_component_size_gt': int(np.max(gt_component_sizes)) if len(gt_component_sizes) > 0 else 0,
            'mean_component_size_pred': float(np.mean(pred_component_sizes)) if len(pred_component_sizes) > 0 else 0.0,
            'mean_component_size_gt': float(np.mean(gt_component_sizes)) if len(gt_component_sizes) > 0 else 0.0,
        }
        
        # Check if graphs are connected
        metrics['is_connected_pred'] = nx.is_connected(pred_G) if pred_G.number_of_nodes() > 0 else False
        metrics['is_connected_gt'] = nx.is_connected(gt_G) if gt_G.number_of_nodes() > 0 else False
        
        return metrics
    
    def compute_shortest_path_metrics(self, 
                                     pred_adj: np.ndarray, 
                                     gt_adj: np.ndarray) -> Dict[str, float]:
        """
        Compute shortest path statistics.
        
        Returns:
            Dictionary with path statistics
        """
        pred_G = nx.from_numpy_array(pred_adj)
        gt_G = nx.from_numpy_array(gt_adj)
        
        metrics = {}
        
        # Average shortest path length (only for connected graphs)
        try:
            if nx.is_connected(pred_G) and pred_G.number_of_nodes() > 1:
                pred_avg_path = nx.average_shortest_path_length(pred_G)
                metrics['avg_shortest_path_pred'] = float(pred_avg_path)
            else:
                metrics['avg_shortest_path_pred'] = None
        except:
            metrics['avg_shortest_path_pred'] = None
        
        try:
            if nx.is_connected(gt_G) and gt_G.number_of_nodes() > 1:
                gt_avg_path = nx.average_shortest_path_length(gt_G)
                metrics['avg_shortest_path_gt'] = float(gt_avg_path)
            else:
                metrics['avg_shortest_path_gt'] = None
        except:
            metrics['avg_shortest_path_gt'] = None
        
        # Path length error (if both are connected)
        if metrics['avg_shortest_path_pred'] is not None and metrics['avg_shortest_path_gt'] is not None:
            metrics['avg_shortest_path_error'] = abs(metrics['avg_shortest_path_pred'] - metrics['avg_shortest_path_gt'])
        else:
            metrics['avg_shortest_path_error'] = None
        
        # Diameter (only for connected graphs)
        try:
            if nx.is_connected(pred_G) and pred_G.number_of_nodes() > 1:
                metrics['diameter_pred'] = nx.diameter(pred_G)
            else:
                metrics['diameter_pred'] = None
        except:
            metrics['diameter_pred'] = None
        
        try:
            if nx.is_connected(gt_G) and gt_G.number_of_nodes() > 1:
                metrics['diameter_gt'] = nx.diameter(gt_G)
            else:
                metrics['diameter_gt'] = None
        except:
            metrics['diameter_gt'] = None
        
        return metrics
    
    def compute_clustering_metrics(self, 
                                   pred_adj: np.ndarray, 
                                   gt_adj: np.ndarray) -> Dict[str, float]:
        """
        Compute clustering coefficient statistics.
        
        Returns:
            Dictionary with clustering statistics
        """
        pred_G = nx.from_numpy_array(pred_adj)
        gt_G = nx.from_numpy_array(gt_adj)
        
        metrics = {}
        
        # Average clustering coefficient
        try:
            if pred_G.number_of_nodes() > 0:
                pred_clustering = nx.average_clustering(pred_G)
                metrics['avg_clustering_pred'] = float(pred_clustering)
            else:
                metrics['avg_clustering_pred'] = 0.0
        except:
            metrics['avg_clustering_pred'] = 0.0
        
        try:
            if gt_G.number_of_nodes() > 0:
                gt_clustering = nx.average_clustering(gt_G)
                metrics['avg_clustering_gt'] = float(gt_clustering)
            else:
                metrics['avg_clustering_gt'] = 0.0
        except:
            metrics['avg_clustering_gt'] = 0.0
        
        metrics['avg_clustering_error'] = abs(metrics['avg_clustering_pred'] - metrics['avg_clustering_gt'])
        
        # Number of triangles
        try:
            pred_triangles = sum(nx.triangles(pred_G).values()) // 3
            metrics['num_triangles_pred'] = int(pred_triangles)
        except:
            metrics['num_triangles_pred'] = 0
        
        try:
            gt_triangles = sum(nx.triangles(gt_G).values()) // 3
            metrics['num_triangles_gt'] = int(gt_triangles)
        except:
            metrics['num_triangles_gt'] = 0
        
        metrics['num_triangles_error'] = abs(metrics['num_triangles_pred'] - metrics['num_triangles_gt'])
        
        return metrics
    
    def compute_all_structure_metrics(self, 
                                     pred_adj: np.ndarray, 
                                     gt_adj: np.ndarray) -> Dict[str, float]:
        """
        Compute all structural metrics.
        
        Args:
            pred_adj: Predicted adjacency matrix
            gt_adj: Ground truth adjacency matrix
            
        Returns:
            Dictionary with all structural metrics
        """
        # Ensure binary and symmetric
        pred_adj = (pred_adj > 0.5).astype(float)
        gt_adj = (gt_adj > 0.5).astype(float)
        np.fill_diagonal(pred_adj, 0)
        np.fill_diagonal(gt_adj, 0)
        pred_adj = (pred_adj + pred_adj.T) / 2.0
        gt_adj = (gt_adj + gt_adj.T) / 2.0
        pred_adj = (pred_adj > 0.5).astype(float)
        gt_adj = (gt_adj > 0.5).astype(float)
        
        all_metrics = {}
        
        # Degree statistics
        degree_metrics = self.compute_degree_statistics(pred_adj, gt_adj)
        all_metrics.update({f'degree_{k}': v for k, v in degree_metrics.items()})
        
        # Components analysis
        component_metrics = self.compute_components_analysis(pred_adj, gt_adj)
        all_metrics.update({f'component_{k}': v for k, v in component_metrics.items()})
        
        # Shortest path metrics
        path_metrics = self.compute_shortest_path_metrics(pred_adj, gt_adj)
        all_metrics.update({f'path_{k}': v for k, v in path_metrics.items()})
        
        # Clustering metrics
        clustering_metrics = self.compute_clustering_metrics(pred_adj, gt_adj)
        all_metrics.update({f'clustering_{k}': v for k, v in clustering_metrics.items()})
        
        return all_metrics

