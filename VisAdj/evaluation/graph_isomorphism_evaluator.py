#!/usr/bin/env python3
"""
Graph Isomorphism Evaluation Metrics

This module provides comprehensive evaluation metrics for the node-link image to 
adjacency matrix conversion task, with special focus on graph isomorphism due to 
permutation invariance.

Key metrics:
- Graph isomorphism checking
- Structural similarity measures
- Node/edge detection accuracy
- Permutation-invariant evaluation
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional, Union
import itertools
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import json
from pathlib import Path

class GraphIsomorphismEvaluator:
    """Evaluates graph isomorphism and structural similarity."""
    
    def __init__(self):
        """Initialize the evaluator."""
        pass
    
    def adjacency_matrix_to_graph(self, adj_matrix: np.ndarray) -> nx.Graph:
        """Convert adjacency matrix to NetworkX graph."""
        G = nx.from_numpy_array(adj_matrix.astype(float))
        return G
    
    def check_graph_isomorphism(self, 
                              adj_matrix1: np.ndarray, 
                              adj_matrix2: np.ndarray) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Check if two graphs are isomorphic.
        
        Args:
            adj_matrix1: First adjacency matrix
            adj_matrix2: Second adjacency matrix
            
        Returns:
            Tuple of (is_isomorphic, permutation_matrix)
        """
        # Remove self-loops before checking isomorphism
        adj_matrix1 = adj_matrix1.copy()
        adj_matrix2 = adj_matrix2.copy()
        np.fill_diagonal(adj_matrix1, 0)
        np.fill_diagonal(adj_matrix2, 0)
        
        G1 = self.adjacency_matrix_to_graph(adj_matrix1)
        G2 = self.adjacency_matrix_to_graph(adj_matrix2)
        
        # Check if graphs have same number of nodes
        if G1.number_of_nodes() != G2.number_of_nodes():
            return False, None
        
        # Check if graphs have same number of edges
        if G1.number_of_edges() != G2.number_of_edges():
            return False, None
        
        # Use NetworkX isomorphism checker
        if nx.is_isomorphic(G1, G2):
            # Find the isomorphism mapping
            matcher = nx.algorithms.isomorphism.GraphMatcher(G1, G2)
            if matcher.is_isomorphic():
                mapping = matcher.mapping
                # Convert mapping to permutation matrix
                n = len(mapping)
                perm_matrix = np.zeros((n, n))
                for i, j in mapping.items():
                    perm_matrix[i, j] = 1
                return True, perm_matrix
        
        return False, None
    
    def find_best_permutation(self, 
                            predicted_adj: np.ndarray, 
                            ground_truth_adj: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Find the best permutation of predicted adjacency matrix to match ground truth.
        
        Args:
            predicted_adj: Predicted adjacency matrix
            ground_truth_adj: Ground truth adjacency matrix
            
        Returns:
            Tuple of (permuted_predicted_adj, similarity_score)
        """
        # Remove self-loops before finding permutation
        predicted_adj = predicted_adj.copy()
        ground_truth_adj = ground_truth_adj.copy()
        np.fill_diagonal(predicted_adj, 0)
        np.fill_diagonal(ground_truth_adj, 0)
        
        n = ground_truth_adj.shape[0]
        
        if predicted_adj.shape != ground_truth_adj.shape:
            # Pad or truncate to match dimensions
            min_size = min(n, predicted_adj.shape[0])
            predicted_adj = predicted_adj[:min_size, :min_size]
            ground_truth_adj = ground_truth_adj[:min_size, :min_size]
            n = min_size
        
        # Try all permutations (only feasible for small graphs)
        if n <= 8:  # Limit to avoid computational explosion
            best_perm = None
            best_score = -1
            
            for perm in itertools.permutations(range(n)):
                perm_matrix = np.array([[1 if i == j else 0 for j in range(n)] for i in perm])
                permuted_pred = perm_matrix @ predicted_adj @ perm_matrix.T
                
                # Calculate similarity score
                score = self._calculate_adjacency_similarity(permuted_pred, ground_truth_adj)
                
                if score > best_score:
                    best_score = score
                    best_perm = perm_matrix
            
            if best_perm is not None:
                permuted_predicted = best_perm @ predicted_adj @ best_perm.T
                return permuted_predicted, best_score
        
        # For larger graphs, use heuristic approach
        return self._heuristic_permutation_matching(predicted_adj, ground_truth_adj)
    
    def _heuristic_permutation_matching(self, 
                                      predicted_adj: np.ndarray, 
                                      ground_truth_adj: np.ndarray) -> Tuple[np.ndarray, float]:
        """Heuristic approach for permutation matching in larger graphs."""
        n = predicted_adj.shape[0]
        
        # Use Hungarian algorithm on degree-based matching
        pred_degrees = np.sum(predicted_adj, axis=1)
        gt_degrees = np.sum(ground_truth_adj, axis=1)
        
        # Create cost matrix based on degree differences
        cost_matrix = np.abs(pred_degrees[:, np.newaxis] - gt_degrees[np.newaxis, :])
        
        # Solve assignment problem
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        # Create permutation matrix
        perm_matrix = np.zeros((n, n))
        for i, j in zip(row_indices, col_indices):
            perm_matrix[i, j] = 1
        
        # Apply permutation
        permuted_predicted = perm_matrix @ predicted_adj @ perm_matrix.T
        
        # Calculate similarity
        similarity = self._calculate_adjacency_similarity(permuted_predicted, ground_truth_adj)
        
        return permuted_predicted, similarity
    
    def _calculate_adjacency_similarity(self, 
                                    adj1: np.ndarray, 
                                    adj2: np.ndarray) -> float:
        """Calculate similarity between two adjacency matrices."""
        # Ensure same shape
        min_size = min(adj1.shape[0], adj2.shape[0])
        adj1 = adj1[:min_size, :min_size]
        adj2 = adj2[:min_size, :min_size]
        
        # Calculate various similarity metrics
        intersection = np.sum(adj1 * adj2)
        union = np.sum(np.maximum(adj1, adj2))
        
        if union == 0:
            return 1.0 if np.sum(adj1) == 0 and np.sum(adj2) == 0 else 0.0
        
        # Jaccard similarity
        jaccard = intersection / union
        
        # Element-wise accuracy
        accuracy = np.mean(adj1 == adj2)
        
        # Weighted combination
        similarity = 0.7 * jaccard + 0.3 * accuracy
        
        return similarity
    
    def evaluate_prediction(self, 
                           predicted_adj: np.ndarray, 
                           ground_truth_adj: np.ndarray) -> Dict[str, float]:
        """
        Comprehensive evaluation of predicted adjacency matrix.
        
        Args:
            predicted_adj: Predicted adjacency matrix
            ground_truth_adj: Ground truth adjacency matrix
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Remove self-loops before evaluation
        predicted_adj = predicted_adj.copy()
        ground_truth_adj = ground_truth_adj.copy()
        np.fill_diagonal(predicted_adj, 0)
        np.fill_diagonal(ground_truth_adj, 0)
        
        metrics = {}
        
        # Basic shape and size metrics
        metrics['num_nodes_pred'] = predicted_adj.shape[0]
        metrics['num_nodes_gt'] = ground_truth_adj.shape[0]
        metrics['num_edges_pred'] = int(np.sum(predicted_adj) / 2)  # Undirected
        metrics['num_edges_gt'] = int(np.sum(ground_truth_adj) / 2)
        
        # Check isomorphism
        is_isomorphic, perm_matrix = self.check_graph_isomorphism(predicted_adj, ground_truth_adj)
        metrics['is_isomorphic'] = float(is_isomorphic)
        
        # Find best permutation and calculate similarity
        permuted_pred, similarity = self.find_best_permutation(predicted_adj, ground_truth_adj)
        metrics['best_similarity'] = similarity
        
        # Element-wise metrics on best permutation
        if permuted_pred.shape == ground_truth_adj.shape:
            metrics['element_accuracy'] = np.mean(permuted_pred == ground_truth_adj)
            metrics['precision'] = self._calculate_precision(permuted_pred, ground_truth_adj)
            metrics['recall'] = self._calculate_recall(permuted_pred, ground_truth_adj)
            metrics['f1_score'] = self._calculate_f1_score(permuted_pred, ground_truth_adj)
        else:
            metrics['element_accuracy'] = 0.0
            metrics['precision'] = 0.0
            metrics['recall'] = 0.0
            metrics['f1_score'] = 0.0
        
        return metrics
    
    def _calculate_precision(self, pred: np.ndarray, gt: np.ndarray) -> float:
        """Calculate precision for edge prediction."""
        true_positives = np.sum(pred * gt)
        false_positives = np.sum(pred * (1 - gt))
        
        if true_positives + false_positives == 0:
            return 1.0 if np.sum(gt) == 0 else 0.0
        
        return true_positives / (true_positives + false_positives)
    
    def _calculate_recall(self, pred: np.ndarray, gt: np.ndarray) -> float:
        """Calculate recall for edge prediction."""
        true_positives = np.sum(pred * gt)
        false_negatives = np.sum((1 - pred) * gt)
        
        if true_positives + false_negatives == 0:
            return 1.0 if np.sum(pred) == 0 else 0.0
        
        return true_positives / (true_positives + false_negatives)
    
    def _calculate_f1_score(self, pred: np.ndarray, gt: np.ndarray) -> float:
        """Calculate F1 score for edge prediction."""
        precision = self._calculate_precision(pred, gt)
        recall = self._calculate_recall(pred, gt)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * precision * recall / (precision + recall)
    
    def evaluate_batch(self, 
                      predictions: List[np.ndarray], 
                      ground_truths: List[np.ndarray]) -> Dict[str, float]:
        """
        Evaluate a batch of predictions.
        
        Args:
            predictions: List of predicted adjacency matrices
            ground_truths: List of ground truth adjacency matrices
            
        Returns:
            Dictionary of aggregated metrics
        """
        if len(predictions) != len(ground_truths):
            raise ValueError("Number of predictions must match number of ground truths")
        
        all_metrics = []
        for pred, gt in zip(predictions, ground_truths):
            metrics = self.evaluate_prediction(pred, gt)
            all_metrics.append(metrics)
        
        # Aggregate metrics
        aggregated = {}
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics]
            aggregated[f'{key}_mean'] = np.mean(values)
            aggregated[f'{key}_std'] = np.std(values)
            aggregated[f'{key}_min'] = np.min(values)
            aggregated[f'{key}_max'] = np.max(values)
        
        return aggregated


def main():
    """Example usage of the GraphIsomorphismEvaluator."""
    
    # Create evaluator
    evaluator = GraphIsomorphismEvaluator()
    
    # Example adjacency matrices
    adj1 = np.array([
        [0, 1, 1, 0],
        [1, 0, 1, 1],
        [1, 1, 0, 0],
        [0, 1, 0, 0]
    ])
    
    # Same graph with permuted nodes
    adj2 = np.array([
        [0, 0, 1, 1],
        [0, 0, 1, 0],
        [1, 1, 0, 1],
        [1, 0, 1, 0]
    ])
    
    # Different graph
    adj3 = np.array([
        [0, 1, 0, 0],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0]
    ])
    
    print("Testing graph isomorphism evaluation...")
    
    # Test isomorphism
    is_iso, perm = evaluator.check_graph_isomorphism(adj1, adj2)
    print(f"Graphs 1 and 2 are isomorphic: {is_iso}")
    
    is_iso, perm = evaluator.check_graph_isomorphism(adj1, adj3)
    print(f"Graphs 1 and 3 are isomorphic: {is_iso}")
    
    # Test evaluation
    metrics = evaluator.evaluate_prediction(adj2, adj1)
    print(f"\nEvaluation metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()

