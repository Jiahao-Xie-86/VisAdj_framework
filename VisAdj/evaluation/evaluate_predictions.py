"""
Evaluate Saved Predictions

Evaluates saved predicted adjacency matrices against ground truth using:
- Graph isomorphism rate (primary metric)
- Graph edit distance (primary metric)
- Node/edge detection metrics (secondary)
"""

import numpy as np
import json
import sys
from pathlib import Path
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent
project_root = project_root.resolve()  # Convert to absolute path
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from sam_graph_split.evaluation.graph_isomorphism_evaluator import GraphIsomorphismEvaluator
from sam_graph_split.evaluation.topo_metrics_helper import compute_topo_metrics
import networkx as nx


def evaluate_predictions(
    predictions_dir: str,
    ground_truth_dir: str,
    split: str = 'test',
    compute_ged: bool = False,
    ged_timeout: float = 2.0,
    max_nodes_for_ged: int = 15,
    compute_topo: bool = True,
    topo_k_hops: int = 2,  # Number of hops for subgraph extraction
    topo_max_seeds: int = 50,  # Limit seeds for speed
    topo_seed_sampling: str = 'junctions',  # 'all', 'random', 'junctions'
):
    """
    Evaluate saved predictions against ground truth.
    
    Args:
        predictions_dir: Directory containing predicted adjacency matrices
        ground_truth_dir: Directory containing ground truth adjacency matrices
        split: Dataset split ('test', 'val', 'train')
        compute_ged: Whether to compute graph edit distance (slow)
        ged_timeout: Timeout for GED computation in seconds
        max_nodes_for_ged: Only compute GED for graphs with <= this many nodes
    """
    predictions_dir = Path(predictions_dir)
    ground_truth_dir = Path(ground_truth_dir)
    
    print("=" * 80)
    print("EVALUATION")
    print("=" * 80)
    print(f"Predictions: {predictions_dir}")
    print(f"Ground truth: {ground_truth_dir}")
    print(f"Split: {split}")
    print("=" * 80)
    
    # Load predictions metadata
    pred_metadata_path = predictions_dir / "predictions.json"
    if not pred_metadata_path.exists():
        raise FileNotFoundError(f"Predictions metadata not found: {pred_metadata_path}")
    
    with open(pred_metadata_path, 'r') as f:
        predictions = json.load(f)
    
    # Load ground truth metadata
    # Try multiple possible locations
    gt_metadata_paths = [
        ground_truth_dir / split / "metadata.json",
        ground_truth_dir / "metadata" / f"{split}_metadata.json",
    ]
    gt_metadata_path = None
    for path in gt_metadata_paths:
        if path.exists():
            gt_metadata_path = path
            break
    
    if gt_metadata_path is None:
        raise FileNotFoundError(f"Ground truth metadata not found. Tried: {gt_metadata_paths}")
    
    with open(gt_metadata_path, 'r') as f:
        gt_metadata = json.load(f)
    
    # Create mapping from image filename to ground truth adjacency filename
    gt_map = {}
    for item in gt_metadata:
        image_filename = item.get('image_filename', '')
        if not image_filename:
            # Try alternative keys
            image_filename = item.get('filename', '')
        
        # Try multiple possible keys for adjacency filename
        adj_filename = item.get('adjacency_matrix_filename', '')  # OCTA500 format: 000000.npy
        if not adj_filename:
            adj_filename = item.get('adjacency_filename', '')  # 20cities format: 000000_adj.npy, full_complete: 000000_adjacency.npy
        
        if not adj_filename:
            # Try to construct from image filename (fallback)
            if image_filename:
                # Try multiple formats in order of preference
                image_stem = Path(image_filename).stem
                # Check which format exists in the directory
                possible_formats = [
                    image_stem + '.npy',           # OCTA500: 000000.npy
                    image_stem + '_adj.npy',      # 20cities: 000000_adj.npy
                    image_stem + '_adjacency.npy', # full_complete: 000000_adjacency.npy
                ]
                # We'll check existence later when loading, so just use first format as default
                adj_filename = possible_formats[0]  # Default to .npy format
        
        if image_filename and adj_filename:
            gt_map[image_filename] = adj_filename
    
    # Load ground truth adjacency matrices
    gt_adj_dir = ground_truth_dir / split / "adjacency_matrices"
    if not gt_adj_dir.exists():
        # Try alternative location
        gt_adj_dir = ground_truth_dir / "adjacency_matrices" / split
    if not gt_adj_dir.exists():
        raise FileNotFoundError(f"Ground truth adjacency directory not found. Tried: {ground_truth_dir / split / 'adjacency_matrices'}")
    
    pred_adj_dir = predictions_dir / "adjacency_matrices"
    
    # Initialize evaluator
    evaluator = GraphIsomorphismEvaluator()
    
    # Metrics
    isomorphism_rates = []
    graph_edit_distances = []
    topo_precisions = []
    topo_recalls = []
    topo_f1_scores = []
    all_results = []
    
    print(f"\nEvaluating {len(predictions)} predictions...")
    
    for pred in tqdm(predictions):
        # Find corresponding ground truth
        image_filename = pred.get('image_filename', '')
        if not image_filename:
            # Try to infer from adjacency filename
            adj_filename = pred.get('adjacency_filename', '') or pred.get('adjacency_matrix_filename', '')
            if adj_filename:
                # Handle different filename formats
                if adj_filename.endswith('_adj.npy'):
                    # Format: 000000_adj.npy -> 000000.png
                    image_filename = adj_filename.replace('_adj.npy', '.png')
                elif adj_filename.endswith('_adjacency.npy'):
                    # Format: 000000_adjacency.npy -> 000000.png
                    image_filename = adj_filename.replace('_adjacency.npy', '.png')
                elif adj_filename.endswith('.npy'):
                    # Format: 000000.npy -> 000000.png
                    image_filename = Path(adj_filename).stem + '.png'
        
        if image_filename not in gt_map:
            print(f"Warning: No ground truth found for {image_filename}")
            continue
        
        # Load predicted adjacency
        pred_adj_filename = pred.get('adjacency_filename', '') or pred.get('adjacency_matrix_filename', '')
        if not pred_adj_filename:
            print(f"Warning: No adjacency filename in prediction metadata: {pred}")
            continue
        
        pred_adj_path = pred_adj_dir / pred_adj_filename
        if not pred_adj_path.exists():
            # Try alternative formats if the specified file doesn't exist
            image_stem = Path(image_filename).stem
            alternative_formats = [
                image_stem + '.npy',           # OCTA500: 000000.npy
                image_stem + '_adj.npy',       # 20cities: 000000_adj.npy
                image_stem + '_adjacency.npy', # full_complete: 000000_adjacency.npy
            ]
            
            # Remove the current format from alternatives if it's already in the list
            if pred_adj_filename in alternative_formats:
                alternative_formats.remove(pred_adj_filename)
            
            # Try alternatives
            found = False
            for alt_format in alternative_formats:
                alt_path = pred_adj_dir / alt_format
                if alt_path.exists():
                    pred_adj_path = alt_path
                    pred_adj_filename = alt_format  # Update for consistency
                    found = True
                    break
            
            if not found:
                print(f"Warning: Prediction file not found: {pred_adj_dir / pred_adj_filename} (also tried: {alternative_formats})")
                continue
        
        pred_adj = np.load(pred_adj_path)
        
        # Load ground truth adjacency
        gt_adj_filename = gt_map[image_filename]
        gt_adj_path = gt_adj_dir / gt_adj_filename
        if not gt_adj_path.exists():
            # Try alternative formats if the mapped filename doesn't exist
            image_stem = Path(image_filename).stem
            alternative_formats = [
                image_stem + '.npy',           # OCTA500: 000000.npy
                image_stem + '_adj.npy',       # 20cities: 000000_adj.npy
                image_stem + '_adjacency.npy', # full_complete: 000000_adjacency.npy
            ]
            
            # Remove the current format from alternatives if it's already in the list
            if gt_adj_filename in alternative_formats:
                alternative_formats.remove(gt_adj_filename)
            
            # Try alternatives
            found = False
            for alt_format in alternative_formats:
                alt_path = gt_adj_dir / alt_format
                if alt_path.exists():
                    gt_adj_path = alt_path
                    gt_adj_filename = alt_format  # Update for consistency
                    found = True
                    break
            
            if not found:
                print(f"Warning: Ground truth file not found: {gt_adj_dir / gt_adj_filename} (also tried: {alternative_formats})")
                continue
        
        gt_adj = np.load(gt_adj_path)
        
        # Ensure binary and symmetric, and remove self-loops
        pred_adj_binary = (pred_adj > 0.5).astype(float)
        np.fill_diagonal(pred_adj_binary, 0)  # Remove self-loops
        pred_adj_binary = (pred_adj_binary + pred_adj_binary.T) / 2.0
        pred_adj_binary = (pred_adj_binary > 0.5).astype(float)
        
        gt_adj_binary = gt_adj.astype(float)
        np.fill_diagonal(gt_adj_binary, 0)  # Remove self-loops
        gt_adj_binary = (gt_adj_binary + gt_adj_binary.T) / 2.0
        gt_adj_binary = (gt_adj_binary > 0.5).astype(float)
        
        # PRIMARY METRIC 1: Graph Isomorphism
        is_isomorphic, _ = evaluator.check_graph_isomorphism(
            pred_adj_binary, gt_adj_binary
        )
        isomorphism_rates.append(1.0 if is_isomorphic else 0.0)
        
        # PRIMARY METRIC 2: Graph Edit Distance (optional, slow)
        ged = None
        if compute_ged:
            n_pred = pred_adj_binary.shape[0]
            n_gt = gt_adj_binary.shape[0]
            
            if n_pred <= max_nodes_for_ged and n_gt <= max_nodes_for_ged:
                try:
                    G_pred = evaluator.adjacency_matrix_to_graph(pred_adj_binary)
                    G_gt = evaluator.adjacency_matrix_to_graph(gt_adj_binary)
                    ged = nx.graph_edit_distance(G_pred, G_gt, timeout=ged_timeout)
                except Exception:
                    ged = None
        
        if ged is not None:
            graph_edit_distances.append(float(ged))
        else:
            graph_edit_distances.append(None)
        
        # Compute secondary metrics using evaluator
        metrics = evaluator.evaluate_prediction(pred_adj_binary, gt_adj_binary)
        
        # Compute simplified TOPO metrics if requested
        # This uses graph-distance-based subgraph extraction (no coordinates needed)
        topo_result = None
        if compute_topo:
            topo_result = compute_topo_metrics(
                pred_adj_binary,
                gt_adj_binary,
                k_hops=topo_k_hops,
                num_seeds=topo_max_seeds,
                seed_sampling=topo_seed_sampling,
            )
        
        if topo_result is not None:
            topo_precisions.append(topo_result['precision'])
            topo_recalls.append(topo_result['recall'])
            topo_f1_scores.append(topo_result['f1_score'])
        else:
            topo_precisions.append(None)
            topo_recalls.append(None)
            topo_f1_scores.append(None)
        
        all_results.append({
            'image_filename': image_filename,
            'is_isomorphic': bool(is_isomorphic),
            'graph_edit_distance': float(ged) if ged is not None else None,
            'num_nodes_pred': int(metrics['num_nodes_pred']),
            'num_nodes_gt': int(metrics['num_nodes_gt']),
            'num_edges_pred': int(metrics['num_edges_pred']),
            'num_edges_gt': int(metrics['num_edges_gt']),
            'best_similarity': float(metrics['best_similarity']),
            'element_accuracy': float(metrics['element_accuracy']),
            'precision': float(metrics['precision']),
            'recall': float(metrics['recall']),
            'f1_score': float(metrics['f1_score']),
            'topo_precision': float(topo_result['precision']) if topo_result else None,
            'topo_recall': float(topo_result['recall']) if topo_result else None,
            'topo_f1_score': float(topo_result['f1_score']) if topo_result else None,
        })
    
    # Aggregate metrics
    if len(all_results) == 0:
        print("No valid predictions found!")
        return
    
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    
    # PRIMARY METRICS
    print(f"\n{'='*80}")
    print("PRIMARY METRICS (Permutation-Invariant)")
    print(f"{'='*80}")
    
    # Graph Isomorphism Rate
    isomorphism_rate = np.mean(isomorphism_rates)
    print(f"\n1. Graph Isomorphism Rate:")
    print(f"   Rate: {isomorphism_rate:.4f} ({isomorphism_rate*100:.2f}%)")
    print(f"   Isomorphic: {int(np.sum(isomorphism_rates))} / {len(isomorphism_rates)} samples")
    
    # Graph Edit Distance
    if compute_ged:
        valid_geds = [ged for ged in graph_edit_distances if ged is not None]
        if valid_geds:
            mean_ged = np.mean(valid_geds)
            std_ged = np.std(valid_geds)
            min_ged = np.min(valid_geds)
            max_ged = np.max(valid_geds)
            print(f"\n2. Graph Edit Distance:")
            print(f"   Mean: {mean_ged:.2f} ± {std_ged:.2f}")
            print(f"   Min:  {min_ged:.2f}")
            print(f"   Max:  {max_ged:.2f}")
            print(f"   Valid samples: {len(valid_geds)} / {len(graph_edit_distances)}")
            if len(valid_geds) < len(graph_edit_distances):
                timeout_count = len(graph_edit_distances) - len(valid_geds)
                print(f"   Timeouts/Skipped: {timeout_count} samples")
        else:
            print(f"\n2. Graph Edit Distance:")
            print(f"   N/A (all computations timed out or skipped)")
    
    # TOPO METRICS (if computed)
    if compute_topo:
        valid_topo_precisions = [p for p in topo_precisions if p is not None]
        valid_topo_recalls = [r for r in topo_recalls if r is not None]
        valid_topo_f1s = [f for f in topo_f1_scores if f is not None]
        
        if len(valid_topo_f1s) > 0:
            print(f"\n{'='*80}")
            print("TOPO METRICS (Road Network Topology)")
            print(f"{'='*80}")
            print(f"\nF1 Score:    {np.mean(valid_topo_f1s):.4f} ± {np.std(valid_topo_f1s):.4f}")
            print(f"Precision:   {np.mean(valid_topo_precisions):.4f} ± {np.std(valid_topo_precisions):.4f}")
            print(f"Recall:      {np.mean(valid_topo_recalls):.4f} ± {np.std(valid_topo_recalls):.4f}")
            print(f"Valid samples: {len(valid_topo_f1s)} / {len(topo_f1_scores)}")
            skipped_topo = len(topo_f1_scores) - len(valid_topo_f1s)
            if skipped_topo > 0:
                print(f"Skipped: {skipped_topo} samples (missing coordinates or computation failed)")
        else:
            print(f"\n{'='*80}")
            print("TOPO METRICS")
            print(f"{'='*80}")
            print("N/A (no valid computations)")
    
    # SECONDARY METRICS
    print(f"\n{'='*80}")
    print("SECONDARY METRICS")
    print(f"{'='*80}")
    
    # Aggregate secondary metrics
    node_count_errors = [abs(r['num_nodes_pred'] - r['num_nodes_gt']) for r in all_results]
    edge_count_errors = [abs(r['num_edges_pred'] - r['num_edges_gt']) for r in all_results]
    best_similarities = [r['best_similarity'] for r in all_results]
    element_accuracies = [r['element_accuracy'] for r in all_results]
    precisions = [r['precision'] for r in all_results]
    recalls = [r['recall'] for r in all_results]
    f1_scores = [r['f1_score'] for r in all_results]
    
    print(f"\nNode Count:")
    print(f"  Average error: {np.mean(node_count_errors):.2f} ± {np.std(node_count_errors):.2f}")
    print(f"  Perfect match: {sum(e == 0 for e in node_count_errors)} / {len(node_count_errors)} ({sum(e == 0 for e in node_count_errors)/len(node_count_errors)*100:.1f}%)")
    
    print(f"\nEdge Count:")
    print(f"  Average error: {np.mean(edge_count_errors):.2f} ± {np.std(edge_count_errors):.2f}")
    print(f"  Perfect match: {sum(e == 0 for e in edge_count_errors)} / {len(edge_count_errors)} ({sum(e == 0 for e in edge_count_errors)/len(edge_count_errors)*100:.1f}%)")
    
    print(f"\nStructural Similarity:")
    print(f"  Best similarity: {np.mean(best_similarities):.4f} ± {np.std(best_similarities):.4f}")
    print(f"  Element accuracy: {np.mean(element_accuracies):.4f} ± {np.std(element_accuracies):.4f}")
    
    print(f"\nEdge Prediction (after best permutation):")
    print(f"  Precision: {np.mean(precisions):.4f} ± {np.std(precisions):.4f}")
    print(f"  Recall:    {np.mean(recalls):.4f} ± {np.std(recalls):.4f}")
    print(f"  F1 Score:  {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
    
    # Save results
    results_path = predictions_dir / "evaluation_results.json"
    
    # Prepare primary metrics
    valid_geds = [ged for ged in graph_edit_distances if ged is not None] if compute_ged else []
    primary_metrics = {
        'isomorphism_rate': float(isomorphism_rate),
        'isomorphic_count': int(np.sum(isomorphism_rates)),
        'total_samples': len(isomorphism_rates),
    }
    
    if compute_ged and valid_geds:
        primary_metrics['graph_edit_distance'] = {
            'mean': float(np.mean(valid_geds)),
            'std': float(np.std(valid_geds)),
            'min': float(np.min(valid_geds)),
            'max': float(np.max(valid_geds)),
            'valid_count': len(valid_geds),
            'timeout_count': len(graph_edit_distances) - len(valid_geds),
        }
    
    if compute_topo:
        valid_topo_precisions = [p for p in topo_precisions if p is not None]
        valid_topo_recalls = [r for r in topo_recalls if r is not None]
        valid_topo_f1s = [f for f in topo_f1_scores if f is not None]
        
        if len(valid_topo_f1s) > 0:
            primary_metrics['topo_metrics'] = {
                'f1_score': {
                    'mean': float(np.mean(valid_topo_f1s)),
                    'std': float(np.std(valid_topo_f1s)),
                },
                'precision': {
                    'mean': float(np.mean(valid_topo_precisions)),
                    'std': float(np.std(valid_topo_precisions)),
                },
                'recall': {
                    'mean': float(np.mean(valid_topo_recalls)),
                    'std': float(np.std(valid_topo_recalls)),
                },
                'valid_count': len(valid_topo_f1s),
                'skipped_count': len(topo_f1_scores) - len(valid_topo_f1s),
            }
    
    with open(results_path, 'w') as f:
        json.dump({
            'predictions_dir': str(predictions_dir),
            'ground_truth_dir': str(ground_truth_dir),
            'split': split,
            'num_samples': len(all_results),
            'primary_metrics': primary_metrics,
            'secondary_metrics': {
                'node_count_error': {
                    'mean': float(np.mean(node_count_errors)),
                    'std': float(np.std(node_count_errors)),
                },
                'edge_count_error': {
                    'mean': float(np.mean(edge_count_errors)),
                    'std': float(np.std(edge_count_errors)),
                },
                'best_similarity': {
                    'mean': float(np.mean(best_similarities)),
                    'std': float(np.std(best_similarities)),
                },
                'element_accuracy': {
                    'mean': float(np.mean(element_accuracies)),
                    'std': float(np.std(element_accuracies)),
                },
                'precision': {
                    'mean': float(np.mean(precisions)),
                    'std': float(np.std(precisions)),
                },
                'recall': {
                    'mean': float(np.mean(recalls)),
                    'std': float(np.std(recalls)),
                },
                'f1_score': {
                    'mean': float(np.mean(f1_scores)),
                    'std': float(np.std(f1_scores)),
                },
            },
            'per_sample_results': all_results,
        }, f, indent=2)
    
    print(f"\nResults saved to: {results_path}")
    print("=" * 80)
    
    return primary_metrics


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate saved predictions')
    parser.add_argument('--predictions-dir', type=str, required=True, help='Directory with predictions')
    parser.add_argument('--ground-truth-dir', type=str, required=True, help='Directory with ground truth')
    parser.add_argument('--split', type=str, default='test', choices=['test', 'val', 'train'], help='Dataset split')
    parser.add_argument('--compute-ged', action='store_true', help='Compute graph edit distance (slow)')
    parser.add_argument('--ged-timeout', type=float, default=2.0, help='Timeout for GED computation in seconds')
    parser.add_argument('--max-nodes-for-ged', type=int, default=15, help='Only compute GED for graphs with <= this many nodes')
    parser.add_argument('--compute-topo', action='store_true', default=True, help='Compute TOPO metrics')
    parser.add_argument('--no-compute-topo', dest='compute_topo', action='store_false', help='Disable TOPO metrics computation')
    parser.add_argument('--topo-k-hops', type=int, default=2, help='Number of hops for subgraph extraction (default: 2)')
    parser.add_argument('--topo-max-seeds', type=int, default=50, help='Maximum number of seeds for TOPO computation')
    parser.add_argument('--topo-seed-sampling', type=str, default='junctions', choices=['all', 'random', 'junctions'], help='How to sample seeds: all nodes, random subset, or junctions/endpoints only')
    
    args = parser.parse_args()
    
    evaluate_predictions(
        predictions_dir=args.predictions_dir,
        ground_truth_dir=args.ground_truth_dir,
        split=args.split,
        compute_ged=args.compute_ged,
        ged_timeout=args.ged_timeout,
        max_nodes_for_ged=args.max_nodes_for_ged,
        compute_topo=args.compute_topo,
        topo_k_hops=args.topo_k_hops,
        topo_max_seeds=args.topo_max_seeds,
        topo_seed_sampling=args.topo_seed_sampling,
    )

