import numpy as np
from typing import List
import torch
from torch_geometric.data import Data

def edge_dropping(data: Data, drop_rate: float = 0.1) -> Data:
    """
    Remove complete bidirectional edges from the graph (edge dropping)
    
    Args:
        data: torch_geometric graph
        drop_rate: Bidirectional edge removal rate (0.0 to 1.0)
        
    Returns:
        Graph with edges removed
    """
    if data.edge_index.size(1) == 0:
        return data.clone()
    
    edge_set = set()
    for i in range(data.edge_index.size(1)):
        src, dst = data.edge_index[0, i].item(), data.edge_index[1, i].item()
        edge_pair = tuple(sorted([src, dst]))
        edge_set.add(edge_pair)
    
    unique_edges = list(edge_set)
    num_unique_edges = len(unique_edges)
    
    if num_unique_edges == 0:
        return data.clone()
    
    num_to_drop = max(1, int(num_unique_edges * drop_rate))
    
    edges_to_drop = set(unique_edges[:num_to_drop])
    
    keep_mask = []
    for i in range(data.edge_index.size(1)):
        src, dst = data.edge_index[0, i].item(), data.edge_index[1, i].item()
        edge_pair = tuple(sorted([src, dst]))
        
        keep_mask.append(edge_pair not in edges_to_drop)
    
    keep_mask = torch.tensor(keep_mask, dtype=torch.bool)
    
    new_edge_index = data.edge_index[:, keep_mask]
    
    new_edge_attr = None
    if hasattr(data, 'edge_attr') and data.edge_attr is not None and data.edge_attr.size(0) > 0:
        new_edge_attr = data.edge_attr[keep_mask]
    
    new_data = Data(
        x=data.x.clone(),
        edge_index=new_edge_index,
        edge_attr=new_edge_attr,
        num_nodes=data.num_nodes
    )
    
    if hasattr(data, 'y') and data.y is not None:
        new_data.y = data.y.clone()
    
    return new_data

def node_dropping(data: Data, drop_rate: float = 0.1) -> Data:
    """
    Remove nodes randomly from the graph (node dropping)
    
    Args:
        data: torch_geometric graph
        drop_rate: Node removal rate (0.0 to 1.0)
        
    Returns:
        Graph with nodes removed
    """
    if data.num_nodes <= 1:
        return data.clone()
    
    num_nodes = data.num_nodes
    num_to_drop = max(1, int(num_nodes * drop_rate))
    
    nodes_to_keep = torch.randperm(num_nodes)[num_to_drop:]
    nodes_to_keep = torch.sort(nodes_to_keep)[0]
    
    node_mapping = torch.full((num_nodes,), -1, dtype=torch.long)
    node_mapping[nodes_to_keep] = torch.arange(len(nodes_to_keep))
    
    edge_mask = (node_mapping[data.edge_index[0]] >= 0) & (node_mapping[data.edge_index[1]] >= 0)
    
    new_edge_attr = None
    if hasattr(data, 'edge_attr') and data.edge_attr is not None and data.edge_attr.size(0) > 0:
        if edge_mask.sum() > 0:
            new_edge_attr = data.edge_attr[edge_mask]
        else:
            new_edge_attr = torch.empty((0, data.edge_attr.size(1)), dtype=torch.float)
    
    if edge_mask.sum() == 0:
        new_edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        new_edge_index = node_mapping[data.edge_index[:, edge_mask]]
    
    new_data = Data(
        x=data.x[nodes_to_keep],
        edge_index=new_edge_index,
        edge_attr=new_edge_attr,
        num_nodes=len(nodes_to_keep)
    )
    
    if hasattr(data, 'y') and data.y is not None:
        new_data.y = data.y.clone()
    
    return new_data

def feature_masking(data: Data, mask_rate: float = 0.1) -> Data:
    """
    Mask node features randomly (feature masking)
    
    Args:
        data: torch_geometric graph
        mask_rate: Feature masking rate (0.0 to 1.0)
        
    Returns:
        Graph with masked features
    """
    new_data = data.clone()

    mask_value = float('-inf')
    
    if new_data.x.size(0) == 0:
        return new_data
    
    mask = torch.rand_like(new_data.x) < mask_rate
    
    new_data.x = new_data.x.clone()
    new_data.x[mask] = mask_value
    
    return new_data

def edge_perturbation(data: Data, add_rate: float = 0.05, remove_rate: float = 0.05) -> Data:
    """
    Perturb the graph by adding and removing complete bidirectional edges (edge perturbation)
    
    Args:
        data: torch_geometric graph
        add_rate: Bidirectional connection addition rate
        remove_rate: Bidirectional connection removal rate
        
    Returns:
        Perturbed graph
    """
    perturbed_data = edge_dropping(data, remove_rate)
    
    existing_bidirectional = set()
    for i in range(perturbed_data.edge_index.size(1)):
        src, dst = perturbed_data.edge_index[0, i].item(), perturbed_data.edge_index[1, i].item()
        edge_pair = tuple(sorted([src, dst]))
        existing_bidirectional.add(edge_pair)
    
    num_nodes = data.num_nodes
    max_possible_connections = num_nodes * (num_nodes - 1) // 2
    current_connections = len(existing_bidirectional)
    available_connections = max_possible_connections - current_connections
    
    num_connections_to_add = int(available_connections * add_rate)
    
    if num_connections_to_add > 0:
        all_possible_connections = set()
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                all_possible_connections.add((i, j))
        
        available_connections_list = list(all_possible_connections - existing_bidirectional)
        
        if available_connections_list:
            torch.manual_seed(42)
            num_to_add = min(num_connections_to_add, len(available_connections_list))
            
            indices = torch.randperm(len(available_connections_list))[:num_to_add]
            
            new_bidirectional_edges = []
            for idx in indices:
                src, dst = available_connections_list[idx.item()]
                new_bidirectional_edges.extend([[src, dst], [dst, src]])
            
            if new_bidirectional_edges:
                new_edge_index = torch.tensor(new_bidirectional_edges, dtype=torch.long).t()
                
                perturbed_data.edge_index = torch.cat([perturbed_data.edge_index, new_edge_index], dim=1)
                
                if hasattr(perturbed_data, 'edge_attr') and perturbed_data.edge_attr is not None and perturbed_data.edge_attr.size(0) > 0:
                    mean_edge_attr = perturbed_data.edge_attr.mean(dim=0, keepdim=True)
                    new_edge_attrs = mean_edge_attr.repeat(new_edge_index.size(1), 1)
                    perturbed_data.edge_attr = torch.cat([perturbed_data.edge_attr, new_edge_attrs], dim=0)
    
    return perturbed_data

def augment_dataset(graphs: List[Data], augmentation_methods: List[str], 
                          edge_drop_rate: float = 0.1, node_drop_rate: float = 0.1, 
                          feature_mask_rate: float = 0.1, edge_add_rate: float = 0.05,
                          edge_remove_rate: float = 0.05, augment_percentage: float = 0.2, 
                          seed: int = 42) -> List[Data]:
    """
    Apply data augmentation techniques to a list of graphs.
    
    Args:
        graphs: List of torch_geometric Data objects representing the graphs
        augmentation_methods: List of methods ['edge_drop', 'node_drop', 'feature_mask', 'edge_perturb']
        edge_drop_rate: Rate of edge removal (0.0 to 1.0)
        node_drop_rate: Rate of node removal (0.0 to 1.0)
        feature_mask_rate: Rate of feature masking (0.0 to 1.0)
        edge_add_rate: Rate of edge addition for perturbation
        edge_remove_rate: Rate of edge removal for perturbation
        augment_percentage: Size of the augmented dataset as a fraction of the original
        seed: Seed for reproducibility

    Returns:
        List of augmented graphs (original + augmented)

    Raises:
        ValueError: If unknown augmentation methods are specified
    """

    if not graphs:
        raise ValueError("List of graphs cannot be empty")

    valid_methods = {'edge_drop', 'node_drop', 'feature_mask', 'edge_perturb'}
    for method in augmentation_methods:
        if method not in valid_methods:
            raise ValueError(f"Unknown augmentation method: {method}. Valid methods: {valid_methods}")


    rng = np.random.RandomState(seed)
    
    working_graphs = [graph.clone() for graph in graphs]

    target_new_graphs = int(len(graphs) * augment_percentage)
    
    augmented_graphs = []
    augmented_count = 0
    
    
    while augmented_count < target_new_graphs:
        try:
            iteration_augmented: List[Data] = []
            
            for method in augmentation_methods:
                if method == "edge_drop":
                    graph_to_augment = rng.randint(low=0, high=len(working_graphs))
                    original_graph = working_graphs[graph_to_augment]

                    augmented_graph = edge_dropping(
                        original_graph, 
                        drop_rate=edge_drop_rate
                    )
                elif method == "node_drop":
                    graph_to_augment = rng.randint(low=0, high=len(working_graphs))
                    original_graph = working_graphs[graph_to_augment]
                
                    augmented_graph = node_dropping(
                        original_graph, 
                        drop_rate=node_drop_rate
                    )
                elif method == "feature_mask":
                    graph_to_augment = rng.randint(low=0, high=len(working_graphs))
                    original_graph = working_graphs[graph_to_augment]
                

                    augmented_graph = feature_masking(
                        original_graph, 
                        mask_rate=feature_mask_rate,
                    )
                elif method == "edge_perturb":
                    graph_to_augment = rng.randint(low=0, high=len(working_graphs))
                    original_graph = working_graphs[graph_to_augment]
                
                    augmented_graph = edge_perturbation(
                        original_graph, 
                        add_rate=edge_add_rate,
                        remove_rate=edge_remove_rate
                    )
      
                augmented_graph.augmentation_method = method
                augmented_graph.parent_idx = graph_to_augment
                
                iteration_augmented.append(augmented_graph)
            
            unique_augmented = iteration_augmented[:target_new_graphs - augmented_count]
            
            for aug_graph in unique_augmented:
                augmented_graphs.append(aug_graph)
                augmented_count += 1
                
                if augmented_count >= target_new_graphs:
                    break
            
            if augmented_count >= target_new_graphs:
                break
                
        except Exception as e:
            print(f"Error during augmentation: {e}")
            continue

    all_graphs = working_graphs + augmented_graphs

    print(f"Augmenting finished: {len(working_graphs)} originals + {len(augmented_graphs)} augmented = {len(all_graphs)} total")

    return all_graphs