# Graph Examples

This section provides practical examples of using AugChem for graph-based molecular data augmentation.

## Example 1: Basic Graph Augmentation

```python
import torch
from torch_geometric.data import Data
from augchem.modules.graph.graphs_modules import augment_dataset

# Create sample molecular graphs
def create_sample_graph(num_nodes, num_edges):
    x = torch.randn(num_nodes, 9)  # 9 node features
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_attr = torch.randn(num_edges, 4)  # 4 edge features
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

# Create a small dataset
graphs = [
    create_sample_graph(10, 18),
    create_sample_graph(12, 22),
    create_sample_graph(8, 14),
    create_sample_graph(15, 28)
]

# Apply augmentation
augmented_graphs = augment_dataset(
    graphs=graphs,
    augmentation_methods=['edge_drop', 'node_drop', 'feature_mask'],
    edge_drop_rate=0.1,
    node_drop_rate=0.05,
    feature_mask_rate=0.15,
    augment_percentage=0.5,  # 50% more data
    seed=42
)

print(f"Dataset expanded from {len(graphs)} to {len(augmented_graphs)} graphs")
```

## Example 2: Processing Real Molecular Data

```python
from rdkit import Chem
import torch
from torch_geometric.data import Data
from augchem.modules.graph.graphs_modules import augment_dataset

def smiles_to_graph(smiles):
    """Convert SMILES to PyTorch Geometric graph"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # Node features
    atom_features = []
    for atom in mol.GetAtoms():
        features = [
            atom.GetAtomicNum(),
            atom.GetDegree(),
            atom.GetFormalCharge(),
            int(atom.GetHybridization()),
            int(atom.GetIsAromatic()),
            atom.GetNumRadicalElectrons(),
            atom.GetTotalNumHs(),
            int(atom.IsInRing()),
            atom.GetMass()
        ]
        atom_features.append(features)
    
    # Edge features
    edge_indices = []
    edge_attrs = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_indices.extend([[i, j], [j, i]])  # Bidirectional
        
        bond_features = [
            bond.GetBondTypeAsDouble(),
            int(bond.GetIsAromatic()),
            int(bond.IsInRing()),
            int(bond.GetIsConjugated())
        ]
        edge_attrs.extend([bond_features, bond_features])
    
    return Data(
        x=torch.tensor(atom_features, dtype=torch.float),
        edge_index=torch.tensor(edge_indices, dtype=torch.long).t().contiguous(),
        edge_attr=torch.tensor(edge_attrs, dtype=torch.float)
    )

# Example molecules
molecules = [
    "CCO",  # Ethanol
    "CC(=O)O",  # Acetic acid
    "c1ccccc1",  # Benzene
    "CCN(CC)CC",  # Triethylamine
    "CC(C)O"  # Isopropanol
]

# Convert to graphs
graphs = [smiles_to_graph(smiles) for smiles in molecules]
graphs = [g for g in graphs if g is not None]

# Augment the dataset
augmented = augment_dataset(
    graphs=graphs,
    augmentation_methods=['edge_drop', 'feature_mask', 'edge_perturb'],
    augment_percentage=1.0,  # Double the dataset
    seed=42
)

print(f"Augmented {len(molecules)} molecules to {len(augmented)} graphs")
```

## Example 3: Integration with Machine Learning

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split

# Simple GNN for molecular property prediction
class MolecularGNN(nn.Module):
    def __init__(self, num_node_features, hidden_dim=64):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv3(x, edge_index))
        
        # Global pooling
        x = global_mean_pool(x, batch)
        
        return self.classifier(x)

# Prepare data with augmentation
original_graphs = graphs  # From previous example
augmented_graphs = augment_dataset(
    original_graphs,
    augmentation_methods=['edge_drop', 'node_drop', 'feature_mask'],
    augment_percentage=0.3
)

# Add dummy targets for demonstration
for graph in augmented_graphs:
    graph.y = torch.randn(1)  # Random property value

# Split data
train_graphs, test_graphs = train_test_split(
    augmented_graphs, test_size=0.2, random_state=42
)

# Create data loaders
train_loader = DataLoader(train_graphs, batch_size=16, shuffle=True)
test_loader = DataLoader(test_graphs, batch_size=16, shuffle=False)

# Initialize model
model = MolecularGNN(num_node_features=9)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Training loop
model.train()
for epoch in range(10):
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(out, batch.y.view(-1, 1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1:2d}, Loss: {total_loss/len(train_loader):.4f}")

print("Training completed!")
```

## Example 4: Comparative Analysis

```python
import matplotlib.pyplot as plt
import numpy as np

def analyze_augmentation_impact(original_graphs, augmented_graphs):
    """Analyze the impact of augmentation on dataset statistics"""
    
    # Extract statistics
    def get_stats(graphs):
        nodes = [g.num_nodes for g in graphs]
        edges = [g.edge_index.size(1) for g in graphs]
        return nodes, edges
    
    orig_nodes, orig_edges = get_stats(original_graphs)
    aug_nodes, aug_edges = get_stats(augmented_graphs)
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Node distribution
    axes[0, 0].hist(orig_nodes, bins=20, alpha=0.7, label='Original', color='blue')
    axes[0, 0].hist(aug_nodes, bins=20, alpha=0.7, label='Augmented', color='red')
    axes[0, 0].set_xlabel('Number of Nodes')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Node Count Distribution')
    axes[0, 0].legend()
    
    # Edge distribution
    axes[0, 1].hist(orig_edges, bins=20, alpha=0.7, label='Original', color='blue')
    axes[0, 1].hist(aug_edges, bins=20, alpha=0.7, label='Augmented', color='red')
    axes[0, 1].set_xlabel('Number of Edges')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Edge Count Distribution')
    axes[0, 1].legend()
    
    # Node vs Edge scatter
    axes[1, 0].scatter(orig_nodes, orig_edges, alpha=0.6, label='Original', color='blue')
    axes[1, 0].scatter(aug_nodes, aug_edges, alpha=0.6, label='Augmented', color='red')
    axes[1, 0].set_xlabel('Number of Nodes')
    axes[1, 0].set_ylabel('Number of Edges')
    axes[1, 0].set_title('Nodes vs Edges Relationship')
    axes[1, 0].legend()
    
    # Statistics summary
    stats_text = f"""Dataset Statistics:
    
Original Dataset:
• Graphs: {len(original_graphs)}
• Avg nodes: {np.mean(orig_nodes):.1f} ± {np.std(orig_nodes):.1f}
• Avg edges: {np.mean(orig_edges):.1f} ± {np.std(orig_edges):.1f}

Augmented Dataset:
• Graphs: {len(augmented_graphs)}
• Avg nodes: {np.mean(aug_nodes):.1f} ± {np.std(aug_nodes):.1f}
• Avg edges: {np.mean(aug_edges):.1f} ± {np.std(aug_edges):.1f}

Augmentation Impact:
• Size increase: {((len(augmented_graphs)/len(original_graphs))-1)*100:.1f}%
• Node diversity: {(np.std(aug_nodes)/np.std(orig_nodes)-1)*100:+.1f}%
• Edge diversity: {(np.std(aug_edges)/np.std(orig_edges)-1)*100:+.1f}%"""
    
    axes[1, 1].text(0.05, 0.95, stats_text, transform=axes[1, 1].transAxes,
                   fontsize=10, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].axis('off')
    axes[1, 1].set_title('Dataset Comparison')
    
    plt.tight_layout()
    plt.show()

# Run analysis
analyze_augmentation_impact(graphs, augmented_graphs)
```

## Example 5: Advanced Augmentation Strategy

```python
from augchem.modules.graph.graphs_modules import (
    edge_dropping, node_dropping, feature_masking, edge_perturbation
)

def create_diverse_augmentations(graph, num_variants=5):
    """Create diverse augmentations of a single graph"""
    
    variants = []
    
    for i in range(num_variants):
        # Randomly select augmentation parameters
        edge_drop_rate = np.random.uniform(0.05, 0.15)
        node_drop_rate = np.random.uniform(0.02, 0.08)
        mask_rate = np.random.uniform(0.10, 0.20)
        
        # Apply different combinations
        if i % 3 == 0:
            # Edge-focused augmentation
            variant = edge_dropping(graph, drop_rate=edge_drop_rate)
            variant = feature_masking(variant, mask_rate=mask_rate)
            variant.augmentation_type = "edge_focused"
            
        elif i % 3 == 1:
            # Node-focused augmentation
            variant = node_dropping(graph, drop_rate=node_drop_rate)
            variant = feature_masking(variant, mask_rate=mask_rate)
            variant.augmentation_type = "node_focused"
            
        else:
            # Perturbation-focused augmentation
            variant = edge_perturbation(graph, add_rate=0.03, remove_rate=0.05)
            variant = feature_masking(variant, mask_rate=mask_rate)
            variant.augmentation_type = "perturbation_focused"
        
        variants.append(variant)
    
    return variants

# Apply diverse augmentation to each graph
diverse_augmented = []
for i, graph in enumerate(graphs):
    variants = create_diverse_augmentations(graph, num_variants=3)
    diverse_augmented.extend(variants)
    print(f"Graph {i}: created {len(variants)} variants")

print(f"Total augmented graphs: {len(diverse_augmented)}")

# Analyze augmentation types
augmentation_types = [g.augmentation_type for g in diverse_augmented]
type_counts = {t: augmentation_types.count(t) for t in set(augmentation_types)}
print("Augmentation type distribution:", type_counts)
```

## Example 6: Individual Augmentation Techniques

```python
from augchem.modules.graph.graphs_modules import (
    edge_dropping, node_dropping, feature_masking, edge_perturbation
)

# Use a sample graph for demonstration
sample_graph = graphs[0]  # Ethanol graph from previous example

print("Original graph statistics:")
print(f"  Nodes: {sample_graph.num_nodes}")
print(f"  Edges: {sample_graph.edge_index.size(1)}")
print(f"  Node features shape: {sample_graph.x.shape}")
print(f"  Edge features shape: {sample_graph.edge_attr.shape}")
print()

# 1. Edge Dropping
print("1. Edge Dropping:")
for drop_rate in [0.1, 0.2, 0.3]:
    edge_dropped = edge_dropping(sample_graph, drop_rate=drop_rate)
    edges_removed = sample_graph.edge_index.size(1) - edge_dropped.edge_index.size(1)
    print(f"  Drop rate {drop_rate:.1f}: {sample_graph.edge_index.size(1)} -> {edge_dropped.edge_index.size(1)} edges ({edges_removed} removed)")

print()

# 2. Node Dropping
print("2. Node Dropping:")
for drop_rate in [0.05, 0.1, 0.15]:
    node_dropped = node_dropping(sample_graph, drop_rate=drop_rate)
    nodes_removed = sample_graph.num_nodes - node_dropped.num_nodes
    print(f"  Drop rate {drop_rate:.2f}: {sample_graph.num_nodes} -> {node_dropped.num_nodes} nodes ({nodes_removed} removed)")

print()

# 3. Feature Masking
print("3. Feature Masking:")
for mask_rate in [0.1, 0.2, 0.3]:
    feature_masked = feature_masking(sample_graph, mask_rate=mask_rate)
    total_features = feature_masked.x.numel()
    masked_features = (feature_masked.x == float('-inf')).sum().item()
    print(f"  Mask rate {mask_rate:.1f}: {masked_features}/{total_features} features masked ({masked_features/total_features*100:.1f}%)")

print()

# 4. Edge Perturbation
print("4. Edge Perturbation:")
perturbation_configs = [
    (0.05, 0.05),
    (0.1, 0.1),
    (0.03, 0.07)
]

for add_rate, remove_rate in perturbation_configs:
    edge_perturbed = edge_perturbation(sample_graph, add_rate=add_rate, remove_rate=remove_rate)
    edge_change = edge_perturbed.edge_index.size(1) - sample_graph.edge_index.size(1)
    print(f"  Add {add_rate:.2f}, Remove {remove_rate:.2f}: {sample_graph.edge_index.size(1)} -> {edge_perturbed.edge_index.size(1)} edges ({edge_change:+d} net change)")
```

## Example 7: Batch Processing and Quality Control

```python
from torch_geometric.loader import DataLoader

def validate_graph_quality(graphs):
    """Check graph quality after augmentation"""
    
    issues = []
    
    for i, graph in enumerate(graphs):
        # Check for isolated nodes
        edge_index = graph.edge_index
        if edge_index.size(1) > 0:
            connected_nodes = torch.unique(edge_index.flatten())
            isolated_nodes = graph.num_nodes - len(connected_nodes)
            if isolated_nodes > 0:
                issues.append(f"Graph {i}: {isolated_nodes} isolated nodes")
        
        # Check for self-loops
        if edge_index.size(1) > 0:
            self_loops = (edge_index[0] == edge_index[1]).sum().item()
            if self_loops > 0:
                issues.append(f"Graph {i}: {self_loops} self-loops")
        
        # Check for negative features (from masking)
        if torch.any(graph.x == float('-inf')):
            masked_count = (graph.x == float('-inf')).sum().item()
            issues.append(f"Graph {i}: {masked_count} masked features")
    
    return issues

# Create larger dataset for demonstration
larger_graphs = []
for i in range(20):
    smiles_list = [
        "CCO", "CC(=O)O", "c1ccccc1", "CCN(CC)CC", "CC(C)O",
        "CC(C)(C)O", "CC=O", "C1CCCCC1", "c1ccc2ccccc2c1", "CCCCO"
    ]
    graph = smiles_to_graph(smiles_list[i % len(smiles_list)])
    if graph is not None:
        larger_graphs.append(graph)

print(f"Created dataset with {len(larger_graphs)} graphs")

# Apply batch augmentation
batch_augmented = augment_dataset(
    graphs=larger_graphs,
    augmentation_methods=['edge_drop', 'node_drop', 'feature_mask', 'edge_perturb'],
    edge_drop_rate=0.12,
    node_drop_rate=0.08,
    feature_mask_rate=0.20,
    edge_add_rate=0.05,
    edge_remove_rate=0.05,
    augment_percentage=0.6,
    seed=42
)

print(f"Batch augmentation: {len(larger_graphs)} -> {len(batch_augmented)} graphs")

# Quality validation
print("\nQuality validation:")
original_issues = validate_graph_quality(larger_graphs)
augmented_issues = validate_graph_quality(batch_augmented)

print(f"Original dataset issues: {len(original_issues)}")
if original_issues:
    for issue in original_issues[:5]:  # Show first 5 issues
        print(f"  {issue}")

print(f"Augmented dataset issues: {len(augmented_issues)}")
if augmented_issues:
    for issue in augmented_issues[:5]:  # Show first 5 issues
        print(f"  {issue}")

# Create DataLoader for training
train_loader = DataLoader(batch_augmented, batch_size=32, shuffle=True)

print(f"\nCreated DataLoader with batch size 32")
print(f"Number of batches: {len(train_loader)}")

# Examine first batch
for batch in train_loader:
    print(f"First batch: {batch.num_graphs} graphs, {batch.x.size(0)} total nodes")
    break
```

## Example 8: Real-World Drug Discovery Application

```python
# Simulate a more complex drug discovery scenario
import pandas as pd

# Create realistic molecular dataset
drug_molecules = [
    "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O",        # Ibuprofen
    "CC(=O)Oc1ccccc1C(=O)O",                   # Aspirin
    "CC(=O)Nc1ccc(cc1)O",                      # Paracetamol
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",           # Caffeine
    "Clc1ccc(cc1)C(c2ccccc2)N3CCCC3",         # Loratadine
    "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C",    # Testosterone
    "CCN(CC)CCNC(=O)c1cc(ccc1OC)S(=O)(=O)N",  # Sulpiride
    "Cc1ccc(cc1)C(=O)c2ccc(cc2)N(C)C",        # Michler's ketone
]

# Convert to graphs
drug_graphs = []
drug_names = ["Ibuprofen", "Aspirin", "Paracetamol", "Caffeine", 
              "Loratadine", "Testosterone", "Sulpiride", "Michler's ketone"]

for smiles, name in zip(drug_molecules, drug_names):
    graph = smiles_to_graph(smiles)
    if graph is not None:
        graph.name = name
        graph.y = torch.randn(1)  # Simulated activity
        drug_graphs.append(graph)

print(f"Drug discovery dataset: {len(drug_graphs)} compounds")

# Apply pharmaceutical-grade augmentation
pharma_augmented = augment_dataset(
    graphs=drug_graphs,
    augmentation_methods=['edge_drop', 'feature_mask', 'edge_perturb'],
    edge_drop_rate=0.08,      # Conservative for drugs
    feature_mask_rate=0.12,   # Preserve chemical meaning
    edge_add_rate=0.03,       # Minimal structural changes
    edge_remove_rate=0.05,
    augment_percentage=0.4,   # 40% expansion
    seed=42
)

print(f"Pharmaceutical augmentation: {len(drug_graphs)} -> {len(pharma_augmented)} compounds")

# Analyze by original compound
print("\nAugmentation breakdown by compound:")
compound_counts = {}
for graph in pharma_augmented:
    if hasattr(graph, 'name'):
        compound_counts[graph.name] = compound_counts.get(graph.name, 0) + 1
    else:
        # This is an augmented graph, try to find parent
        compound_counts['Augmented'] = compound_counts.get('Augmented', 0) + 1

for compound, count in compound_counts.items():
    print(f"  {compound}: {count} variants")

# Prepare for virtual screening
virtual_library = DataLoader(pharma_augmented, batch_size=16, shuffle=False)

print(f"\nVirtual screening library prepared:")
print(f"  Total compounds: {len(pharma_augmented)}")
print(f"  Batches for screening: {len(virtual_library)}")

# Simulate screening results
screening_results = []
for batch in virtual_library:
    # Simulate screening scores
    scores = torch.randn(batch.num_graphs)
    screening_results.extend(scores.tolist())

print(f"  Screening completed: {len(screening_results)} scores generated")
print(f"  Top hit score: {max(screening_results):.3f}")
print(f"  Mean score: {sum(screening_results)/len(screening_results):.3f}")
```

These examples demonstrate the comprehensive capabilities of AugChem's graph augmentation toolkit for molecular research, from basic data expansion to sophisticated drug discovery applications.
