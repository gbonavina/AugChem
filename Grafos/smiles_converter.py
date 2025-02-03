from rdkit import Chem
import networkx as nx
import matplotlib.pyplot as plt

def atomic_num_to_symbol(atomic_num: int):
    periodic_table = Chem.GetPeriodicTable()
    return periodic_table.GetElementSymbol(atomic_num)

def smiles_to_graph(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    molecular_graph = nx.Graph()

    for atom in mol.GetAtoms():
        atomic_num = atom.GetAtomicNum()
        symbol = atomic_num_to_symbol(atomic_num=atomic_num)

        molecular_graph.add_node(atom.GetIdx(), atomic_num=atomic_num, symbol=symbol)
    
    for bond in mol.GetBonds():
        molecular_graph.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond_type=bond.GetBondType())

    return molecular_graph

def draw_molecular_graph(graph):
    pos = nx.spring_layout(graph)

    node_labels = {node: graph.nodes[node]['symbol'] for node in graph.nodes}
    nx.draw_networkx_nodes(graph, pos, node_size=500, node_color='lightblue')
    nx.draw_networkx_labels(graph, pos, labels=node_labels, font_size=12, font_weight='bold')

    edge_labels = {(u, v): graph.edges[u, v]['bond_type'] for u, v in graph.edges}
    nx.draw_networkx_edges(graph, pos, edge_color='gray')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_color='red')

    plt.title("Grafo Molecular")
    plt.axis('off')  
    plt.show()