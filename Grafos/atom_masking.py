from smiles_converter import smiles_to_graph, draw_molecular_graph
from rdkit import Chem
import networkx as nx
import numpy as np

def atom_masking(smiles: str, mask_probability: float):
    """
    Aplica atom masking em um grafo molecular.
    Substitui o símbolo do átomo por 'MASK' com uma probabilidade especificada.
    """
    molecular_graph = smiles_to_graph(smiles=smiles)
    masked_graph = molecular_graph.copy()

    # Mascarar nós aleatoriamente
    for node in masked_graph.nodes():
        if np.random.rand() < mask_probability:
            masked_graph.nodes[node]['symbol'] = '[M]'  
            masked_graph.nodes[node]['atomic_num'] = 0   

    return masked_graph

if __name__ == '__main__':
    masked_graph = atom_masking('N[C]1C(=C([NH])ON=C1)O', mask_probability=0.2)
    draw_molecular_graph(masked_graph)