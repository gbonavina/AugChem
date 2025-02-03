from smiles_converter import smiles_to_graph, draw_molecular_graph
from rdkit import Chem
import networkx as nx
import numpy as np

BOND_ORDER_MAPPING = {
    Chem.rdchem.BondType.SINGLE: 1,
    Chem.rdchem.BondType.DOUBLE: 2,
    Chem.rdchem.BondType.TRIPLE: 3,
    Chem.rdchem.BondType.AROMATIC: 1.5  
}

def bond_deletion(smiles: str, bond_deletion_probability: float):
    """
    Aplica remoção de ligações considerando sua ordem:
    - Se a ligação for múltipla (dupla ou tripla), reduz sua ordem antes de remover completamente.
    - Se for simples, remove diretamente.
    """
    molecular_graph = smiles_to_graph(smiles=smiles)
    modified_graph = molecular_graph.copy()

    bonds = list(modified_graph.edges(data=True))  

    for u, v, data in bonds:
        if np.random.rand() < bond_deletion_probability:
            bond_type = data.get('bond_type', Chem.rdchem.BondType.SINGLE)  
            bond_order = BOND_ORDER_MAPPING.get(bond_type, 1)  

            if bond_order > 1:
                new_bond_type = [bt for bt, order in BOND_ORDER_MAPPING.items() if order == bond_order - 1]
                if new_bond_type:
                    modified_graph[u][v]['bond_type'] = new_bond_type[0]  
            else:
                modified_graph.remove_edge(u, v)

    return modified_graph

if __name__ == '__main__':
    original = smiles_to_graph('N[C]1C(=C([NH])ON=C1)O')
    draw_molecular_graph(original)

    bond_deleted_graph = bond_deletion('N[C]1C(=C([NH])ON=C1)O', bond_deletion_probability=0.2)
    draw_molecular_graph(bond_deleted_graph)