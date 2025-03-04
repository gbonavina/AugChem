from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import BRICS, Draw
from smiles_converter import smiles_to_data

"""
 2.1.5. Substructure removal
 Substructure removal follows the BRICS decomposition in FP augmentations, where molecular graphs are
 created based on the decomposed fragments from BRICS and are assigned with the same label as the original
 molecule. Fragment graphs contain one or more important functional groups from the molecule, and GNNs
 trained on such augmented data learn to correlate target properties with functional groups
"""

def brics_decompose(smiles: str):
    """
    Realiza a decomposição BRICS de uma molécula representada por SMILES.
    Retorna uma lista de objetos Data do torch_geometric, 
    cada um representando um fragmento da molécula.
    """
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        raise ValueError("SMILES inválido")
    
    fragments = BRICS.BRICSDecompose(mol)
    
    data_list = [smiles_to_data(fragment) for fragment in fragments]
    
    return data_list

if __name__ == '__main__':
    smiles = "CC(=O)Nc1ccc(C(=O)O)cc1"  # Exemplo: Paracetamol
    
    data_list = brics_decompose(smiles)
    
    # Função auxiliar para visualização (usa networkx apenas para desenhar)
    import matplotlib.pyplot as plt
    import networkx as nx
    from torch_geometric.utils import to_networkx

    def draw_molecular_graph(data: Data):
        G = to_networkx(data, node_attrs=['symbol', 'atomic_num'])
        pos = nx.spring_layout(G)
        labels = {node: G.nodes[node]['symbol'] for node in G.nodes()}
        nx.draw(G, pos, labels=labels, with_labels=True, node_color='skyblue', edge_color='gray')
        plt.show()
    
    for fragment_data in data_list:
        draw_molecular_graph(fragment_data)
