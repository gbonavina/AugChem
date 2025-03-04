import torch
from rdkit import Chem
from torch_geometric.data import Data
import numpy as np
from smiles_converter import smiles_to_data

def atom_masking(SMILES: str, mask_probability: float) -> Data:
    """
    Aplica atom masking diretamente no objeto Data do torch_geometric.
    
    Para cada nó, com probabilidade mask_probability:
      - O símbolo é substituído por '[M]'
      - O número atômico é substituído por 0
    Após o masking, atualiza também o tensor 'x' com os novos valores.
    """
    data = smiles_to_data(SMILES)

    new_symbols = []
    new_atomic_nums = []
    
    for i in range(data.num_nodes):
        if np.random.rand() < mask_probability:
            new_symbols.append('[M]')
            new_atomic_nums.append(0)
        else:
            new_symbols.append(data.symbol[i])
            new_atomic_nums.append(int(data.atomic_num[i]))
    
    # Atualiza os atributos do objeto Data
    data.symbol = new_symbols
    data.atomic_num = torch.tensor(new_atomic_nums, dtype=torch.long)
    data.x = torch.tensor(new_atomic_nums, dtype=torch.float).view(-1, 1)
    
    return data

if __name__ == '__main__':
    smiles = 'N[C]1C(=C([NH])ON=C1)O'
    masked_data = atom_masking(smiles, mask_probability=0.2)
    
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

    draw_molecular_graph(masked_data)