import torch
import numpy as np
from rdkit import Chem
from torch_geometric.data import Data
from smiles_converter import smiles_to_data

# Mapeamentos para conversão entre o valor numérico e a ordem da ligação
numeric_to_order = {
    0: 1,  # simples
    1: 2,  # dupla
    2: 3,  # tripla
    3: 1   # aromática (tratada como simples)
}
order_to_numeric = {
    1: 0,  # simples
    2: 1,  # dupla
    3: 2   # tripla
}

def bond_deletion(smiles: str, bond_deletion_probability: float) -> Data:
    """
    Aplica deleção de ligações em um objeto Data do torch_geometric.
    
    Para cada ligação (armazenada em pares):
      - Se a ligação for múltipla (ordem > 1), reduz sua ordem em 1.
      - Se for simples (ordem == 1) ou aromática, remove completamente a ligação.
    
    A função retorna um novo objeto Data com as arestas (e seus tipos) atualizados.
    """
    data = smiles_to_data(smiles)
    
    edge_index_list = data.edge_index.t().tolist()  
    edge_type_list = data.edge_type.tolist()           
    
    new_edge_index = []
    new_edge_type = []
    
    i = 0
    while i < len(edge_index_list):
        bond_edge_1 = edge_index_list[i]
        bond_edge_2 = edge_index_list[i+1]
        bond_numeric = edge_type_list[i]  
        order = numeric_to_order.get(bond_numeric, 1)
        
        # Decide se aplica deleção/modificação para essa ligação
        if np.random.rand() < bond_deletion_probability:
            if order > 1:
                # Reduz a ordem em 1
                new_order = order - 1
                new_numeric = order_to_numeric.get(new_order, 0)
                new_edge_index.append(bond_edge_1)
                new_edge_index.append(bond_edge_2)
                new_edge_type.append(new_numeric)
                new_edge_type.append(new_numeric)
            # Se a ligação for simples (ou aromática), não adiciona, ou seja, remove a ligação.
        else:
            # Mantém a ligação inalterada
            new_edge_index.append(bond_edge_1)
            new_edge_index.append(bond_edge_2)
            new_edge_type.append(bond_numeric)
            new_edge_type.append(bond_numeric)
        
        i += 2  # Processa o próximo par
    
    if new_edge_index:
        new_edge_index_tensor = torch.tensor(new_edge_index, dtype=torch.long).t().contiguous()
        new_edge_type_tensor = torch.tensor(new_edge_type, dtype=torch.long)
    else:
        new_edge_index_tensor = torch.empty((2, 0), dtype=torch.long)
        new_edge_type_tensor = torch.empty((0,), dtype=torch.long)
    
    data.edge_index = new_edge_index_tensor
    data.edge_type = new_edge_type_tensor
    
    return data

if __name__ == '__main__':
    smiles = 'N[C]1C(=C([NH])ON=C1)O'
    modified_data = bond_deletion(smiles, bond_deletion_probability=0.2)
    
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

    draw_molecular_graph(modified_data)
