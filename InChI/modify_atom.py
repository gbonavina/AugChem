import random
from rdkit import Chem
from rdkit.Chem import Draw

# Definição da exceção personalizada para o átomo não encontrado
class AtomError(Exception):
    def __init__(self, message: str = "Não há o átomo especificado nesta molécula.") -> None:
        super().__init__(message)

def change_atoms(smiles: str, changed_atom: str, subs: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("SMILES inválido. Por favor, forneça um SMILES válido.")

    atom_indices = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() == changed_atom]

    if not atom_indices:
        raise AtomError()
    
    random_atom_idx = random.choice(atom_indices)

    mol_editor = Chem.RWMol(mol)
    mol_editor.ReplaceAtom(random_atom_idx, Chem.Atom(subs))
    
    Chem.SanitizeMol(mol_editor)

    return Chem.MolToSmiles(mol_editor)

try:
    modified_smiles = change_atoms('N[C]1C(=C([NH])ON=C1)O', 'C', 'Si')
    print("SMILES da molécula modificada:", modified_smiles)

    # Desenhar a molécula modificada
    # modified_mol = Chem.MolFromSmiles(modified_smiles)
    # img = Draw.MolToImage(modified_mol)
    # img.show()

except AtomError as e:
    print(e)
except ValueError as ve:
    print(ve)
