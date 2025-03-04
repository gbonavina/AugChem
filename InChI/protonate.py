from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.inchi import MolToInchi
from rdkit import RDLogger
from molvs import tautomer
import itertools

RDLogger.DisableLog('rdApp.*')

# Função para gerar todos os estados de protonação possíveis
def generate_protonations(mol):
    protonation_states = []

    # Obtém os átomos que podem ser protonados (neste caso, nitrogênios e oxigênios)
    atoms_to_protonate = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() in ['N', 'O']]

    # Gera todas as combinações possíveis de átomos a serem protonados
    for i in range(1, len(atoms_to_protonate) + 1):
        for combination in itertools.combinations(atoms_to_protonate, i):
            # Faz uma cópia da molécula original para cada combinação de protonação
            new_mol = Chem.Mol(mol)

            for atom_idx in combination:
                atom = new_mol.GetAtomWithIdx(atom_idx)
                # Adiciona uma carga formal +1 para cada átomo na combinação
                atom.SetFormalCharge(atom.GetFormalCharge() + 1)

            # Verificação de validade da molécula
            try:
                # Sanitiza a molécula para garantir que é quimicamente válida
                Chem.SanitizeMol(new_mol)
                protonation_states.append(new_mol)
            except Chem.rdchem.KekulizeException as e:
                print(f"Erro ao kekulizar molécula protonada: {e}")
            except Exception as e:
                print(f"Erro ao sanitizar molécula protonada: {e}")

    return protonation_states