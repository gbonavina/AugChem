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

# Defina o SMILES da molécula inicial
smiles = 'N[C]1C(=C([NH])ON=C1)O'
mol = Chem.MolFromSmiles(smiles)

# Inicializa o TautomerEnumerator do MolVS para gerar tautômeros
enumerator = tautomer.TautomerEnumerator()
tautomers = enumerator.enumerate(mol)

# Armazena tautômeros únicos em InChI
taut_inchis = set()
taut_molecules = []

# Gera tautômeros únicos
for idx, taut in enumerate(tautomers):
    try:
        inchi = MolToInchi(taut)
        if inchi not in taut_inchis:
            # Verificar a validade da molécula tautomerizada
            Chem.SanitizeMol(taut)
            taut_inchis.add(inchi)
            taut_molecules.append(taut)
    except Chem.KekulizeException as e:
        print(f"Erro ao kekulizar tautômero {idx + 1}: {e}")
    except Exception as e:
        print(f"Erro ao sanitizar tautômero {idx + 1}: {e}")

# Lista para armazenar todas as protonações dos tautômeros
all_protonated_molecules = []

# Gera protonações para cada tautômero único
for taut_mol in taut_molecules:
    protonated_molecules = generate_protonations(taut_mol)
    all_protonated_molecules.extend(protonated_molecules)

# Armazena as protonações únicas em InChI
unique_protonations = set()
unique_molecules = []

# Filtrar as protonações únicas
for i, protonated_mol in enumerate(all_protonated_molecules):
    try:
        inchi = MolToInchi(protonated_mol)
        if inchi not in unique_protonations:
            unique_protonations.add(inchi)
            unique_molecules.append(protonated_mol)
    except Chem.KekulizeException as e:
        print(f"Erro ao kekulizar molécula protonada: {e}")
    except Exception as e:
        print(f"Erro ao sanitizar molécula protonada: {e}")

# Salva as imagens das variações protonadas
img = Draw.MolsToGridImage(unique_molecules, subImgSize=(300, 300), legends=[f"Protonação {i+1}" for i in range(len(unique_molecules))])
img.save("all_protonations_of_tautomers.png")

# Imprime o número de tautômeros e protonações únicas
print("\nNúmero de tautômeros únicos:", len(taut_molecules))
print("Número de protonações únicas dos tautômeros:", len(unique_protonations))