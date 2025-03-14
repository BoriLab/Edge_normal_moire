# %%
import numpy as np
print(np.__version__)


# %%
import os
import random
import numpy as np
import torch
import pickle
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd

# RDKit을 이용하여 scaffold를 추출한다고 가정 (실제 QM9에는 SMILES가 제공됨)
from rdkit import Chem
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles

##############################################################################
# 시드 고정 함수 추가
##############################################################################
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Function to convert scientific notation in QM9 data to float
def convert_to_float(value):
    try:
        return float(value.replace("*^", "e"))
    except ValueError:
        return float("nan")



def read_xyz(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()
        # Number of atoms
        num_atoms = int(lines[0].strip())
        
        # 첫 번째 코멘트 라인: thermochemical properties 등
        comment = lines[1].strip().split()
        gdb_info = {
            "tag": comment[0],
            "index": int(comment[1]),
            "A": convert_to_float(comment[2]),
            "B": convert_to_float(comment[3]),
            "C": convert_to_float(comment[4]),
            "mu": convert_to_float(comment[5]),
            "alpha": convert_to_float(comment[6]),
            "homo": convert_to_float(comment[7]),
            "lumo": convert_to_float(comment[8]),
            "gap": convert_to_float(comment[9]),
            "r2": convert_to_float(comment[10]),
            "zpve": convert_to_float(comment[11]),
            "U0": convert_to_float(comment[12]),
            "U": convert_to_float(comment[13]),
            "H": convert_to_float(comment[14]),
            "G": convert_to_float(comment[15]),
            "Cv": convert_to_float(comment[16]),
        }
        
        # Atomic coordinates (lines 3 ~ 2+num_atoms)
        data = []
        for line in lines[2 : 2 + num_atoms]:
            parts = line.split()
            atom = parts[0]
            x, y, z = map(convert_to_float, parts[1:4])
            data.append([atom, x, y, z])
        df = pd.DataFrame(data, columns=["Atom", "X", "Y", "Z"])
        
        # 나머지 라인들 처리:
        # 예를 들어, 파일의 마지막 라인(여기서는 10번째 라인)이 InChI 정보를 담고 있다고 가정
        # 실제 파일 형식에 따라 이 인덱스를 조정하세요.
        if len(lines) >= 2 + num_atoms + 3:
            # 예: 10번째 라인에 InChI 문자열이 탭으로 구분되어 있다.
            inchi_line = lines[2 + num_atoms + 2].strip().split()
            if inchi_line:
                inchi = inchi_line[0]
                mol = Chem.MolFromInchi(inchi)
                smiles = Chem.MolToSmiles(mol) if mol is not None else ""
            else:
                smiles = ""
        else:
            smiles = ""
        
        gdb_info["smiles"] = smiles
        
        return num_atoms, gdb_info, df


def make_adjacency_by_distance(coor, threshold=1.5):
    n = coor.shape[0]
    adj = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(coor[i] - coor[j])
            if d < threshold:  # 연결 조건
                adj[i, j] = d
                adj[j, i] = d
    return adj

def extract_edge_info(coor, thresholds=[1.5, 2.0, 2.5]):
    edge_index = []
    edge_attr = []
    n = coor.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            distance = np.linalg.norm(coor[i] - coor[j])
            if any(distance < t for t in thresholds):
                edge_index.append([i, j])
                edge_index.append([j, i])
                edge_attr.append([1 if distance < t else 0 for t in thresholds])
                edge_attr.append([1 if distance < t else 0 for t in thresholds])
    edge_index = np.array(edge_index).T
    edge_attr = np.array(edge_attr)
    return edge_index, edge_attr

def one_hot_encode_nodes(nodes):
    flattened_nodes = [atom for molecule in nodes for atom in molecule]
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(flattened_nodes)
    onehot_encoder = OneHotEncoder(sparse_output=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    idx = 0
    one_hot_nodes = []
    for molecule in nodes:
        one_hot_nodes.append(onehot_encoded[idx : idx + len(molecule)])
        idx += len(molecule)
    return one_hot_nodes

def get_scaffold(smiles):
    """Extract Bemis–Murcko scaffold for a given SMILES string using RDKit."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ""
        scaffold = MurckoScaffoldSmiles(mol=mol, includeChirality=True)
        return scaffold
    except Exception as e:
        return ""

def read_xyz_directory(directory_path):
    filename, nodes, coords, adjs, edge_indices, edge_attrs, targets, scaffolds = [], [], [], [], [], [], [], []
    for file in os.listdir(directory_path):
        if file.endswith(".xyz"):
            num_atoms, gdb_info, df = read_xyz(os.path.join(directory_path, file))
            filename.append(file)
            nodes.append(df["Atom"].values)
            coords.append(df[["X", "Y", "Z"]].values)
            adjs.append(make_adjacency_by_distance(df[["X", "Y", "Z"]].values))
            edge_index, edge_attr = extract_edge_info(df[["X", "Y", "Z"]].values)
            edge_indices.append(edge_index)
            edge_attrs.append(edge_attr)
            targets.append(gdb_info["gap"])
            # Compute scaffold using SMILES info (gdb_info["smiles"] must be provided)
            scaffolds.append(get_scaffold(gdb_info["smiles"]))
    one_hot_nodes = one_hot_encode_nodes(nodes)
    return filename, nodes, one_hot_nodes, coords, adjs, edge_indices, edge_attrs, targets, scaffolds

def save_qm9_data_to_pickle(data, file_path="T_qm9_data_with_edges_scaffold.pkl"):
    with open(file_path, "wb") as file:
        pickle.dump(data, file)

##############################################################################
# Scaffold splitting 함수
##############################################################################
def scaffold_split(data, test_ratio=0.1, val_ratio=0.1, seed=42):
    random.seed(seed)
    num_samples = len(data["filename"])
    # Group indices by scaffold
    scaffold_to_indices = {}
    for i, scaffold in enumerate(data["scaffolds"]):
        if scaffold == "":
            scaffold = "None"  # 없는 경우 하나의 그룹으로 처리
        scaffold_to_indices.setdefault(scaffold, []).append(i)
    
    # Sort scaffold groups by size (largest first)
    scaffold_groups = sorted(scaffold_to_indices.values(), key=lambda x: len(x), reverse=True)
    
    test_cutoff = int(num_samples * test_ratio)
    val_cutoff = int(num_samples * val_ratio)
    
    test_indices, val_indices, train_indices = [], [], []
    for group in scaffold_groups:
        # Greedily assign entire group to a split based on remaining quota
        if len(test_indices) + len(group) <= test_cutoff:
            test_indices.extend(group)
        elif len(val_indices) + len(group) <= val_cutoff:
            val_indices.extend(group)
        else:
            train_indices.extend(group)
    
    return train_indices, val_indices, test_indices

class QM9Dataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data["filename"])

    def __getitem__(self, idx):
        return {
            "filename": self.data["filename"][idx],
            "nodes": self.data["nodes"][idx],
            "one_hot_nodes": self.data["one_hot_nodes"][idx],
            "coords": self.data["coords"][idx],
            "adjs": self.data["adjs"][idx],
            "edge_index": self.data["edge_indices"][idx],
            "edge_attr": self.data["edge_attrs"][idx],
            "targets": self.data["targets"][idx],
            "scaffold": self.data["scaffolds"][idx],
        }

def collate_fn(batch):
    return {
        "filename": [data["filename"] for data in batch],
        "nodes": [data["nodes"] for data in batch],
        "one_hot_nodes": [data["one_hot_nodes"] for data in batch],
        "coords": [data["coords"] for data in batch],
        "adjs": [data["adjs"] for data in batch],
        "edge_indices": [data["edge_index"] for data in batch],
        "edge_attrs": [torch.tensor(data["edge_attr"], dtype=torch.float32) for data in batch],
        "targets": torch.tensor([data["targets"] for data in batch]),
        "scaffolds": [data["scaffold"] for data in batch],
    }

##############################################################################
# 데이터 로드, 피클 저장, scaffold 분할 적용, 시드 고정 추가
##############################################################################
def load_and_save_qm9_dataset(
    directory_path,
    batch_size=32,
    test_ratio=0.1,
    val_ratio=0.1,
    pkl_path="T_qm9_data_with_edges_scaffold.pkl",
    seed=42
):
    set_seed(seed)
    filename, nodes, one_hot_nodes, coords, adjs, edge_indices, edge_attrs, targets, scaffolds = read_xyz_directory(directory_path)
    data = {
        "filename": filename,
        "nodes": nodes,
        "one_hot_nodes": one_hot_nodes,
        "coords": coords,
        "adjs": adjs,
        "edge_indices": edge_indices,
        "edge_attrs": edge_attrs,
        "targets": targets,
        "scaffolds": scaffolds,
    }
    
    # Save processed data as a pickle file
    save_qm9_data_to_pickle(data, pkl_path)
    
    # Scaffold splitting을 사용하여 데이터 분할
    train_idx, val_idx, test_idx = scaffold_split(data, test_ratio=test_ratio, val_ratio=val_ratio, seed=seed)
    
    train_data = {key: [data[key][i] for i in train_idx] for key in data}
    val_data = {key: [data[key][i] for i in val_idx] for key in data}
    test_data = {key: [data[key][i] for i in test_idx] for key in data}
    
    train_dataset = QM9Dataset(train_data)
    val_dataset = QM9Dataset(val_data)
    test_dataset = QM9Dataset(test_data)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    return train_loader, val_loader, test_loader

# 사용 예시
if __name__ == "__main__":
    directory_path = "/home/bori9691/2025/dataset/qm9/raw"
    train_loader, val_loader, test_loader = load_and_save_qm9_dataset(directory_path, seed=42)
