# %% 
import pickle
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw

# 저장된 pickle 파일 로드
with open("/home/bori9691/2025/Edge_Moire/data/qm9/T_qm9_data_with_edges_scaffold.pkl", "rb") as f:
    data = pickle.load(f)

# 1. 기본 통계 및 샘플 출력
keys = ["filename", "nodes", "coords", "adjs", "edge_indices", "edge_attrs", "targets", "scaffolds"]
lengths = {key: len(data[key]) for key in keys}
print("각 항목의 길이:")
for key, l in lengths.items():
    print(f"{key}: {l}")

# 모든 리스트의 길이가 동일한지 확인
if len(set(lengths.values())) == 1:
    print("모든 항목의 길이가 동일합니다.")
else:
    print("항목 간 길이가 일치하지 않습니다.")

# 첫 번째 샘플 출력 (예: filename, scaffold, target, 노드와 좌표의 일부)
print("\n첫 번째 샘플 예시:")
print("Filename:", data["filename"][0])
print("Scaffold:", data["scaffolds"][0])
print("Target (gap):", data["targets"][0])
print("Atoms:", data["nodes"][0])
print("Coordinates:\n", data["coords"][0])
print("Adjacency Matrix shape:", data["adjs"][0].shape)

# 3. 시각화 - 첫 번째 분자의 좌표를 산점도로 플롯하고 원자 기호 라벨 표시
def visualize_molecule(coords, atoms, scaffold):
    coords = np.array(coords)
    plt.figure(figsize=(6, 6))
    plt.scatter(coords[:, 0], coords[:, 1], s=100, c='skyblue', edgecolors='k')
    for i, atom in enumerate(atoms):
        plt.text(coords[i, 0], coords[i, 1], atom, fontsize=12, ha='center', va='center')
    plt.title(f"Molecule Visualization\nScaffold: {scaffold}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis('equal')
    plt.show()

# 시각화 예시: 첫 번째 분자
visualize_molecule(data["coords"][0], data["nodes"][0], data["scaffolds"][0])

# 만약 SMILES 정보가 있다면, RDKit을 이용해 분자 구조 이미지를 생성할 수도 있습니다.
# 여기서는 scaffold splitting을 위해 사용된 scaffold 문자열이 있으므로, 이를 활용할 수 있습니다.
if data["scaffolds"][0] != "":
    try:
        mol = Chem.MolFromSmiles(data["scaffolds"][0])
        if mol:
            img = Draw.MolToImage(mol, size=(300,300))
            img.show()
        else:
            print("SMILES로부터 Molecule 생성에 실패했습니다.")
    except Exception as e:
        print("RDKit 오류:", e)
else:
    print("첫 번째 샘플에 SMILES 정보가 없습니다.")

# 4. 데이터 분할 비율 확인
# scaffold_split 함수를 사용하여 데이터 분할 (이미 scaffold 리스트가 data에 포함되어 있음)
def scaffold_split(data, test_ratio=0.1, val_ratio=0.1, seed=42):
    import random
    random.seed(seed)
    num_samples = len(data["filename"])
    scaffold_to_indices = {}
    for i, scaffold in enumerate(data["scaffolds"]):
        if scaffold == "":
            scaffold = "None"
        scaffold_to_indices.setdefault(scaffold, []).append(i)
    # scaffold 그룹을 크기 순으로 내림차순 정렬
    scaffold_groups = sorted(scaffold_to_indices.values(), key=lambda x: len(x), reverse=True)
    
    test_cutoff = int(num_samples * test_ratio)
    val_cutoff = int(num_samples * val_ratio)
    
    test_indices, val_indices, train_indices = [], [], []
    for group in scaffold_groups:
        if len(test_indices) + len(group) <= test_cutoff:
            test_indices.extend(group)
        elif len(val_indices) + len(group) <= val_cutoff:
            val_indices.extend(group)
        else:
            train_indices.extend(group)
    return train_indices, val_indices, test_indices

train_idx, val_idx, test_idx = scaffold_split(data, test_ratio=0.1, val_ratio=0.1, seed=42)
total_samples = len(data["filename"])
print("\n데이터 분할 결과:")
print(f"Train set: {len(train_idx)} samples ({len(train_idx)/total_samples*100:.1f}%)")
print(f"Validation set: {len(val_idx)} samples ({len(val_idx)/total_samples*100:.1f}%)")
print(f"Test set: {len(test_idx)} samples ({len(test_idx)/total_samples*100:.1f}%)")

# %%
