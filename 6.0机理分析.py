import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, Draw
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from rdkit.Chem.Draw import rdMolDraw2D
from PIL import Image
import io
import os

# ===================== 1. 全局配置 (使用 Pathlib) =====================
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "data")          # 数据文件夹
model_dir = os.path.join(base_dir, "model")       # 模型保存文件夹
pic_dir = os.path.join(base_dir, "pic")           # 图片保存文件夹

DATA_PATH = Path(data_dir)
SAVE_DIR = Path(model_dir)
PIC_DIR = Path(pic_dir)
CLEANED_DATA_PATH = DATA_PATH / "4.1数据集_剔除异常后.xlsx"
PCA_PATH = SAVE_DIR / "pca_model.pkl"
SUBSTRUCT_DIR = PIC_DIR / "Key_Substructures"

# 确保输出目录存在
SAVE_DIR.mkdir(parents=True, exist_ok=True)
SUBSTRUCT_DIR.mkdir(parents=True, exist_ok=True)

# SMILES 字典
MATERIAL_SMILES = {
    "PVA": "CC(O)", "PEO": "OCCO", "PCL": "OCCCCCC(=O)", "PA6": "NCCCCCC(=O)",
    "PVP": "CC1CCCN1C(=O)", "PLA": "OC(C)C(=O)", "PAN": "CC(C#N)", "PMMA": "CC(C)(C(=O)OC)",
    "PVC": "CC(Cl)", "PVDF": "CC(F)(F)", "PS": "CC(c1ccccc1)", "PET": "OCC(=O)c1ccc(C(=O))cc1",
    "PGA": "OCC(=O)", "PLGA": "OC(C)C(=O)OCC(=O)", "PDLLA": "OC(C)C(=O)",
    "PNIPAAm": "CC(C(=O)NC(C)C)", "PVAc": "CC(OC(=O)C)", "PA66": "NCCCCCCNC(=O)CCCCC(=O)",
    "PPCL": "CC(C)Cl", "PAM14": "CC(C(=O)NCCCCCCCCCCCCCC)",
    "PSf": "CC(C)(c1ccc(cc1)Oc2ccc(cc2)S(=O)(=O)c3ccc(cc3)O)c4ccc(cc4)O",
    "PI": "O=C1C2=CC=CC=C2C(=O)N1c3ccccc3", "CA": "CC(=O)OCC1OC(OC(C)=O)C(OC(C)=O)C(O)C1O",
    "Silk": "NCC(=O)NC(C)C(=O)", "Chitosan": "OCC1OC(N)C(O)C(O)C1O", "Hylon": "OCC1OC(O)C(O)C(O)C1O",
    "Gelatin": "NCC(=O)NC(C)C(=O)", "Collagen": "NCC(=O)NC(C)C(=O)",
    "HA": "CC(=O)NC1C(O)OC(CO)C(O)C1OC2C(O)C(O)C(O)C(C(=O)O)O2", "EC": "CCOCC1OC(OCC)C(OCC)C(O)C1O",
}

target_pc_names = ['PC_6', 'PC_11', 'PC_12', 'PC_14']
target_pcs = [6, 11, 12, 14]

# ===================== 2. 核心分析模块 =====================

def get_molecular_descriptors(smiles):
    """提取分子物理化学描述符"""
    mol = Chem.MolFromSmiles(smiles)
    if not mol: return [np.nan] * 7
    return [
        Descriptors.MolWt(mol),  # 分子量
        Descriptors.MolLogP(mol),  # 脂水分配系数
        Descriptors.NumHDonors(mol),  # 氢键供体
        Descriptors.NumHAcceptors(mol),  # 氢键受体
        Descriptors.NumRotatableBonds(mol),  # 可旋转键
        Descriptors.TPSA(mol),  # 拓扑极性表面积
        Descriptors.HeavyAtomCount(mol)  # 重原子数
    ]


def visualize_substructures(pca, pc_idx, mol_library, save_path, top_n=15, radius=2):
    """
    还原并可视化特定主成分(PC)的核心化学结构片段
    """
    # 1. 获取该 PC 的载荷权重
    pc_loadings = pca.components_[pc_idx]
    # 2. 找出影响权重最大的前 top_n 个指纹位点索引
    top_indices = np.argsort(np.abs(pc_loadings))[-top_n:][::-1]

    results = []
    print(f"\n>>> 正在解析 PC{pc_idx + 1} 的 Top-{top_n} 结构特征...")

    for rank, bit_idx in enumerate(top_indices, 1):
        bit_idx = int(bit_idx)
        loading_val = pc_loadings[bit_idx]

        # 遍历分子库寻找该位点由哪个分子贡献
        for mat_name, mol in mol_library.items():
            if mol is None: continue

            bit_info = {}
            _ = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=2048, bitInfo=bit_info)

            if bit_idx in bit_info:
                # 3. 提取该位点关联的所有原子索引 (修正了之前的迭代错误)
                atom_indices = [item[0] for item in bit_info[bit_idx]]

                # 4. 使用 Cairo 接口绘制并渲染
                d = rdMolDraw2D.MolDraw2DCairo(300, 300)
                # 确保 highlightAtoms 接收的是列表
                rdMolDraw2D.PrepareAndDrawMolecule(d, mol, highlightAtoms=atom_indices)
                d.FinishDrawing()

                # 5. 转为 PIL 图片对象并保存
                img = Image.open(io.BytesIO(d.GetDrawingText()))
                img_name = f"PC{pc_idx + 1}_Rank{rank}_Bit{bit_idx}.png"
                img.save(save_path / img_name)

                results.append({
                    "PC": f"PC{pc_idx + 1}",
                    "Rank": rank,
                    "Bit": bit_idx,
                    "Loading": loading_val,
                    "Material": mat_name
                })
                break  # 找到第一个触发该特征的分子即可，避免冗余

    return pd.DataFrame(results)


# ===================== 3. 主程序逻辑 =====================

def run_mechanism_analysis(target_pc_names, target_pcs):
    print(f"{' Starting AI4S Mechanism Analysis ':=^50}")

    # 1. 数据加载与对齐
    df = pd.read_excel(CLEANED_DATA_PATH)
    pca = joblib.load(PCA_PATH)

    # 定义目标列与元数据
    target = "AVG Nanofiber diameter [nm]"
    meta_cols = ['Material', 'URL(or doi)']

    # 2. 计算描述符并填补缺失值
    print(">>> 正在计算分子描述符...")
    desc_names = ['MolWt', 'LogP', 'HDonors', 'HAcceptors', 'RotBonds', 'TPSA', 'HeavyAtoms']
    raw_descs = df['Material'].map(lambda x: get_molecular_descriptors(MATERIAL_SMILES.get(x, '')))
    desc_df = pd.DataFrame(raw_descs.tolist(), columns=desc_names, index=df.index)

    # === 新增：缺失值检查逻辑 ===
    total_missing = desc_df.isnull().sum().sum()
    if total_missing == 0:
        print("✅ 描述符计算完成，未检测到缺失值。")
        desc_cleaned = desc_df.values  # 无需填补，直接使用
    else:
        print(f"⚠️ 检测到 {total_missing} 个缺失值，正在使用中位数进行填补...")
        imputer = SimpleImputer(strategy='median')
        desc_cleaned = imputer.fit_transform(desc_df)
    # ===========================

    # 3. PC 与 描述符相关性分析
    print(">>> 正在进行 PC-物理描述符相关性分析...")
    # target_pc_names = target_pc_names
    available_pcs = [col for col in target_pc_names if col in df.columns]

    # 计算相关性矩阵
    analysis_df = pd.concat([df[available_pcs], pd.DataFrame(desc_cleaned, columns=desc_names, index=df.index)], axis=1)
    corr_matrix = analysis_df.corr().loc[available_pcs, desc_names]

    # === 新增：保存原始相关性数据 ===
    corr_data_path = DATA_PATH / "5Key_PC_Descriptor_Correlation_Data.xlsx"
    corr_matrix.T.to_excel(corr_data_path)
    print(f"✅ 相关性原始数据已保存至: {corr_data_path}")

    # ===============================
    # 绘制学术级热图
    # 动态计算自适应的画布大小 (宽: PC数*2, 高: 描述符数*1.5)
    # 1. 调整画布比例 (让它更方正，即使是 4x7)
    fig_width, fig_height = 8, 6  # 给定一个合理的固定比例
    plt.figure(figsize=(fig_width, fig_height), dpi=300)

    # 2. 增强热图视觉
    ax = sns.heatmap(
        corr_matrix,
        annot=True,
        cmap='RdBu_r',
        center=0,
        fmt=".2f",
        linewidths=2,
        linecolor='white',
        square=True,
        cbar_kws={"shrink": 0.6, "label": "Pearson Correlation", "pad": 0.1},  # pad 调整间距
        annot_kws={"size": 10, "weight": "bold"}
    )

    # 2. 调整 cbar 字体大小 (这是最有效的办法)
    cbar = ax.collections[0].colorbar
    cbar.set_label("Pearson Correlation", fontsize=10)  # 标签字体小一些
    cbar.ax.tick_params(labelsize=9)  # 刻度数字字体小一些

    # 3. 字体美化与旋转
    plt.xticks(rotation=45, ha='right', fontsize=11, )
    plt.yticks(rotation=0, fontsize=11, )
    plt.title("Correlation: Key PCs vs Descriptors", fontsize=13, pad=20,fontweight='bold')

    plt.tight_layout()
    plt.savefig(PIC_DIR / "图7 关键主成分与分子物理化学描述符的相关性分析热图.png", dpi=800, bbox_inches='tight')

    # 4. 批量执行关键 PC 的结构溯源
    print(">>> 正在批量进行结构溯源分析...")
    mol_lib = {k: Chem.MolFromSmiles(v) for k, v in MATERIAL_SMILES.items() if v}

    # 定义你要溯源的 PC 序号 (对应 Python 索引为 序号-1)
    # target_pcs = [6, 11, 12, 14]

    for pc_num in target_pcs:
        print(f"--- 正在处理 PC_{pc_num} ---")
        pc_idx = pc_num - 1  # 索引转换

        # 提取并保存
        report = visualize_substructures(pca, pc_idx=pc_idx, mol_library=mol_lib, save_path=SUBSTRUCT_DIR)
        report.to_excel(DATA_PATH / f"5 PC_{pc_num}_Substructure_Analysis.xlsx", index=False)
        print(f"✅ PC_{pc_num} 结构溯源完成，图片已存至: {SUBSTRUCT_DIR}")

    print(f"{' 批量分析任务全部完成 ':=^50}")


if __name__ == "__main__":
    run_mechanism_analysis(target_pc_names, target_pcs)