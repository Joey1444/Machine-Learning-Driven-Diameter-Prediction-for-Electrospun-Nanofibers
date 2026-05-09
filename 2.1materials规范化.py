import pandas as pd
import os

# ===================== 配置 =====================
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "data")          # 数据文件夹

file_path = os.path.join(data_dir, "1静电纺丝数据集.xlsx")
save_path = os.path.join(data_dir, "2静电纺丝数据集.xlsx")

df = pd.read_excel(file_path)

# 材料名称清洗函数
def clean_material_name(name):
    name = str(name).strip()
    # 去除括号及内容
    if '(' in name:
        name = name.split('(')[0].strip()

    # 全称→缩写映射
    replace_map = {
        "Silk fibroin": "Silk",
        "Cellulose acetate": "CA",
        "Nylon": "PA6",
        "Nylon 6": "PA6",
        "Nylon 6.6": "PA66",
        "PP-CL": "PPCL",
        "Aromatic PI": "PI",
        "γ-PGA": "PGA",
        "Collagene": "Collagen",
        "Gelatine": "Gelatin",
        "Hylon VII": "Hylon"
    }

    for raw, clean_name in replace_map.items():
        if raw in name:
            return clean_name
    return name

# 替换材料名称
df["Material"] = df["Material"].apply(clean_material_name)

# 保存处理后的文件
df.to_excel(save_path, index=False)
print("✅ 材料名称替换完成！文件已保存为：2静电纺丝数据集.xlsx")
'''
CA
Silk
PVA
PEO
PCL
Chitosan
PA6
PVP
Hylon
PLA
PAN
PSf
Gelatin
HA
PET
Collagen
PI
PVC
PPCL
PVAc
EC
PAM14
PDLLA
PLGA
PMMA
PNIPAAm
PS
PVDF
PGA
共计29种材料
'''