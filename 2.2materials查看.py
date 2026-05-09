'''
python控制台出图更好看
'''

import pandas as pd
import os

# ===================== 配置 =====================
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "data")          # 数据文件夹
model_dir = os.path.join(base_dir, "model")       # 模型保存文件夹
pic_dir = os.path.join(base_dir, "pic")           # 图片保存文件夹

file_path = os.path.join(data_dir, "2静电纺丝数据集.xlsx")

df = pd.read_excel(file_path)
material_data = df["Material"]  # 提取Material列

# ===================== 统计材料种类与数量 =====================
material_count = material_data.value_counts()
features = material_count.index.tolist()
values = material_count.values

print("="*50)
print("数据集材料种类及样本数量统计")
print("="*50)
print(material_count)

# ===================== 创建1行2列子图 =====================
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# ===================== 字体设置（适中偏大） =====================
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 200
plt.rcParams['savefig.dpi'] = 500
# 字体大小：标题12-13，轴标签11，刻度9，饼图文字10
plt.rcParams['font.size'] = 10            # 全局基准
plt.rcParams['axes.labelsize'] = 11       # 坐标轴标签
plt.rcParams['axes.titlesize'] = 13       # 子图标题
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 10
# ===================== 创建子图 =====================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
# ---------------------- 子图1：水平柱状图 ----------------------
features_num = len(features)
y = np.arange(features_num) * 1.3
ax1.barh(y, values, color='#8A9BA8', alpha=0.85, height=0.7)
ax1.set_xlabel('Sample count', fontsize=11, weight='bold')
ax1.set_ylabel('Material', fontsize=11, weight='bold')
ax1.set_title('Sample distribution of electrospinning materials', fontsize=13, weight='bold')
ax1.set_yticks(y)
ax1.set_yticklabels(features, fontsize=9)
ax1.grid(axis='x', alpha=0.2, linestyle='--')
ax1.spines[['top', 'right']].set_visible(False)
ax1.set_xlim(0, 200)
ax1.set_ylim(-0.8, y[-1] + 0.8)
for i, v in enumerate(values):
    show_x = min(v, 200) + 3
    ax1.text(show_x, y[i], f'{v}', va='center', fontsize=9)
# ---------------------- 子图2：饼图 ----------------------
threshold = 40
main_materials = material_count[material_count >= threshold]
other_sum = material_count[material_count < threshold].sum()
if other_sum > 0:
    pie_data = pd.concat([main_materials, pd.Series([other_sum], index=['Others'])])
else:
    pie_data = main_materials
colors = [
    '#6B7280', '#A8B6CC', '#E0A866', '#94A89A', '#C87979', '#8A9BA8',
    '#B0C4DE', '#D4A5A5', '#7D84B2', '#B8CAA9', '#E2C6B6', '#A9B4C2',
    '#C17E7E', '#9AB7D3', '#D3C5A5', '#889FB3', '#BBBE64', '#B59090'
]
# 饼图文字大小：标签和百分比都设为10
wedges, texts, autotexts = ax2.pie(
    pie_data.values,
    labels=pie_data.index,
    autopct='%.1f%%',
    startangle=90,
    colors=colors,
    textprops={'fontsize': 8}
)
ax2.set_title('Proportion of material samples', fontsize=13, weight='bold')
plt.tight_layout()
plt.savefig(os.path.join(pic_dir, '图2 静电纺丝材料分布图.png'), dpi=800, bbox_inches='tight')

plt.show()