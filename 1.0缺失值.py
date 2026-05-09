"""
静电纺丝数据集 - 自动化数据清洗与缺失值可视化分析

核心功能：
  1. 读取原始Excel数据集，自动去除重复记录
  2. 仅删除纤维直径目标列缺失的样本，保留特征列缺失数据
  3. 自动解析温度/湿度范围字符串（如 20-25 → 22.5），转换为数值中值
  4. 计算并对比数据清洗前后各特征的缺失率
  5. 生成水平柱状图可视化缺失率变化
  6. 导出清洗完成的标准数据集到Excel文件

使用说明：
  - 将原始数据文件放入 ./data/ 文件夹
  - 直接运行脚本，自动完成清洗、绘图、导出
  - 输出文件保存在 ./data/ 目录下，无索引

依赖库：pandas, matplotlib, numpy, openpyxl
"""
#
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# ===================== 配置 =====================
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "data")          # 数据文件夹
model_dir = os.path.join(base_dir, "model")       # 模型保存文件夹
pic_dir = os.path.join(base_dir, "pic")           # 图片保存文件夹

file_path = os.path.join(data_dir, "0静电纺丝数据集_未清洗.xlsx")
save_path = os.path.join(data_dir, "1静电纺丝数据集.xlsx")


target_cols = [
    "Flow rate [ml/h]",
    "Tip distance [cm]",
    "Voltage [kV]",
    "Concentration [w/w%]",
    "Temperature [℃]",
    "Humidity [RH%]",
    "AVG Nanofiber diameter [nm]",
    "CV [%]"
]

# ===================== 数据清洗 =====================
# 1. 读取数据并去重
df_raw_full = pd.read_excel(file_path)
original_len = len(df_raw_full)
df_raw_full = df_raw_full.drop_duplicates().reset_index(drop=True)
new_len = len(df_raw_full)

print(f">>> 去重完成：删除了 {original_len - new_len} 条重复数据，剩余 {new_len} 条。")
# >>> 去重完成：删除了 17 条重复数据，剩余 1977 条。

# 2. 去除缺失值：仅除去目标列"AVG Nanofiber diameter [nm]"缺失的行
df_clean_full = df_raw_full.dropna(subset=["AVG Nanofiber diameter [nm]"])
print(f">>> 删除了 {len(df_raw_full) - len(df_clean_full)} 行缺失目标值的数据")
# >>> 删除了 310 行缺失目标值的数据

# 3. 处理"Temperature [℃]", "Humidity [RH%]"的范围数据：取范围的中值
def parse_range_to_median(value):
    # 1. 显式跳过：如果是空值（NaN/None），直接返回原始空值
    if pd.isna(value) or str(value).strip().lower() == 'nan':
        return np.nan

    # 2. 如果本身就是数值类型（int/float），直接返回
    if isinstance(value, (int, float)):
        return value

    val_str = str(value).strip()

    # 3. 处理范围字符串
    for sep in ['-', '–', '~', '—']:
        if sep in val_str:
            try:
                # 过滤掉拆分后可能出现的空字符串
                parts = [float(x.strip()) for x in val_str.split(sep) if x.strip()]
                if len(parts) == 2:
                    return sum(parts) / 2
            except ValueError:
                # 如果包含非数字（如 "20-室温"），则返回 NaN 或原值
                return np.nan

    # 4. 尝试直接转换纯数字字符串
    try:
        return float(val_str)
    except ValueError:
        return np.nan

# --- 执行处理 ---
cols_to_fix = ["Temperature [℃]", "Humidity [RH%]"]

for col in cols_to_fix:
    if col in df_clean_full.columns:
        df_clean_full[col] = df_clean_full[col].apply(parse_range_to_median)

        # errors='coerce' 会把无法解析的残余字符串强制转为 NaN，原本就是 NaN 的位置会保持不变。
        df_clean_full[col] = pd.to_numeric(df_clean_full[col], errors='coerce')

print(">>> 范围数据处理完成。空值已跳过并保持为 NaN。")

# 4. 从完整数据中提取目标列，用于后续绘图、统计
df_raw = df_raw_full[target_cols]
df_clean = df_clean_full[target_cols]

# 导出：完整列 + 已删除缺失值，无行索引
df_clean_full.to_excel(save_path, index=False, engine="openpyxl")

# ===================== 缺失率计算（原有逻辑不变） =====================
miss_raw = (df_raw.isnull().sum() / len(df_raw)) * 100
miss_clean = (df_clean.isnull().sum() / len(df_clean)) * 100

# ===================== 按【去除后】缺失率降序排列 =====================
sort_idx = miss_clean.sort_values(ascending=False).index
miss_clean = miss_clean[sort_idx]
miss_raw = miss_raw[sort_idx]
features = sort_idx.tolist()

# ===================== 英文绘图 =====================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 200

fig, ax = plt.subplots(figsize=(7, 3.5))
y = np.arange(len(features))
w = 0.35

# 【关键修改】：创建一个替换列表，专门用于绘图显示
display_features = [f.replace("Temperature [℃]", "Temperature [deg C]") for f in features]

ax.barh(y - w/2, miss_raw, w, color='#5B6F99', alpha=0.85, label='Before removal')
ax.barh(y + w/2, miss_clean, w, color='#94A89A', alpha=0.85, label='After removal')

max_val = max(miss_raw.max(), miss_clean.max())
ax.set_xlim(0, max_val * 1.12)

ax.set_xlabel('Missing ratio (%)', fontsize=11, weight='bold')
ax.set_ylabel('Parameter', fontsize=11, weight='bold')

fig.suptitle('Effect of removing missing "AVG Nanofiber diameter [nm]"', fontsize=12, weight='bold', ha='center')

ax.set_yticks(y)
# 【关键修改】：使用替换后的 display_features 列表
ax.set_yticklabels(display_features, fontsize=9)
ax.legend(fontsize=10, loc='upper right')

ax.grid(axis='x', alpha=0.2, linestyle='--')
ax.spines[['top', 'right']].set_visible(False)

for i, (r, c) in enumerate(zip(miss_raw, miss_clean)):
    ax.text(r + max_val*0.015, i - w/2, f'{r:.1f}%', va='center', fontsize=8)
    ax.text(c + max_val*0.015, i + w/2, f'{c:.1f}%', va='center', fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(pic_dir, '图1 移除AVG Nanofiber diameter [nm]前后缺失比例.png'), dpi=800, bbox_inches='tight')
plt.show()

# ===================== 打印对比表 =====================
print('='*65)
print(f'原始样本：{len(df_raw)} | 清洗后：{len(df_clean)} | 删除：{len(df_raw)-len(df_clean)}')
print('='*65)
print(f"{'Feature':<33} {'Before(%)':<10} {'After(%)'}")
print('-'*60)
for f in features:
    print(f"{f:<33} {miss_raw[f]:<10.1f} {miss_clean[f]:.1f}")

# 保存成功提示
print(f"\n✅ 表格处理成功！")
print(f"📁 文件路径：{save_path}")

'''
原始样本：1977 | 清洗后：1667 | 删除：310
=================================================================
Feature                           Before(%)  After(%)
------------------------------------------------------------
Humidity [RH%]                    52.0       56.9
CV [%]                            50.1       40.8
Temperature [℃]                   28.9       31.9
Concentration [w/w%]              25.9       29.4
Flow rate [ml/h]                  7.1        7.6
Tip distance [cm]                 5.3        6.0
Voltage [kV]                      0.6        0.5
AVG Nanofiber diameter [nm]       15.7       0.0

✅ 表格处理成功！
📁 文件路径：xxx
'''