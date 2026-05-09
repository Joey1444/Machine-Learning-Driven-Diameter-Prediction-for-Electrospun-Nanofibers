import pandas as pd
import joblib
import os
import numpy as np
from sklearn.model_selection import KFold, cross_validate
from sklearn.metrics import make_scorer, r2_score, mean_absolute_error

# ===================== 1. 配置 =====================
# 基础目录
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "data")          # 数据文件夹
SAVE_DIR = os.path.join(base_dir, "model")       # 模型保存文件夹
pic_dir = os.path.join(base_dir, "pic")           # 图片保存文件夹

# 异常值去除后
DATA_PATH_AFTER = os.path.join(data_dir, "4.1数据集_剔除异常后.xlsx")
# 异常值去除前
DATA_PATH_BEFORE = os.path.join(data_dir, "3用于训练的数据集_未还原.xlsx")

def load_and_clean_data(path):
    df = pd.read_excel(path)
    target_col = "AVG Nanofiber diameter [nm]"
    y = df[target_col]
    X = df.drop(columns=[target_col, "Material", "URL(or doi)"], errors='ignore')
    X = X.select_dtypes(include=[np.number])
    X.columns = X.columns.str.replace(r'[\[\]<]', '_', regex=True)
    return X, y


def evaluate_with_cv(data_path, dataset_name):
    X, y = load_and_clean_data(data_path)

    # 保持与 GridSearchCV 完全一致的切分逻辑
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # 定义评分指标
    scoring = {
        'r2': 'r2',
        'mae': 'neg_mean_absolute_error'
    }

    model_files = [f for f in os.listdir(SAVE_DIR) if f.endswith(".pkl")]

    print(f"\n===== 数据集：{dataset_name} =====")
    print(f"\n{'模型名称':<20} | {'CV R² (平均)':<12} | {'CV MAE (nm)':<12}")
    print("-" * 55)

    for f in model_files:
        model_name = f.replace("_best model.pkl", "")
        try:
            model = joblib.load(os.path.join(SAVE_DIR, f))

            # 执行 5 折交叉验证
            scores = cross_validate(model, X, y, cv=kf, scoring=scoring)

            mean_r2 = np.mean(scores['test_r2'])
            mean_mae = -np.mean(scores['test_mae'])  # 取负数转回正值

            print(f"{model_name:<20} | {mean_r2:<12.4f} | {mean_mae:<12.2f}")

        except Exception as e:
            print(f"{model_name:<20} | 评估失败: {e}")


if __name__ == "__main__":
    evaluate_with_cv(DATA_PATH_BEFORE, "剔除异常值前")
    evaluate_with_cv(DATA_PATH_AFTER, "剔除异常值后")

'''
1.异常值去除前
模型名称                 | CV R² (平均)   | CV MAE (nm) 
-------------------------------------------------------
GradientBoosting     | 0.7764       | 350.04     
RandomForest         | 0.7829       | 332.54      
SVR                  | 0.6525       | 443.34      
XGBoost              | 0.7781       | 340.91 

2.异常值去除后
模型名称                 | CV R² (平均)   | CV MAE (nm) 
GradientBoosting     | 0.9093       | 249.00      
RandomForest         | 0.9161       | 242.44     
SVR                  | 0.7743       | 366.33       
XGBoost              | 0.9122       | 242.98   
'''


# ===================== B. 绘图 =====================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_academic_minimalist():
    # 1. 数据准备
    # 按 clean_r2 降序排列
    models = ["RandomForest", "XGBoost", "GradientBoosting", "SVR"]

    # 异常值去除前（原始数据）
    raw_r2 = [0.7829, 0.7781, 0.7764, 0.6525]
    raw_mae = [332.54, 340.91, 350.04, 443.34]

    # 异常值去除后（clean数据）
    clean_r2 = [0.9161, 0.9122, 0.9093, 0.7743]
    clean_mae = [242.44, 242.98, 249.00, 366.33]

    delta_mae = [r - c for r, c in zip(raw_mae, clean_mae)]
    # 2. 初始化画布 - 移除所有背景网格
    sns.set_style("white")  # 改为 white 样式，彻底清除背景线
    plt.rcParams.update({'font.size': 13, 'font.family': 'sans-serif'})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    # 3. 左图：R2 对比
    df_r2 = pd.DataFrame({
        'Model': models * 2,
        'Value': raw_r2 + clean_r2,
        'Dataset': ['Before'] * 4 + ['After'] * 4
    })
    bars1 = sns.barplot(x="Model", y="Value", hue="Dataset", data=df_r2,
                        palette=['#D3D3D3', '#8A9BA8'], edgecolor='#333333',
                        ax=ax1, width=0.5)
    ax1.set_title('Improvement in Prediction Accuracy (R²)', fontsize=16, fontweight='bold')
    ax1.set_ylim(0.5, 1.0)
    ax1.set_ylabel('R² Score', fontsize=14)
    ax1.set_xlabel('', fontsize=14)
    # 隐藏上方和右侧的边框线 (Spines)，提升高级感
    sns.despine()
    for container in bars1.containers:
        ax1.bar_label(container, fmt='%.4f', padding=3, fontsize=10, fontweight='bold')
    # 4. 右图：Delta MAE (斜线填充，无背景)
    bars2 = ax2.bar(models, delta_mae, color='white', edgecolor='#333333',
                    width=0.5, hatch='//', linewidth=1.5)
    ax2.set_title('Error Reduction (Δ MAE)', fontsize=16, fontweight='bold')
    ax2.set_ylabel('Reduced Error (nm)', fontsize=14)
    ax2.set_ylim(0, 120)
    sns.despine()  # 同样隐藏右图的边框线
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height + 2,
                 f'{height:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(pic_dir, '图4异常值剔除对模型预测精度的影响.png'), dpi=800,
                bbox_inches='tight')
    plt.show()
if __name__ == "__main__":
    plot_academic_minimalist()