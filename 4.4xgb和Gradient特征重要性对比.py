import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
from sklearn.model_selection import KFold
import os
warnings.filterwarnings('ignore')

# ===================== 1.路径配置 =====================
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "data")          # 数据文件夹
model_dir = os.path.join(base_dir, "model")       # 模型保存文件夹
pic_dir = os.path.join(base_dir, "pic")           # 图片保存文件夹

CLEANED_DATA_PATH = os.path.join(data_dir, "4.1数据集_剔除异常后.xlsx")

# 全局随机种子
RANDOM_STATE = 42

# ===================== 2. 数据加载与预处理 =====================
def load_and_preprocess_data(data_path):
    """加载数据，返回清洗后的特征X和目标y"""
    df = pd.read_excel(data_path)
    target = "AVG Nanofiber diameter [nm]"
    meta_cols = ["Material", "URL(or doi)"]

    # 剔除目标列和索引列
    X = df.drop(columns=[target] + meta_cols, errors='ignore')
    y = df[target]

    # 保留数值列（包括布尔列转换）
    bool_cols = X.select_dtypes(include=['bool']).columns
    if len(bool_cols) > 0:
        X[bool_cols] = X[bool_cols].astype(int)
    X = X.select_dtypes(include=[np.number])

    # 统一列名格式（移除方括号和尖括号）
    X.columns = X.columns.str.replace(r'[\[\]<]', '__', regex=True)

    return X, y


# ===================== 3. 模型评估 =====================
def evaluate_model(model, X_test, y_test):
    """
    现在作为 CV 循环中的原子评估单元：
    评估模型并打印指标
    """
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"📊 Fold 评估结果: R² = {r2:.4f}, MAE = {mae:.2f} nm")
    return r2, mae, y_pred


def plot_dual_feature_importance(model1_data, model2_data, feature_names, model_names, save_path, top_k=15):
    """
    绘制特征重要性对比图
    """
    # --- 自适应字体设置 ---
    font_base = 16  # 基础字号
    plt.rcParams.update({
        'font.sans-serif': ['SimHei', 'Arial', 'DejaVu Sans'],
        'axes.unicode_minus': False,
        'font.size': font_base,
        'axes.labelsize': font_base + 2,
        'axes.titlesize': font_base + 4,
        'xtick.labelsize': font_base,
        'ytick.labelsize': font_base
    })
    sns.set_style("whitegrid")

    # 1. 还原列名的辅助函数
    def restore_name(name):
        """精确还原特征名，目标格式：Name [Unit]"""
        if "Temperature" in name: return "Temperature [deg C]"
        if '__' in name:
            parts = name.split('__')
            base_name = parts[0].replace('_', ' ')
            unit = parts[1].rstrip('_')
            return f"{base_name} [{unit}]"
        return name

    clean_names = [restore_name(n) for n in feature_names]

    # 2. 提取重要性数据
    imp1 = model1_data if isinstance(model1_data, np.ndarray) else model1_data.feature_importances_
    imp2 = model2_data if isinstance(model2_data, np.ndarray) else model2_data.feature_importances_

    # 3. 定义配色
    color_palette = ['#D3D3D3', '#8A9BA8']

    # 4. 创建画布
    fig, axes = plt.subplots(1, 2, figsize=(24, 11))
    data_list = [
        (imp1, model_names[0], axes[0], color_palette[0]),
        (imp2, model_names[1], axes[1], color_palette[1])
    ]

    for imp, name, ax, bar_color in data_list:
        indices = np.argsort(imp)[-top_k:][::-1]
        top_names = [clean_names[i] for i in indices]
        top_imps = imp[indices]

        # 绘图
        ax.barh(range(top_k), top_imps, color=bar_color, edgecolor='#333333', alpha=0.9, height=0.6)

        # 隐藏上、右边框
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(False) # 清除网格

        # 轴设置
        ax.set_yticks(range(top_k))
        ax.set_yticklabels(top_names)
        ax.set_title(f'Top {top_k} Features (Mean Importance: {name})')
        ax.set_xlabel('Importance Score')
        ax.invert_yaxis()

        # 数值标注 (字体随 font_base 调整)
        max_v = max(top_imps) if len(top_imps) > 0 else 1
        for i, v in enumerate(top_imps):
            ax.text(v + (max_v * 0.01), i, f'{v:.4f}', va='center', fontsize=font_base-2, fontweight='bold', color='#333333')

    # 使用 pad 增加间距以实现自适应
    plt.tight_layout(pad=3.0)
    plt.savefig(save_path, dpi=800, bbox_inches='tight')
    plt.close()
    print(f"✅ 双模型对比图已保存: {save_path}")

def main():
    # 0. 初始化路径与模型
    X, y = load_and_preprocess_data(CLEANED_DATA_PATH)

    model_configs = [
        {"name": "RandomForest", "path": os.path.join(model_dir, "RandomForest_best_model.pkl")},
        {"name": "XGBoost", "path": os.path.join(model_dir, "XGBoost_best_model.pkl")},
    ]
    # 用字典来存储每个模型的重要性
    importance_results = {}

    for config in model_configs:
        print(f"\n{'=' * 20} 正在评估: {config['name']} {'=' * 20}")
        model = joblib.load(config['path'])
        kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

        cv_r2, cv_mae, importances_list = [], [], []

        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            model.fit(X_train, y_train)
            r2, mae, y_pred = evaluate_model(model, X_test, y_test)

            cv_r2.append(r2)
            cv_mae.append(mae)
            importances_list.append(model.feature_importances_)

        # 打印指标
        print(f"📊 {config['name']} CV 结果: R² = {np.mean(cv_r2):.4f}, MAE = {np.mean(cv_mae):.2f} nm")

        # 将聚合结果存入字典
        importance_results[config['name']] = np.mean(importances_list, axis=0)
        df = pd.DataFrame(importance_results, index=X.columns)
        save_path = os.path.join(data_dir, "4.4importance_results.xlsx")
        df.to_excel(save_path, index_label='Feature')

    # 确保字典里包含了两个模型
    if len(importance_results) == 2:
        plot_dual_feature_importance(
            model1_data=importance_results["RandomForest"],
            model2_data=importance_results["XGBoost"],
            feature_names=X.columns.tolist(),
            model_names=["RandomForest", "XGBoost"],
            save_path=os.path.join(pic_dir, "图5 RandomForest与XGBoost特征增益贡献的前15位参数对比")
        )

if __name__ == "__main__":
    main()
'''
==================== 正在评估: RandomForest ====================
📊 Fold 评估结果: R² = 0.9282, MAE = 247.76 nm
📊 Fold 评估结果: R² = 0.9233, MAE = 204.60 nm
📊 Fold 评估结果: R² = 0.8955, MAE = 262.85 nm
📊 Fold 评估结果: R² = 0.8788, MAE = 281.75 nm
📊 Fold 评估结果: R² = 0.9549, MAE = 215.26 nm
📊 RandomForest CV 结果: R² = 0.9161, MAE = 242.44 nm

==================== 正在评估: XGBoost ====================
📊 Fold 评估结果: R² = 0.9230, MAE = 245.99 nm
📊 Fold 评估结果: R² = 0.9228, MAE = 201.75 nm
📊 Fold 评估结果: R² = 0.8860, MAE = 263.82 nm
📊 Fold 评估结果: R² = 0.8686, MAE = 289.55 nm
📊 Fold 评估结果: R² = 0.9609, MAE = 213.81 nm
📊 XGBoost CV 结果: R² = 0.9122, MAE = 242.98 nm
✅ 双模型对比图已保存: xxx
'''