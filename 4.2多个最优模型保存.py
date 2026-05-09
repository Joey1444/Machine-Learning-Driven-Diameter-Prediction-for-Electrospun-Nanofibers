import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# ===================== 1. 配置路径 =====================
# 基础目录
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "data")          # 数据文件夹
SAVE_DIR = os.path.join(base_dir, "model")       # 模型保存文件夹
# 数据集路径（全局变量）
DATA_PATH = os.path.join(data_dir, "4.1数据集_剔除异常后.xlsx")

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)


def load_data(path):
    df = pd.read_excel(path)
    target_col = "AVG Nanofiber diameter [nm]"

    # 1. 先分离目标
    y = df[target_col]

    # 2. 剔除目标列和不需要的文本列
    X = df.drop(columns=[target_col, "Material", "URL(or doi)"], errors='ignore')

    # 3. 核心修复：强制只保留数值列 (int, float)
    # 这会剔除所有 object(文本) 或其他不支持的类型
    X = X.select_dtypes(include=[np.number])

    # # 4. 处理布尔列 (如果存在，转换为 0/1)
    # bool_cols = X.select_dtypes(include=['bool']).columns
    # if len(bool_cols) > 0:
    #     X[bool_cols] = X[bool_cols].astype(int)

    X.columns = X.columns.str.replace(r'[\[\]<]', '_', regex=True)

    print(f"数据加载完成。当前特征数量: {X.shape[1]}")
    return X, y


# ===================== 2. 定义搜索空间 =====================
search_spaces = {
    "XGBoost": {
        "model": XGBRegressor(n_jobs=-1, random_state=42, objective='reg:squarederror'),
        "params": {
            'n_estimators': [300, 500, 800],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7, 9],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'reg_alpha': [0, 0.1, 0.5, 1],  # L1 正则化
            'reg_lambda': [1, 1.5, 2]       # L2 正则化
        }
    },
    "RandomForest": {
        "model": RandomForestRegressor(n_jobs=-1, random_state=42),
        "params": {
            "n_estimators": [300, 500, 800, 1000],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ['sqrt', 'log2', 1.0] # 允许使用所有特征
        }
    },
    "GradientBoosting": {
        "model": GradientBoostingRegressor(random_state=42),
        "params": {
            "n_estimators": [300, 500, 800],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [3, 5, 7],
            "subsample": [0.7, 0.8, 0.9, 1.0],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4]
        }
    },
    "SVR": {
        "model": SVR(),
        "params": {
            "C": [0.1, 1, 10, 50, 100, 200],
            "kernel": ["rbf", "linear", "poly"],
            "degree": [2, 3], # 仅对 poly 核有效
            "gamma": ["scale", "auto", 0.01, 0.1, 1],
            "epsilon": [0.01, 0.05, 0.1, 0.2]
        }
    }
}


# ===================== 3. 执行对比 =====================
def run_comparison(X, y):
    results = []
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for name, config in search_spaces.items():
        print(f">>> 正在训练与搜索: {name} ...")
        grid = GridSearchCV(config["model"], config["params"], cv=kf, scoring='r2', n_jobs=-1)
        grid.fit(X, y)

        # 保存模型
        joblib.dump(grid.best_estimator_, os.path.join(SAVE_DIR, f"{name}_best_model.pkl"))

        results.append({
            "Model": name,
            "Best R2": grid.best_score_,
            "Best Params": grid.best_params_
        })

    return pd.DataFrame(results)


# ===================== 3. 绘图 =====================
def plot_model_comparison(results_df):
    """
    绘制符合学术风格的模型性能对比柱状图
    """
    import matplotlib.font_manager as fm

    # 1. 样式初始化 (保持与你提供的风格一致)
    sns.set_style("whitegrid")
    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['axes.titlesize'] = 18
    plt.rcParams['font.family'] = 'sans-serif'  # 若有 simhei 可指定

    # 2. 画布创建
    plt.figure(figsize=(12, 7))

    # 3. 绘制条形图
    # 使用你指定的 #8A9BA8 灰蓝色系，配合深色边框，视觉更稳重
    bars = sns.barplot(
        x="Model",
        y="Best R2",
        data=results_df.sort_values(by="Best R2", ascending=False),
        color='#8A9BA8',
        edgecolor='#333333',
        linewidth=1.5
    )

    # 4. 坐标轴与标题
    plt.xlabel('Machine Learning Models', fontsize=16, labelpad=15)
    plt.ylabel('Cross-Validated R² Score', fontsize=16, labelpad=15)
    plt.title('Comparison of Model Performance', fontsize=18, fontweight='bold', pad=20)

    # 设置 Y 轴范围以突出模型间性能差异
    min_r2 = results_df["Best R2"].min()
    plt.ylim(min_r2 - 0.05, 1.0)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=13)

    # 5. 在条形图上方添加数值标注 (类似特征重要性的标注逻辑)
    for bar in bars.patches:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.005,
            f'{height:.4f}',
            ha='center', va='bottom', fontsize=13, fontweight='bold', color='#444444'
        )

    plt.tight_layout()
    # 建议保存为高分辨率以便论文使用
    # plt.savefig('Model_Comparison.png', dpi=600, bbox_inches='tight')
    plt.show()


# ===================== 4. 主流程 =====================
if __name__ == "__main__":
    X, y = load_data(DATA_PATH)
    results_df = run_comparison(X, y)

    print("\n=== 最终模型性能对比 ===")
    print(results_df[["Model", "Best R2"]].sort_values(by="Best R2", ascending=False))

    # 调用新的绘图函数
    plot_model_comparison(results_df)

'''
=== 最终模型性能对比 ===
              Model   Best R2
1      RandomForest  0.916132
0           XGBoost  0.912667
2  GradientBoosting  0.909341
3               SVR  0.774316
'''