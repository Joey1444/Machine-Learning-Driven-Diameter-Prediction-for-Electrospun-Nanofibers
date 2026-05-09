import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import warnings
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import logging
from sklearn.model_selection import KFold, cross_val_score
from tqdm import tqdm
import os

warnings.filterwarnings('ignore')

# ===================== 1. 环境配置与列名还原 =====================
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
RANDOM_STATE = 42

base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "data")          # 数据文件夹
model_dir = os.path.join(base_dir, "model")       # 模型保存文件夹
pic_dir = os.path.join(base_dir, "pic")           # 图片保存文件夹

CLEANED_DATA_PATH = os.path.join(data_dir, "4.1数据集_剔除异常后.xlsx")


def restore_name(name):
    """精确还原特征名，目标格式：Name [Unit]"""
    if "Temperature" in name: return "Temperature [deg C]"
    if '__' in name:
        parts = name.split('__')
        base_name = parts[0].replace('_', ' ')
        unit = parts[1].rstrip('_')
        return f"{base_name} [{unit}]"
    return name

# ===================== 2. 共线性诊断 ====================
def check_multicollinearity(X_train):
    """
    检查训练集特征的多重共线性
    - 绘制下三角热图 (莫兰迪配色，大字体)
    - 计算方差膨胀因子 (VIF)
    """
    print("\n" + "=" * 60)
    print("🔍 多重共线性诊断 ")
    print("=" * 60)

    # 1. 准备绘图数据（还原列名）
    X_plot = X_train.rename(columns=restore_name)
    corr = X_plot.corr()

    # 2. 生成莫兰迪(Morandi)配色方案
    # 使用 seaborn 的 Diverging Palette 生成蓝-灰-粉的柔和渐变
    morandi_cmap = sns.diverging_palette(h_neg=240, h_pos=10, s=80, l=45, sep=20, n=256, as_cmap=True)

    # 3. 设置只显示下半部分三角的掩码 (Mask)
    # k=1 表示遮盖对角线及其上方的严格上三角
    mask = np.triu(np.ones_like(corr, dtype=bool), k=0)

    # 4. 创建画布，启用自动调整大小
    fig, ax = plt.subplots(figsize=(16, 13))

    # 5. 绘制热图
    sns.heatmap(
        corr,
        mask=mask,  # 应用掩码，只显下三角
        cmap=morandi_cmap,  # 应用莫兰迪配色
        center=0,  # 0值为中心色（灰色）
        annot=False,  # 不在格子里写数字，保持学术整洁
        square=True,  # 每个格子是正方形
        linewidths=0.8,  # 格子间的线条粗细
        cbar_kws={"shrink": 0.7},  # 侧边色条缩放因子
        ax=ax
    )

    # 6. 字体放大与自动调整设置
    # 大标题
    ax.set_title("Feature Correlation Matrix", fontsize=24, fontweight='bold', pad=30)

    # X轴刻度字体：放大，旋转45度，右对齐
    plt.xticks(rotation=45, ha='right', fontsize=16)

    # Y轴刻度字体：放大
    plt.yticks(fontsize=16)

    # 自动调整布局，防止长特征名被切断
    plt.tight_layout()

    # 7. 保存图片（可选，用于发表）
    # plt.savefig(os.path.join(pic_dir, "图6 LOFO敏感性分析：前12个特征的ΔMAE及模型对比.png"),
    #             dpi=800, bbox_inches='tight')
    plt.show()

    # 8. 找出高相关对 (|r| > 0.8)
    high_corr_pairs = []
    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):
            if abs(corr.iloc[i, j]) > 0.8:
                high_corr_pairs.append((corr.columns[i], corr.columns[j], corr.iloc[i, j]))

    if high_corr_pairs:
        print("\n⚠️ 高度相关的特征对 (|r| > 0.8):")
        for pair in high_corr_pairs:
            print(f"   {pair[0]} ↔ {pair[1]} : r = {pair[2]:.3f}")
    else:
        print("\n✅ 未发现高度相关的特征对 (|r| > 0.8)")

    # 9. 计算 VIF（逻辑不变，还原名字展示）
    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        from statsmodels.tools.tools import add_constant

        X_with_const = add_constant(X_train)
        vif_data = pd.DataFrame()
        # VIF输出同样使用还原后的名字
        vif_data["Feature"] = [restore_name(c) for c in X_with_const.columns]
        vif_data["VIF"] = [variance_inflation_factor(X_with_const.values, i)
                           for i in range(X_with_const.shape[1])]

        vif_data = vif_data[vif_data["Feature"] != "const"].sort_values("VIF", ascending=False)

        print("\n📊 方差膨胀因子 (VIF) 排名 (前10):")
        # 大样本下打印前10，小样本打印全部
        print(vif_data.head(50).to_string(index=False))
        vif_data.to_excel(os.path.join(data_dir, "4.5特征方差膨胀因子.xlsx"), index=False)
    except ImportError:
        print("\n⚠️ 未安装 statsmodels，无法计算 VIF。")
    except Exception as e:
        print(f"\n⚠️ VIF 计算失败: {e}")

    return high_corr_pairs


# ===================== 5 折交叉验证 LOFO 分析 =====================
def run_robust_lofo_analysis(model, X, y, model_name):
    """
    执行 5 折交叉验证下的 LOFO 分析，并自动保存结果
    """
    logging.info(f">>> 正在执行 {model_name} 的 5 折 CV-LOFO 敏感性分析...")

    # 参数清理
    raw_params = model.get_params()
    clean_params = {k: v for k, v in raw_params.items() if
                    k not in ['early_stopping_rounds', 'eval_metric', 'callbacks', 'n_jobs']}
    model_class = xgb.XGBRegressor if "XGB" in str(type(model)) else RandomForestRegressor

    # 1. 计算基准 CV MAE
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    base_scores = cross_val_score(model_class(**clean_params), X, y, cv=kf, scoring='neg_mean_absolute_error')
    base_mae = -base_scores.mean()

    lofo_results = []

    # 2. 逐特征进行 5 折 CV 剔除分析
    for feat in tqdm(X.columns, desc=f"交叉验证分析 {model_name}"):
        X_sub = X.drop(columns=[feat])

        # 重新运行 CV
        sub_scores = cross_val_score(model_class(**clean_params), X_sub, y, cv=kf, scoring='neg_mean_absolute_error')
        cv_mae_sub = -sub_scores.mean()

        delta_mae = cv_mae_sub - base_mae

        lofo_results.append({
            "Feature": restore_name(feat),
            "Model": model_name,
            "Delta_MAE": delta_mae
        })

    # 3. 自动保存结果
    df_results = pd.DataFrame(lofo_results)
    save_path = os.path.join(data_dir, f"4.5LOFO_Result_{model_name}_CV3.xlsx")
    df_results.to_excel(save_path, index=False)
    logging.info(f"✅ {model_name} 分析结果已保存至: {save_path}")

    return df_results


# ===================== 4. 主流程 =====================
def main():
    try:
        DATA_PATH = os.path.join(data_dir, "4.1数据集_剔除异常后.xlsx")

        MODEL_PATHS = {
            "RandomForest": os.path.join(model_dir, "RandomForest_best_model.pkl"),
            "XGBoost": os.path.join(model_dir, "XGBoost_best_model.pkl")
        }

        # 数据加载
        df = pd.read_excel(DATA_PATH)
        target = "AVG Nanofiber diameter [nm]"
        X = df.drop(columns=[target, "Material", "URL(or doi)"], errors='ignore').select_dtypes(include=[np.number])
        X.columns = X.columns.str.replace(r'[\[\]<]', '__', regex=True)
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
        # 🎯 已恢复：在分析前进行共线性诊断
        check_multicollinearity(X_train)

        # 执行 LOFO 循环
        all_reports = []
        for m_name, m_path in MODEL_PATHS.items():
            model_obj = joblib.load(m_path)
            # 调用三折 CV LOFO 分析
            report = run_robust_lofo_analysis(model_obj, X, y, m_name)
            all_reports.append(report)

        # 1. 合并所有模型的结果，形成汇总表
        final_report = pd.concat(all_reports, ignore_index=True)

        # # 2. 保存为必存的 Combined_LOFO_Summary.csv
        # final_report.to_csv("Combined_LOFO_Summary.csv", index=False)
        # logging.info("✅ 汇总表 Combined_LOFO_Summary.csv 已保存。")

        # 3. 绘图逻辑 (直接读取汇总表)
        top_features = final_report.groupby("Feature")["Delta_MAE"].mean().sort_values(ascending=False).head(10).index
        plot_data = final_report[final_report["Feature"].isin(top_features)]

        plt.figure(figsize=(12, 8))
        sns.barplot(data=plot_data, y="Feature", x="Delta_MAE", hue="Model", palette="muted")
        plt.title("LOFO Sensitivity Comparison", fontsize=14, fontweight='bold')
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"运行出错: {e}")


if __name__ == "__main__":
    main()