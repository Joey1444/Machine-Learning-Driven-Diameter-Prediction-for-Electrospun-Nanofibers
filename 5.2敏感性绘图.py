import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
# ==========================================
# 1. 配置路径
# ==========================================
# 基础目录
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "data")          # 数据文件夹
model_dir = os.path.join(base_dir, "model")       # 模型保存文件夹
pic_dir = os.path.join(base_dir, "pic")           # 图片保存文件夹

path_xgb = os.path.join(data_dir, "4.5LOFO_Result_XGBoost_CV3.xlsx")
path_rf = os.path.join(data_dir, "4.5LOFO_Result_RandomForest_CV3.xlsx")

output_dir = pic_dir  # 输出图目录统一使用项目 pic 文件夹
try:
    # ==========================================
    # 2. 读取与预处理数据
    # ==========================================
    df_xgb = pd.read_excel(path_xgb)
    df_rf = pd.read_excel(path_rf)
    # 确保都有 Model 列，用于分组绘图
    df_xgb['Model'] = 'XGBoost'
    df_rf['Model'] = 'RandomForest'
    # 3. 合并数据为长格式 (Long Format)
    final_report = pd.concat([df_xgb, df_rf], ignore_index=True)
    # 4. 筛选 Top 12 核心特征 (按两模型均值排序)
    mean_imp = final_report.groupby("Feature")["Delta_MAE"].mean().sort_values(ascending=False)
    top_features = mean_imp.head(12).index
    plot_data = final_report[final_report["Feature"].isin(top_features)]
    plot_data.to_excel(os.path.join(data_dir, "4.6top12_Core_Features_LOFO.xlsx"), index=False)

    # ==========================================
    # 5. 可视化绘图配置 (莫兰迪极简学术版)
    # ==========================================
    # 5.1 全局字体与自适应配置 (必须在创建子图前设置)
    plt.rcParams.update({
        'font.size': 16,  # 全局默认字体大小
        'axes.titlesize': 22,  # 标题字体 (加大)
        'axes.labelsize': 18,  # 坐标轴标签字体
        'xtick.labelsize': 15,  # X轴刻度字体
        'ytick.labelsize': 15,  # Y轴刻度字体 (确保特征名清晰)
        'legend.fontsize': 15,  # 图例字体
        'font.family': 'sans-serif',  # 屏幕阅读友好的无衬线字体
        'figure.dpi': 100  # 屏幕显示时提高清晰度
    })
    # 5.2 创建画布与子图
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.set_style("white")  # 确保无背景网格
    # 莫兰迪配色库
    morandi_palette = {
        "XGBoost": "#8CA3B3",  # 蓝灰
        "RandomForest": "#B89F9D"  # 粉灰
    }
    # 5.3 绘制分组条形图
    current_order = plot_data.groupby("Feature")["Delta_MAE"].mean().sort_values(ascending=False).index

    barplot = sns.barplot(
        data=plot_data,
        y="Feature",
        x="Delta_MAE",
        hue="Model",
        palette=morandi_palette,
        orient='h',
        edgecolor='w',
        alpha=0.9,
        ax=ax,
        order=current_order  # <--- 使用针对当前 plot_data 的排序列表
    )
    # ==========================================
    # 6. 极简风格修饰与标签优化
    # ==========================================
    # 6.1 极简风格：仅保留左、下边框 (Frame)
    sns.despine(left=False, bottom=False, right=True, top=True)
    # 6.2 学术化标签与标题 (已更新为精准描述)
    ax.set_title("Marginal Contribution of Features via LOFO", fontweight='bold', pad=30)
    ax.set_xlabel("ΔMAE after feature removal (nm)", labelpad=15)
    ax.set_ylabel("Features", labelpad=15)
    # 6.3 调整图例位置，避免遮挡
    ax.legend(title='Predictive Model', loc='lower right')  # 默认在轴内部右下
    # 6.4 调整布局，防止长特征名溢出
    plt.tight_layout()
    # ==========================================
    # 7. 保存与显示
    # ==========================================
    plt.savefig(os.path.join(pic_dir, '图6 LOFO敏感性分析：前12个特征的ΔMAE及模型对比.png'), dpi=800,
                bbox_inches='tight')
    plt.show()
    print("✅ 数据已成功合并，莫兰迪风格极简图表已生成。")
    print(f"分析的核心特征数: {len(top_features)}")
except FileNotFoundError as e:
    print(f"❌ 文件未找到，请检查路径是否正确: {e}")
except Exception as e:
    print(f"❌ 运行出错: {e}")