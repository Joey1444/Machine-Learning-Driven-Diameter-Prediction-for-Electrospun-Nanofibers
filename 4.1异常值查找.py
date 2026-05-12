import pandas as pd
import numpy as np
import os
from sklearn.ensemble import GradientBoostingRegressor

# ===================== 配置 =====================
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "data")          # 数据文件夹
model_dir = os.path.join(base_dir, "model")       # 模型保存文件夹
pic_dir = os.path.join(base_dir, "pic")           # 图片保存文件夹

DATA_PATH = os.path.join(data_dir, "3用于训练的数据集_未还原.xlsx")

def clean_and_split_data():
    # 1. 加载数据
    df = pd.read_excel(DATA_PATH)
    target = "AVG Nanofiber diameter [nm]"

    # 2. 区分数值特征与元数据列
    # 明确定义哪些列是用于计算的，哪些是元数据
    meta_cols = ['Material', 'URL(or doi)']

    # 获取用于建模的数值列（剔除目标列和元数据列）
    feature_cols = [c for c in df.columns if c != target and c not in meta_cols]

    X = df[feature_cols].copy()
    y = df[target]

    # 清洗列名（确保与模型训练时一致，如将 'Concentration [w/w%]' 变为 'Concentration _w/w%_'）
    X.columns = X.columns.str.replace(r'[\[\]<]', '_', regex=True)

    # 3. 识别异常值
    # 注意：此处使用的 init_model 仅用于计算残差，不参与生产
    init_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    init_model.fit(X, y)
    preds = init_model.predict(X)

    residuals = np.abs(y - preds)
    upper_threshold = residuals.mean() + 3 * residuals.std()
    lower_threshold = residuals.mean() - 3 * residuals.std()
    mask = (residuals >= lower_threshold) & (residuals <= upper_threshold)

    # 4. 分离数据（因为此时 df 包含了所有列，mask 过滤行后会自动保留元数据）
    df_clean = df[mask].copy()
    df_outliers = df[~mask].copy()

    # 5. 保存文件
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    clean_path = os.path.join(data_dir, "4.1数据集_剔除异常后.xlsx")
    outlier_path = os.path.join(data_dir, "4.0数据集_异常值.xlsx")

    df_clean.to_excel(clean_path, index=False)
    df_outliers.to_excel(outlier_path, index=False)

    print(f"✅ 数据处理完成！")
    print(f"  - 原始样本数: {len(df)}")
    print(f"  - 保留正常样本: {len(df_clean)}")
    print(f"  - 剔除异常样本: {len(df_outliers)}")
    print(f"  - 已保留元数据列: {meta_cols}")

    return df_clean


if __name__ == "__main__":
    clean_and_split_data()

'''
✅ 数据处理完成！
  - 原始样本数: 1667
  - 保留正常样本: 1640
  - 剔除异常样本: 27
  - 已保留元数据列: ['Material', 'URL(or doi)']
'''

