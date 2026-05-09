'''
'''
# ===================== 模块导入 =====================
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.model_selection import GridSearchCV
from rdkit import Chem
from rdkit.Chem import AllChem
import joblib
from xgboost.callback import EarlyStopping
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer
from functools import lru_cache

warnings.filterwarnings('ignore')

# ===================== 全局配置 =====================
TUNE_MODE = False  # XGBoost的网格搜索 | True: 运行网格搜索(耗时) | False: 使用预设最优参数(快速)
DEFAULT_FP_RADIUS = 2  # 分子指纹半径
DEFAULT_FP_NBITS = 2048  # 分子指纹长度（位数）
DEFAULT_PCA_VARIANCE = 0.95  # PCA主成分分析累积方差贡献率阈值
DEFAULT_TEST_SIZE = 0.2  # 测试集比例
DEFAULT_RANDOM_STATE = 42  # 随机种子

# ===================== 配置 =====================
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "data")          # 数据文件夹
model_dir = os.path.join(base_dir, "model")       # 模型保存文件夹
pic_dir = os.path.join(base_dir, "pic")           # 图片保存文件夹

# 数据路径
INPUT_FILE_PATH = os.path.join(data_dir, "2静电纺丝数据集.xlsx")  # 规范化的数据集输入
SAVE_PATH_UNRESTORED = os.path.join(data_dir, "3用于训练的数据集_未还原.xlsx")
SAVE_PATH_RESTORED = os.path.join(data_dir, "3用于训练的数据集_还原.xlsx")

# 模型路径
SCALER_SAVE_PATH = os.path.join(model_dir, "scaler_proc.pkl")
PCA_SAVE_PATH = os.path.join(model_dir, "pca_model.pkl")

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


# ===================== 1. 数据加载=====================
def load_and_clean_data(file_path: str, target_cols=None):
    """加载数据的目标列"""
    if target_cols is None:
        target_cols = [
            'URL(or doi)', "Material", "Flow rate [ml/h]", "Tip distance [cm]", "Voltage [kV]",
            "Concentration [w/w%]", "Temperature [℃]", "Humidity [RH%]",
            "CV [%]", "AVG Nanofiber diameter [nm]"
        ]
    df = pd.read_excel(file_path, usecols=target_cols)
    return df

# ===================== 2. 分子指纹生成 =====================
@lru_cache(maxsize=128)
def _get_single_fingerprint(material_name: str, radius: int, nbits: int) -> np.ndarray:
    """
    _get_single_fingerprint：基于材料名称生成单个材料的Morgan分子指纹
    核心功能：将材料名称通过SMILES字符串转换为标准化分子特征向量，用于机器学习模型输入
    """
    smiles = MATERIAL_SMILES.get(material_name)
    if smiles is None:
        print(f"警告：材料 '{material_name}' 不在 SMILES 字典中，返回零向量")
        return np.zeros(nbits)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"警告：材料 '{material_name}' 的 SMILES 解析失败: {smiles}")
        return np.zeros(nbits)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nbits)
    return np.array(fp)


def generate_fingerprint_matrix(df: pd.DataFrame, material_col: str = "Material",
                                radius: int = DEFAULT_FP_RADIUS, nbits: int = DEFAULT_FP_NBITS) -> np.ndarray:
    """
    为 DataFrame 中的材料列生成 Morgan 分子指纹矩阵
    遍历所有材料名称，批量生成指纹并拼接为模型可直接输入的二维特征矩阵
    """
    fingerprints = [_get_single_fingerprint(name, radius, nbits) for name in df[material_col]]
    return np.vstack(fingerprints)


# ===================== 3. 缺失值处理+特征工程+pca降维 =====================
def preprocess_electrospinning_data(train_df, test_df):
    """
    数据预处理：含数据缩放、中位数填充、分类化及 KNN 填充
    """
    low_miss_cols = ["Flow rate [ml/h]", "Tip distance [cm]", "Voltage [kV]"]
    knn_cols = ["Concentration [w/w%]", "Temperature [℃]"]
    level_labels = ['Low', 'Medium', 'High']

    train_df = train_df.copy()
    test_df = test_df.copy()

    # ===================== 步骤1：中位数填充 (低缺失) =====================
    for col in low_miss_cols:
        median_val = train_df[col].median()
        train_df[col] = train_df[col].fillna(median_val)
        test_df[col] = test_df[col].fillna(median_val)

    # ===================== 步骤2：数据缩放 (Scaling) =====================
    scaler = StandardScaler()
    scale_features = knn_cols + low_miss_cols  # 对knn_cols+knn_cols列进行缩放

    train_df[scale_features] = scaler.fit_transform(train_df[scale_features])
    test_df[scale_features] = scaler.transform(test_df[scale_features])

    # ===================== 步骤3：KNN 填充 =====================
    knn_imputer = KNNImputer(n_neighbors=10)
    train_df[knn_cols] = knn_imputer.fit_transform(train_df[knn_cols])
    test_df[knn_cols] = knn_imputer.transform(test_df[knn_cols])

    # ===================== 步骤4：还原数据 (Inverse Transform) =====================
    # 填充完成后，将数据还原回原始量纲，方便后续的分箱处理和结果解读
    train_df[scale_features] = scaler.inverse_transform(train_df[scale_features])
    test_df[scale_features] = scaler.inverse_transform(test_df[scale_features])

    # ===================== 步骤5：湿度与 CV 分类化 =====================
    def get_categorical_dummies(train_source, target_df, col_name, prefix):
        """
        根据 train_source 的分位数，对 target_df 进行分箱并直接返回哑变量 DataFrame
        删除了参照类别（Low），避免完美共线性，仅返回 Medium, High, Missing 三列
        """
        valid_vals = train_source[col_name].dropna()

        # 定义四个目标类别名，但最终只返回三个，删除 Low
        all_categories = [f"{prefix}_Low", f"{prefix}_Medium", f"{prefix}_High", f"{prefix}_Missing"]

        if valid_vals.empty:
            # 如果全为空，直接返回全是 0 的 DataFrame，但 Missing 列为 1
            dummy_df = pd.DataFrame(0, index=target_df.index, columns=all_categories)
            dummy_df[f"{prefix}_Missing"] = 1
            dummy_df = dummy_df.drop(columns=[f"{prefix}_Low"])  # 删除 Low 列
            return dummy_df

        # 1. 计算分位数边界
        q1, q2 = np.percentile(valid_vals, [33.3, 66.6])
        bins = [-np.inf, q1, q2, np.inf]
        labels = ['Low', 'Medium', 'High']

        # 2. 生成分类序列
        # 将空值填充为 'Missing'
        cuts = pd.cut(target_df[col_name], bins=bins, labels=labels)
        cat_series = cuts.astype(object).fillna('Missing')

        # 3. 转换为哑变量
        # columns 参数确保生成的列顺序一致，且涵盖所有四个类别
        dummy_df = pd.get_dummies(cat_series, prefix=prefix, prefix_sep='_')

        # 4. 健壮性检查：确保四个列都存在（即使某次划分中某个级别样本为0）
        for col in all_categories:
            if col not in dummy_df.columns:
                dummy_df[col] = 0

        # 5. 删除参照类别（Low），消除完美共线性
        dummy_df = dummy_df.drop(columns=[f"{prefix}_Low"])

        # 6. 按顺序排列（Medium, High, Missing）
        optimized_categories = [f"{prefix}_Medium", f"{prefix}_High", f"{prefix}_Missing"]
        return dummy_df[optimized_categories]

    # ===================== 执行函数 =====================
    # 处理湿度
    hum_dummies_train = get_categorical_dummies(train_df, train_df, 'Humidity [RH%]', 'Humidity')
    hum_dummies_test = get_categorical_dummies(train_df, test_df, 'Humidity [RH%]', 'Humidity')
    train_df = pd.concat([train_df, hum_dummies_train], axis=1)
    test_df = pd.concat([test_df, hum_dummies_test], axis=1)

    # 处理 CV
    cv_dummies_train = get_categorical_dummies(train_df, train_df, 'CV [%]', 'CV')
    cv_dummies_test = get_categorical_dummies(train_df, test_df, 'CV [%]', 'CV')
    train_df = pd.concat([train_df, cv_dummies_train], axis=1)
    test_df = pd.concat([test_df, cv_dummies_test], axis=1)

    return train_df, test_df  # 确保返回的是 concat 后的新 DataFrame


def build_feature_vectors(train_df, test_df, process_features,
                          fp_radius=DEFAULT_FP_RADIUS, fp_nbits=DEFAULT_FP_NBITS,
                          pca_variance=DEFAULT_PCA_VARIANCE, random_state=DEFAULT_RANDOM_STATE):
    """构建特征：工艺参数 + PCA 降维后的分子指纹"""
    # 工艺参数矩阵
    X_train_proc = train_df[process_features].values
    X_test_proc = test_df[process_features].values
    # 分子指纹
    train_fps = generate_fingerprint_matrix(train_df, radius=fp_radius, nbits=fp_nbits)
    test_fps = generate_fingerprint_matrix(test_df, radius=fp_radius, nbits=fp_nbits)
    # PCA 降维（仅对指纹）
    pca = PCA(n_components=pca_variance, random_state=random_state)
    train_fps_pca = pca.fit_transform(train_fps)
    test_fps_pca = pca.transform(test_fps)
    print(f"PCA 降维：原始维度 {fp_nbits} -> {train_fps_pca.shape[1]} 维（保留 {pca_variance * 100:.0f}% 方差）")
    print(f"前5个主成分解释方差比：{pca.explained_variance_ratio_[:5]}")
    # 标准化工艺参数
    scaler = StandardScaler()
    X_train_proc_scaled = scaler.fit_transform(X_train_proc)
    X_test_proc_scaled = scaler.transform(X_test_proc)
    # 拼接
    X_train = np.hstack([X_train_proc_scaled, train_fps_pca])
    X_test = np.hstack([X_test_proc_scaled, test_fps_pca])
    y_train = train_df["AVG Nanofiber diameter [nm]"].values
    y_test = test_df["AVG Nanofiber diameter [nm]"].values
    # 返回必要对象用于后续绘图
    return X_train, X_test, y_train, y_test, pca, train_fps_pca, scaler

# ===================== 4. 模型训练与评估 =====================
def train_xgboost_model(X_train, y_train, X_test, y_test, tune_params=TUNE_MODE):
    if tune_params:
        print(">>> [状态] 正在进行网格搜索寻优（Grid Search），请稍候...")
        # 1. 定义基础模型
        base_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            random_state=DEFAULT_RANDOM_STATE,
            n_jobs=-1
        )

        # 2. 定义搜索空间（包含你之前测试过的所有维度）
        param_grid = {
            'n_estimators': [100, 300, 500],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],
            'subsample': [0.7, 0.9],
            'colsample_bytree': [0.7, 0.9],
            'reg_alpha': [0, 0.5, 1],
            'reg_lambda': [1, 2]
        }

        # 3. 执行网格搜索 (5折交叉验证)
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            scoring='r2',
            cv=5,
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(X_train, y_train)

        # 提取搜索到的最优模型
        model = grid_search.best_estimator_
        print(f"✅ [结果] 网格搜索完成！")
        print(f"✅ [参数] 最优组合: {grid_search.best_params_}")
        print(f"✅ [得分] 最佳训练集 CV R²: {grid_search.best_score_:.4f}")

    else:
        print(">>> [状态] 使用预设最优参数进行稳定训练...")
        # 4. 直接使用之前网格搜索得到的“黄金参数”
        # 这些参数是基于你之前的运行结果确定的，能保证测试集 R² 达到约 0.76
        best_stable_params = {
            'n_estimators': 300,
            'learning_rate': 0.05,
            'max_depth': 7,
            'subsample': 0.9,
            'colsample_bytree': 0.7,
            'reg_alpha': 0.5,
            'reg_lambda': 2,
            'objective': 'reg:squarederror',
            'random_state': DEFAULT_RANDOM_STATE,
            'n_jobs': -1
        }
        model = xgb.XGBRegressor(**best_stable_params)
        # 替换 model.fit 这一行
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )

    # 5. 最终预测与评估
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print("\n" + "=" * 60)
    print("📊 最终模型评估结果")
    print("-" * 60)
    print(f"🔹 测试集 R²   : {r2:.4f}")
    print(f"🔹 测试集 MAE   : {mae:.2f} nm")
    print(f"🔹 测试集 RMSE  : {rmse:.2f} nm")
    print("=" * 60)

    # 6. 计算模型稳定性（交叉验证均值）
    kfold = KFold(n_splits=5, shuffle=True, random_state=DEFAULT_RANDOM_STATE)
    cv_scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring='r2')
    print(f"🔹 模型 Cross-Validation R² 稳定性: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

    # 7. 提取特征重要性
    importance = model.feature_importances_

    return model, y_pred, r2, mae, rmse, importance, cv_scores


def plot_pca_scree(pca, train_fps_pca, pca_variance_ratio, save_path='PCA_scree_plot.png'):
    """绘制 PCA 碎石图（学术出版风格）"""
    explained_var = pca.explained_variance_ratio_
    cumsum_var = np.cumsum(explained_var)
    n_components = train_fps_pca.shape[1]

    # 1. 全局样式设置
    plt.rcParams.update({
        'font.sans-serif': ['Arial'],
        'axes.unicode_minus': False,
        'font.size': 14,
        'axes.labelsize': 16,
        'axes.titlesize': 18,
        'legend.fontsize': 13
    })
    # 使用 'white' 风格，手动控制网格线，避免 sns 自动生成实线网格
    sns.set_style("white")

    fig, ax1 = plt.subplots(figsize=(14, 8))

    # 配色方案
    bar_color = '#6C91B2'
    line_color = '#B87C6C'
    threshold_color = '#8B9A7B'
    vline_color = '#D0A37E'

    # 2. 绘制柱状图 (Individual variance)
    ax1.bar(range(1, n_components + 1), explained_var, alpha=0.75, color=bar_color,
            label='Individual explained variance', zorder=3)
    ax1.set_xlabel('Principal Component', fontsize=16, labelpad=12)
    ax1.set_ylabel('Explained Variance Ratio', fontsize=16, labelpad=12)

    # 3. 设置右侧坐标轴
    ax2 = ax1.twinx()
    # 【关键修改】：仅在 ax2 上添加虚线网格，确保 zorder=0 位于最底层
    ax2.grid(True, linestyle='--', color='#D3D3D3', alpha=0.6, zorder=0)

    # 4. 绘制折线图 (Cumulative variance)
    ax2.plot(range(1, n_components + 1), cumsum_var, color=line_color, linestyle='-',
             linewidth=3, marker='o', markersize=7, label='Cumulative explained variance', zorder=4)
    ax2.set_ylabel('Cumulative Variance Ratio', fontsize=16, labelpad=12)

    # 5. 辅助线与标注
    ax2.axhline(y=pca_variance_ratio, color=threshold_color, linestyle='--', linewidth=2, label='95% threshold')
    idx_95 = np.argmax(cumsum_var >= pca_variance_ratio) + 1
    ax2.axvline(x=idx_95, color=vline_color, linestyle=':', linewidth=2.5, label=f'{idx_95} components reach 95%')

    # 修改后的标注代码
    ax2.annotate(f'{idx_95} PCs',
                 xy=(idx_95, pca_variance_ratio),  # 箭头指向点
                 xytext=(idx_95 + 1.5, pca_variance_ratio + 0.05),  # 标签位置（距离更近）
                 arrowprops=dict(arrowstyle='->', color=vline_color, lw=1.2),
                 fontsize=12,
                 color=vline_color,
                 va='bottom', ha='left')

    # # 6. 【关键修改】：边框控制（隐藏顶部和右侧，只保留左、下）
    # for ax in [ax1, ax2]:
    #     ax.spines['top'].set_visible(False)
    # ax2.spines['right'].set_visible(True)  # 保留右轴 spine
    # ax1.spines['right'].set_visible(False)  # ax1 右侧隐藏

    # 刻度设置
    step = max(1, n_components // 10)
    ax1.set_xticks(range(1, n_components + 1, step))
    ax1.tick_params(axis='both', labelsize=13)

    # 7. 图例与标题
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    fig.legend(lines1 + lines2, labels1 + labels2, loc='center', bbox_to_anchor=(0.7, 0.65),
               frameon=True, fancybox=True, fontsize=13)

    plt.title('PCA Scree Plot of Morgan Fingerprints', fontsize=18, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(pic_dir, '图3 Morgan分子指纹特征降维的PCA主成分碎石图及其累计方差解释率.png'), dpi=800,
                bbox_inches='tight')
    plt.show()
#
# def plot_pred_vs_true(y_test, y_pred, r2, mae, save_path='Pred_vs_True_scatter.png'):
#     """绘制预测值与真实值散点图"""
#     plt.figure(figsize=(8, 8))
#     sns.scatterplot(x=y_test, y=y_pred, alpha=0.7, edgecolor='k', s=80)
#     min_val = min(y_test.min(), y_pred.min())
#     max_val = max(y_test.max(), y_pred.max())
#     plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Ideal (y=x)')
#     plt.fill_between([min_val, max_val], [min_val * 0.85, max_val * 0.85],
#                      [min_val * 1.15, max_val * 1.15], color='gray', alpha=0.2, label='±15% error band')
#     plt.xlabel('True Fiber Diameter (nm)', fontsize=12)
#     plt.ylabel('Predicted Fiber Diameter (nm)', fontsize=12)
#     plt.title(f'Predicted vs True Fiber Diameter\nR² = {r2:.3f}, MAE = {mae:.1f} nm', fontsize=13)
#     plt.legend()
#     plt.grid(True, linestyle='--', alpha=0.6)
#     plt.axis('equal')
#     plt.tight_layout()
#     plt.savefig(save_path, dpi=300, bbox_inches='tight')
#     plt.show()
#
#
# def plot_feature_importance(importance, feature_names, top_k=15):
#     """
#     绘制特征重要性条形图（将 Temperature [℃] 替换为 Temperature）
#     """
#     import matplotlib.font_manager as fm
#     import os
#
#     # 1. 样式初始化
#     font_path = 'C:/Windows/Fonts/simhei.ttf'
#     if os.path.exists(font_path):
#         prop = fm.FontProperties(fname=font_path)
#     else:
#         prop = fm.FontProperties(family='sans-serif')
#
#     sns.set_style("whitegrid")
#     plt.rcParams['font.size'] = 14
#     plt.rcParams['axes.labelsize'] = 16
#     plt.rcParams['axes.titlesize'] = 18
#
#     # --- 核心修改：仅替换特征名称中的 Temperature 字符串 ---
#     # 遍历列表，将包含指定名称的项简化
#     clean_feature_names = [
#         name.replace("Temperature [℃]", "Temperature") for name in feature_names
#     ]
#     # --------------------------------------------------
#
#     # 2. 提取数据（使用清洗后的名称）
#     indices = np.argsort(importance)[-top_k:][::-1]
#     top_features = [clean_feature_names[i] for i in indices]
#     top_importances = importance[indices]
#
#     # 3. 创建画布
#     plt.figure(figsize=(14, 8))
#
#     # 4. 绘制条形图
#     bars = plt.barh(range(top_k), top_importances, color='#8A9BA8',
#                     edgecolor='#333333', alpha=0.85, linewidth=1.5)
#
#     # 5. 坐标轴细化
#     # 注意：此处 yticks 使用清洗后的名称
#     plt.yticks(range(top_k), top_features, fontsize=14)
#     plt.xticks(fontsize=13)
#     plt.xlabel('Feature Importance Score', fontsize=16, labelpad=12)
#     plt.title(f'Top-{top_k} Feature Importances (XGBoost)', fontsize=18, fontweight='bold', pad=20)
#     plt.gca().invert_yaxis()
#
#     # 6. 数值标注
#     max_val = max(top_importances) if len(top_importances) > 0 else 1
#     for i, imp in enumerate(top_importances):
#         plt.text(imp + (max_val * 0.01), i, f'{imp:.4f}',
#                  va='center', fontsize=13, fontweight='bold', color='#444444')
#
#     # 7. 布局与展示
#     plt.tight_layout()
#     # 提示：如果保存 Word 仍需高清，请记得在 plt.show() 前加上 savefig
#     # plt.savefig('Feature_importance_top15.png', dpi=600, bbox_inches='tight')
#     plt.show()


# ===================== 6. 完整流程=====================
def run_full_pipeline(file_path: str):
    """
    完整流程：数据加载 -> 预处理 -> 特征工程 -> 模型训练 -> 结果保存与对象打包
    """
    print(">>> 正在启动数据处理流程...")

    # 1. 加载数据
    df = load_and_clean_data(file_path)

    # 2. 划分数据集
    train_df_raw, test_df_raw = train_test_split(df, test_size=DEFAULT_TEST_SIZE, random_state=DEFAULT_RANDOM_STATE)

    # 提取元数据
    meta_cols = ['Material', 'URL(or doi)']
    train_meta = train_df_raw[meta_cols].reset_index(drop=True)
    test_meta = test_df_raw[meta_cols].reset_index(drop=True)
    meta_all = pd.concat([train_meta, test_meta]).reset_index(drop=True)

    # 3. 预处理
    train_df, test_df = preprocess_electrospinning_data(train_df_raw, test_df_raw)

    # 4. 确定工艺特征列表
    base_process = ["Flow rate [ml/h]", "Tip distance [cm]", "Voltage [kV]",
                    "Concentration [w/w%]", "Temperature [℃]"]
    hum_cols = [c for c in train_df.columns if "Humidity_" in c]
    cv_cols = [c for c in train_df.columns if "CV_" in c]
    final_process_features = base_process + hum_cols + cv_cols

    # 5. 特征构建
    X_train, X_test, y_train, y_test, pca, train_fps_pca, scaler_proc = build_feature_vectors(
        train_df, test_df, final_process_features
    )
    joblib.dump(scaler_proc, SCALER_SAVE_PATH)
    joblib.dump(pca, PCA_SAVE_PATH)

    # 6. 模型训练
    model, y_pred, r2, mae, rmse, importance, cv_scores = train_xgboost_model(
        X_train, y_train, X_test, y_test
    )

    # 7. 特征拼接与元数据还原
    pc_names = [f"PC_{i + 1}" for i in range(train_fps_pca.shape[1])]
    all_feature_names = final_process_features + pc_names

    X_all_scaled = np.vstack([X_train, X_test])
    y_all = np.concatenate([y_train, y_test])

    # A. 构造标准化数据集 (用于模型训练)
    full_dataset_scaled = pd.DataFrame(X_all_scaled, columns=all_feature_names)
    full_dataset_scaled['AVG Nanofiber diameter [nm]'] = y_all
    full_dataset_scaled = pd.concat([full_dataset_scaled, meta_all], axis=1)

    # B. 构造物理原尺度数据集 (用于机理分析)
    X_all_inverse = X_all_scaled.copy()
    num_proc_features = len(final_process_features)
    # 仅对工艺参数列进行逆变换
    X_all_inverse[:, :num_proc_features] = scaler_proc.inverse_transform(X_all_scaled[:, :num_proc_features])

    full_dataset_inverse = pd.DataFrame(X_all_inverse, columns=all_feature_names)
    full_dataset_inverse['AVG Nanofiber diameter [nm]'] = y_all
    full_dataset_inverse = pd.concat([full_dataset_inverse, meta_all], axis=1)

    # 8. 执行保存
    try:
        full_dataset_scaled.to_excel(SAVE_PATH_UNRESTORED, index=False)
        full_dataset_inverse.to_excel(SAVE_PATH_RESTORED, index=False)
        print(">>> 成功：所有处理结果已保存。")
        print(f">>> 标准化数据集: {SAVE_PATH_UNRESTORED}")
        print(f">>> 还原尺度数据集: {SAVE_PATH_RESTORED}")
    except Exception as e:
        print(f">>> 保存失败，请检查路径权限或文件是否打开: {e}")

    # 8. 【关键】打包返回结果，确保后续绘图可以直接通过 results["key"] 访问
    return {
        "df_inverse": full_dataset_inverse,
        "pca": pca,
        "importance": importance,
        "feature_names": all_feature_names,
        "train_fps_pca": train_fps_pca,
        "pca_variance_ratio": DEFAULT_PCA_VARIANCE,
        "y_test": y_test,
        "y_pred": y_pred,
        "metrics": (r2, mae, rmse, cv_scores)
    }


# ===================== 执行入口 =====================
if __name__ == "__main__":
    file_path = INPUT_FILE_PATH

    # 1. 执行全流程
    results = run_full_pipeline(file_path)

    # 2. 【核心修改】从字典中直接提取对象，不要重新调用函数
    pca = results["pca"]
    importance = results["importance"]
    feature_names = results["feature_names"]
    train_fps_pca = results["train_fps_pca"]
    # 手动重建 train_df 以获取化学名称信息
    df_raw = load_and_clean_data(file_path)
    train_df, _ = train_test_split(df_raw, test_size=DEFAULT_TEST_SIZE, random_state=DEFAULT_RANDOM_STATE)
    # 如果你需要 y_test 等
    y_test = results["y_test"]
    y_pred = results["y_pred"]

    # 提取模型性能指标
    r2, mae, rmse, cv_scores = results["metrics"]


    # 图 1：PCA 碎石图（展示分子指纹降维效果）
    plot_pca_scree(
        pca=results["pca"],
        train_fps_pca=results["train_fps_pca"],
        pca_variance_ratio=results["pca_variance_ratio"]
    )

    # # 图 2：预测值 vs 真实值散点图（展示模型精度）
    # plot_pred_vs_true(
    #     y_test=results["y_test"],
    #     y_pred=results["y_pred"],
    #     r2=r2,
    #     mae=mae,
    #     save_path='Pred_vs_True_scatter.png'
    # )
    #
    # # 图 3：特征重要性条形图（展示哪些工艺/结构参数最关键）
    # plot_feature_importance(
    #     importance=results["importance"],
    #     feature_names=results["feature_names"],
    #     top_k=15,
    # )

    print("\n>>> 所有图表已保存至当前运行目录。")
    print(f">>> 最终模型测试集 R²: {r2:.4f}")
'''
>>> 正在启动数据处理流程...
PCA 降维：原始维度 2048 -> 15 维（保留 95% 方差）
前5个主成分解释方差比：[0.19615224 0.17803103 0.12453776 0.09631671 0.0719611 ]
>>> [状态] 使用预设最优参数进行稳定训练...

============================================================
📊 最终模型评估结果
------------------------------------------------------------
🔹 测试集 R²   : 0.7590
🔹 测试集 MAE   : 304.87 nm
🔹 测试集 RMSE  : 925.63 nm
============================================================
🔹 模型 Cross-Validation R² 稳定性: 0.8138 (±0.0582)
>>> 成功：所有处理结果已保存。
>>> 标准化数据集: D:\OneDrive - mail.dhu.edu.cn\#DHU\毕业论文\毕设new\代码1\data\3用于训练的数据集_未还原.xlsx
>>> 还原尺度数据集: D:\OneDrive - mail.dhu.edu.cn\#DHU\毕业论文\毕设new\代码1\data\3用于训练的数据集_还原.xlsx

>>> 所有图表已保存至当前运行目录。
>>> 最终模型测试集 R²: 0.7590
'''
