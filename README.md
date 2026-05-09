# 基于机器学习的静电纺纳米纤维直径预测

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> 一个基于机器学习的静电纺纳米纤维直径预测完整流程，结合材料特性与工艺参数进行建模分析。

## 📋 目录

- [项目概述](#项目概述)
- [功能特性](#功能特性)
- [安装说明](#安装说明)
- [项目结构](#项目结构)
- [使用指南](#使用指南)
- [工作流程](#工作流程)
- [模型对比](#模型对比)
- [实验结果](#实验结果)
- [数据说明](#数据说明)
- [引用格式](#引用格式)

## 🔬 项目概述

本项目实现了一个端到端的机器学习解决方案，用于预测静电纺纳米纤维的平均直径。主要技术特点包括：

- **分子指纹编码**：使用 RDKit 将材料化学结构转换为 Morgan 分子指纹
- **PCA 降维**：将高维分子特征降维至主成分
- **多模型对比**：XGBoost、随机森林、梯度提升和 SVR
- **LOFO 敏感性分析**：评估各特征对模型预测的贡献度
- **机理解析**：关联分子描述符与主成分，揭示影响机制

## ✨ 功能特性

- **数据预处理**：自动处理缺失值、范围解析和异常值检测
- **特征工程**：生成 2048 位 Morgan 指纹并结合 PCA 降维
- **模型训练**：使用 GridSearchCV 进行超参数优化，5 折交叉验证
- **模型评估**：全面的评价指标（R²、MAE）和稳定性分析
- **可视化**：符合学术发表标准的高分辨率图表
- **机理解释**：关键主成分的化学亚结构溯源分析

## 🛠️ 安装说明

### 环境要求

- **Python 3.10 或更高版本**（已在 Python 3.10.11 上测试）
- [RDKit](https://www.rdkit.org/)（用于分子指纹生成）

> ⚠️ **注意**：Python 3.12+ 可能与部分科学计算包存在兼容性问题，建议使用 Python 3.10 或 3.11。

### 安装步骤

1. 克隆仓库：
```bash
git clone https://github.com/yourusername/electrospun-nanofiber-prediction.git
cd electrospun-nanofiber-prediction
```

2. 创建虚拟环境（推荐）：
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

3. 安装依赖：
```bash
pip install -r requirements.txt
```

**注意**：如果通过 pip 安装 RDKit 遇到问题，建议使用 conda：
```bash
conda install -c conda-forge rdkit
```

## 📁 项目结构

```
.
├── data/                          # 数据集目录
│   ├── 0静电纺丝数据集_未清洗.xlsx      # 原始数据
│   ├── 1静电纺丝数据集.xlsx             # 清洗后数据
│   ├── 2静电纺丝数据集.xlsx             # 材料名称规范化后
│   ├── 3用于训练的数据集_*.xlsx          # 训练用数据集
│   ├── 4.0数据集_异常值.xlsx            # 检测到的异常值
│   ├── 4.1数据集_剔除异常后.xlsx         # 剔除异常值后的数据
│   └── ...                        # 分析结果和 LOFO 输出
├── model/                         # 训练好的模型
│   ├── XGBoost_best_model.pkl
│   ├── RandomForest_best_model.pkl
│   ├── GradientBoosting_best_model.pkl
│   ├── SVR_best_model.pkl
│   ├── pca_model.pkl              # PCA 模型
│   └── scaler_proc.pkl            # 特征缩放器
├── pic/                           # 生成的图表
│   ├── Key_Substructures/         # 分子亚结构可视化
│   └── 图*.png                    # 分析图表
├── 1.0缺失值.py                    # 数据清洗与缺失值处理
├── 2.1materials规范化.py           # 材料名称规范化
├── 3.0smiles.py                   # 主流程：指纹生成与 XGBoost 训练
├── 4.1异常值查找.py                # 异常值检测
├── 4.2多个最优模型保存.py           # 多模型训练与网格搜索
├── 4.3四个模型对比.py              # 模型对比与评估
├── 4.4xgb和Gradient特征重要性对比.py  # 特征重要性对比
├── 5.1LOFO特征敏感性.py            # LOFO 敏感性分析
├── 5.2敏感性绘图.py                # 敏感性可视化
├── 6.0机理分析.py                  # 机理解析（PC-描述符相关性）
├── requirements.txt               # Python 依赖
└── README.md                      # 项目说明
```

## 🚀 使用指南

### 完整流程执行

按顺序运行以下脚本：

```bash
# 步骤 1：数据清洗
python "1.0缺失值.py"

# 步骤 2：材料名称规范化
python "2.1materials规范化.py"

# 步骤 3：特征工程与初始模型训练
python "3.0smiles.py"

# 步骤 4：异常值检测与剔除
python "4.1异常值查找.py"

# 步骤 5：训练所有模型（含超参数调优）
python "4.2多个最优模型保存.py"

# 步骤 6：模型评估与对比
python "4.3四个模型对比.py"

# 步骤 7：LOFO 敏感性分析
python "5.1LOFO特征敏感性.py"

# 步骤 8：机理解析
python "6.0机理分析.py"
```

### 单独运行

每个脚本可以独立运行（需确保前置数据文件存在）。具体用法请参考各文件头部的注释说明。

## 🔁 工作流程

```
原始数据 → 数据清洗 → 材料规范化 → 特征工程
                              ↓
机理解析 ← 模型训练 ← 异常值剔除 ← 分子指纹生成
                              ↓
                    模型对比与评估
                              ↓
                    LOFO 敏感性分析
```

## 🤖 模型对比

本项目实现并对比了四种回归算法：

| 模型 | 最佳 CV R² | 说明 |
|------|-----------|------|
| **随机森林** | 0.916 | Bagging 集成的决策树 |
| **XGBoost** | 0.913 | 带正则化的梯度提升 |
| **梯度提升** | 0.909 | 序贯误差修正集成 |
| **SVR** | 0.774 | 径向基核支持向量回归 |

*异常值剔除后的结果（5 折交叉验证）*

## 📊 实验结果

### 主要发现

- **最佳性能**：随机森林达到 R² = 0.916 ± 0.03，MAE = 242.44 nm
- **异常值影响**：剔除异常值后 R² 从约 0.78 提升至约 0.91
- **特征重要性**：工艺参数（流速、浓度）与分子指纹（主成分）最具预测力
- **PCA 降维**：2048 位 Morgan 指纹降至 15 个主成分（保留 95% 方差）

### 输出文件

- **图表**：高分辨率（800 DPI）学术发表标准图片
  - 缺失值分析图
  - PCA 碎石图
  - 模型对比图
  - 特征重要性排序图
  - LOFO 敏感性分析图
  - PC-描述符相关性热图
- **模型**：序列化模型文件（.pkl 格式）
- **报告**：Excel 格式的分析结果

## 📈 数据说明

### 输入特征

| 特征 | 单位 | 说明 |
|------|------|------|
| Material | - | 聚合物材料类型（SMILES 编码） |
| Flow rate | ml/h | 溶液流速 |
| Tip distance | cm | 接收距离 |
| Voltage | kV | 施加电压 |
| Concentration | w/w% | 溶液浓度 |
| Temperature | ℃ | 环境温度 |
| Humidity | RH% | 相对湿度 |
| CV | % | 变异系数 |

### 目标变量

- **AVG Nanofiber diameter [nm]**：静电纺纳米纤维平均直径

### 数据集统计

- **原始样本**：1,977 条
- **清洗后**：1,667 条
- **剔除异常值后**：1,639 条
- **覆盖材料**：25+ 种聚合物类型

## 📚 引用格式

如果在研究中使用本代码或数据集，请引用：

```bibtex
@software{nanofiber_ml_prediction,
  title={Machine Learning-Driven Diameter Prediction for Electrospun Nanofibers},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/electrospun-nanofiber-prediction}
}
```

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

## 🙏 致谢

- [RDKit](https://www.rdkit.org/) - 化学信息学功能
- [scikit-learn](https://scikit-learn.org/) - 机器学习工具
- [XGBoost](https://xgboost.readthedocs.io/) - 梯度提升实现

## 📧 联系方式

如有问题或合作意向，请在 GitHub 上提交 Issue 或联系作者。

---

**声明**：本项目为学术研究用途。模型基于特定实验条件训练，应用于不同实验装置时可能需要重新训练。
