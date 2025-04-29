🏥 患者生存预测系统
===========

本项目旨在使用机器学习模型预测 ICU 患者的住院死亡风险。它提供了从数据预处理、相关性分析、模型训练、集成学习、可视化到用户友好 GUI 的全流程，用于实时临床预测。

* * *

🎯 项目目标
-------

*   构建集成模型以预测住院死亡风险
    
*   从患者记录和 ICU 监测中提取关键特征
    
*   为临床场景设计可用的预测界面
    
*   提供结构化结果和洞察以支持决策
    

* * *

📁 项目结构
-------

```
Patient–Survival–Prediction/
│
├── dataSet/
│   └── dataset.csv                # 原始患者级别数据（CSV）
│
├── Model/                         # 已保存模型
│   ├── base_models.pkl            # 训练好的基础模型（XGB、DNN、LR）
│   └── meta_model.pkl             # 训练好的堆叠元模型
│
├── report/
│   └── logs/
│       ├── categorical_correlation.log
│       └── numerical_correlation.log
│
├── src/
│   ├── common/
│   │   └── variable_array.py      # 预定义的分类/数值特征列表
│
│   ├── config/
│   │   └── path_config.py         # 日志、数据和模型路径管理
│
│   ├── correlation_analysis/
│   │   ├── categorical_analysis.py   # 分类特征相关性（如 Cramer’s V）
│   │   └── numerical_analysis.py     # 数值特征相关性（如 Pearson）
│
│   ├── model/
│   │   ├── DNN.py                  # 基于 Keras 的深度神经网络
│   │   ├── Ensemble.py             # 堆叠 & 软投票集成逻辑
│   │   ├── LogisticRegression.py   # scikit-learn 逻辑回归
│   │   └── Xgboost.py              # XGBoost 树模型
│
│   ├── notebook/
│   │   └── EDA.ipynb               # 探索性数据分析笔记本
│
│   └── visualization/
│       ├── DNN/
│       ├── Ensemble/
│       ├── LogisticRegression/
│       ├── XgBoost/
│       └── plot_utils.py           # 绘图工具函数
│
├── patient_risk_interface.py      # 实时死亡风险预测 GUI
│
├── col_description.md             # 完整数据字典（中文）
├── requirements.txt               # Python 依赖列表
├── README.md                      # 项目文档（本文件）
├── .gitignore                     # Git 忽略配置
└── .venv/                         # 本地 Python 虚拟环境
```

* * *

📊 数据集描述
--------

该数据集来源于 Kaggle ICU 死亡率预测数据集。完整特征说明见 `col_description.md`。关键数据类型：

*   **人口统计**：年龄、性别、种族、身高、体重、BMI
    
*   **ICU 信息**：入院来源、ICU 类型、入住类型
    
*   **生命体征**：心率、血压、体温、血氧饱和度
    
*   **评分**：GCS（眼、运动、语言）、APACHE II & III 评分
    
*   **合并症**：糖尿病、肝功能衰竭、艾滋病、白血病、癌症
    
*   **目标变量**：`hospital_death`（1=死亡，0=存活）
    

💡 实现模型
-------

| 模型 | 文件 | 描述 |
| --- | --- | --- |
| 逻辑回归 | `LogisticRegression.py` | 线性基准分类器 |
| XGBoost | `Xgboost.py` | 梯度提升决策树 |
| DNN | `DNN.py` | 多层感知机 (MLP) |
| 堆叠集成 | `Ensemble.py` | 通过元学习器组合 DNN、LR、XGB |
| 软投票 | `Ensemble.py` | 对预测概率进行加权平均 |

* * *

⚙️ 安装与配置
--------

### 第 1 步：创建虚拟环境

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### 第 2 步：安装依赖

```bash
pip install -r requirements.txt
```

* * *

🚀 运行指南
-------

### 📌 1. 运行相关性分析

```bash
python src/correlation_analysis/categorical_analysis.py
python src/correlation_analysis/numerical_analysis.py
```

### 📌 2. 数据探索（可选）

打开 `notebook/EDA.ipynb`，探索缺失值、分布等。

* * *

### 📌 3. 训练单模型

```bash
python src/model/Xgboost.py
python src/model/LogisticRegression.py
python src/model/DNN.py
```

输出：

*   模型文件保存在 `Model/`
    
*   可视化结果保存在 `visualization/`
    

### 📌 4. 训练集成模型

```bash
python src/model/Ensemble.py
```

此步骤将：

*   执行 5 折交叉验证
    
*   保存基础模型与元模型
    
*   输出集成模型指标
    

### 📌 5. 启动 GUI 界面

```bash
streamlit patient_risk_interface.py
```

GUI 功能：

*   输入患者特征
    
*   点击“Predict”
    
*   查看死亡风险（0–1 概率）
    

📈 可视化
------

生成的图表保存在 `visualization/` 下：

*   混淆矩阵
    
*   ROC 曲线
    
*   F1 分数对比
    
*   特征重要性柱状图
    

🧰 依赖要求
-------

```txt
numpy
pandas
scikit-learn
xgboost
tensorflow
keras
matplotlib
seaborn
joblib
tkinter
```

统一安装：

```bash
pip install -r requirements.txt
```

* * *

🧠 项目亮点
-------

*   ✅ 模块化、清晰的架构
    
*   ✅ 可复用的训练与测试脚本
    
*   ✅ 实时风险预测 UI
    
*   ✅ 透明的特征文档
    
*   ✅ 日志文件追踪分析流程
    
