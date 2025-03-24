# 患者生存预测系统

## 项目简介
本项目是一个基于机器学习的患者生存预测系统，旨在通过分析患者的各项生理指标和临床数据，预测患者在ICU中的生存情况。该系统可以帮助医疗团队更好地评估患者风险，制定更有效的治疗方案。

## 主要功能
- 患者数据分析和可视化
- 多维度特征相关性分析
- 机器学习模型训练和预测
- 交互式Web界面
- 实时风险评估

## 技术栈
- Python 3.x
- 机器学习库：scikit-learn, pandas, numpy
- Web框架：Flask
- 数据可视化：matplotlib, seaborn
- 数据处理：pandas, numpy

## 项目结构
```
├── src/                    # 源代码目录
│   ├── app.py             # Web应用主入口
│   ├── patient_risk_interface.py  # 患者风险评估接口
│   ├── visualization/     # 数据可视化模块
│   ├── model/            # 机器学习模型
│   ├── config/           # 配置文件
│   ├── common/           # 公共工具函数
│   ├── correlation_analysis/  # 相关性分析模块
│   └── notebook/         # Jupyter notebooks
├── dataSet/              # 数据集目录
├── Model/                # 训练好的模型存储
├── report/              # 项目报告和文档
└── requirements.txt     # 项目依赖
```

## 数据特征
系统使用的主要特征包括：
- 患者基本信息（年龄、性别、BMI等）
- 生命体征数据（心率、血压、体温等）
- 实验室检查结果
- APACHE评分相关指标
- ICU相关指标

## 安装说明
1. 克隆项目到本地
```bash
git clone [项目地址]
```

2. 创建并激活虚拟环境
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# 或
.venv\Scripts\activate  # Windows
```

3. 安装依赖
```bash
pip install -r requirements.txt
```

## 使用说明
1. 启动Web应用
```bash
python src/app.py
```

2. 访问系统
打开浏览器访问 `http://localhost:5000`

3. 使用预测功能
- 在Web界面输入患者信息
- 系统将自动进行风险评估
- 查看预测结果和详细分析报告

## 模型说明
系统使用多个机器学习模型进行预测：
- 随机森林
- XGBoost
- LightGBM
- 逻辑回归

## 注意事项
- 确保数据格式符合要求
- 定期更新模型
- 注意数据安全性
- 遵循医疗数据使用规范

