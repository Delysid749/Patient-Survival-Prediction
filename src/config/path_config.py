from pathlib import Path

"""
定义全局路径
"""

# 项目根目录
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# 数据路径
DATA_DIR = BASE_DIR / 'dataSet'
DATA_FILE = DATA_DIR / 'dataset.csv'

# 报告路径
REPORT_DIR = BASE_DIR / 'report'
LOG_DIR = REPORT_DIR / 'logs'

# 图表可视化路径
VIS_DIR = BASE_DIR / 'src/visualization'
LR_VIS_DIR = VIS_DIR / 'LogisticRegression'
DNN_VIS_DIR = VIS_DIR / 'DNN'
XG_VIS_DIR = VIS_DIR / 'XgBoost'
EN_VIS_DIR = VIS_DIR / 'Ensemble'

# 模型保存路径
MODEL_DIR = BASE_DIR / 'Model'