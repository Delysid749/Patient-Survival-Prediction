import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
from src.config.path_config import DATA_FILE

def data_loading():
    df = pd.read_csv(DATA_FILE)
    print("前5行数据预览：")
    print(df.head())

    # 输出数值型变量
    numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
    print("\n数值变量:")
    print(numerical_cols)

    # 输出分类变量
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    print("\n分类变量:")
    print(categorical_cols)


if __name__ == '__main__':
    data_loading()
