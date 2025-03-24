import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from src.common.variable_array import numerical_array
from src.config.path_config import DATA_FILE,LOG_DIR
import os

"""
使用pearson系数检验每个数值型变量与target - hospital_death 的相关性
"""
def analyze_numerical_correlation(df, target='hospital_death', threshold=0.05, save_path=None):
    """
    分析所有数值型变量与目标变量之间的皮尔逊相关系数，并筛选出高于阈值的特征。

    参数:
        df: DataFrame，数据集
        target: str，目标变量名
        threshold: float，保留特征的相关性阈值
        save_path: str，可选，若提供则保存热力图

    返回:
        strong_corr_features: list，相关性大于阈值的数值型变量名
        corr_series: Series，全部数值型变量与目标的相关性排序
    """
    # 取出数值型变量 + 目标变量的子集
    corr_df = df[numerical_array].copy()

    # 删除不能计算的列（非数值或全为NaN）
    corr_df = corr_df.select_dtypes(include=['number']).dropna(axis=1, how='all')

    # 计算相关性矩阵
    corr_matrix = corr_df.corr()

    # 提取与目标变量的相关性（绝对值排序）
    corr_series = corr_matrix[target].drop(target).abs().sort_values(ascending=False)

    # 输出所有相关性
    print("各数值型变量与 hospital_death 的皮尔逊相关系数：")
    print(corr_series)

    # 提取与目标变量的相关性（保留正负号）
    corr_series_signed = corr_matrix[target].drop(target).sort_values(ascending=False)

    print("\n 与 hospital_death 的相关性（降序排列，保留正负符号）：")
    for feature, corr in corr_series_signed.items():
        print(f"{feature:35s} {corr:.4f}")

    strong_corr_features = corr_series_signed[abs(corr_series_signed) > threshold].index.tolist()
    return strong_corr_features, corr_series_signed



# 将输出结果保存到 logs 文件夹中
def save_correlation_to_log(corr_series, selected_features, log_path=LOG_DIR / "numerical_correlation.log",threshold=0.05):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("与 hospital_death 的相关性（降序排列，保留正负符号）：\n")
        for feature, corr in corr_series.items():
            f.write(f"{feature:35s} {corr:.4f}\n")

        f.write(f"\n筛选出的相关性 > 阈值{threshold} 的数值型变量：\n")
        for feature in selected_features:
            f.write(f"{feature}\n")

        f.write(f"\n低于阈值{threshold}（被剔除）的数值型变量：\n")
        for feature, corr in corr_series.items():
            if abs(corr) <= threshold:
                f.write(f"{feature:35s} {corr:.4f}\n")
    print(f"\n 相关性分析结果已保存至：{log_path}")


if __name__ == '__main__':
    df = pd.read_csv(DATA_FILE)
    selected_features, correlation_scores = analyze_numerical_correlation(df, threshold=0.05)
    save_correlation_to_log(correlation_scores, selected_features)
    print(f"\n被剔除的低相关性数值型数值型变量（相关性 ≤ 阈值{0.05}）:")
    for feature, corr in correlation_scores.items():
        if abs(corr) <= 0.05:
            print(f"{feature:35s} {corr:.4f}")