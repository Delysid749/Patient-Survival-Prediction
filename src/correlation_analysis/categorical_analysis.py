from src.common.variable_array import categorical_array
from src.config.path_config import DATA_FILE,LOG_DIR
import pandas as pd
from scipy.stats import chi2_contingency
import os

"""
使用卡方检验，检验每个分类变量与target - hospital_death的相关性
"""
def analyze_categorical_correlation(df, target='hospital_death'):
    results = []

    for col in categorical_array:
        try:
            # 构建列联表
            contingency_table = pd.crosstab(df[col], df[target])
            # 执行卡方检验
            chi2, p, dof, expected = chi2_contingency(contingency_table)
            results.append({
                'feature': col,
                'chi2_stat': chi2,
                'p_value': p,
                'significant': p < 0.05
            })
        except Exception as e:
            print(f"Error processing {col}: {e}")

    # 转为 DataFrame 并按 p 值排序
    result_df = pd.DataFrame(results).sort_values(by='p_value')
    print(result_df)

    # 保存到日志
    save_categorical_to_log(result_df)


# 保存卡方检验结果到日志文件
def save_categorical_to_log(result_df, log_path=os.path.join(LOG_DIR, "categorical_correlation.log")):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("分类变量与 hospital_death 的卡方检验结果（按 p 值升序排列）：\n")
        for _, row in result_df.iterrows():
            f.write(f"{row['feature']:30s}  p值={row['p_value']:.4e}  chi2={row['chi2_stat']:.2f}  显著性={row['significant']}\n")
    print(f"\n 卡方检验分析结果已保存至：{log_path}")


if __name__ == '__main__':
    df = pd.read_csv(DATA_FILE)
    analyze_categorical_correlation(df, target='hospital_death')