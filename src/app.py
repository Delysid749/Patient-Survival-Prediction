import pandas as pd
from src.config.path_config import DATA_FILE


# 读取数据
df = pd.read_csv(DATA_FILE)

# 指定要提取的 encounter_id
target_id = 66154

# 要提取的特征字段
TOP_N_NUMERIC_FIELDS = [
    "apache_4a_hospital_death_prob",
    "apache_4a_icu_death_prob",
    "ventilated_apache",
    "intubated_apache",
    "d1_heartrate_max"
]

TOP_CATEGORICAL_FIELDS = [
    "apache_3j_bodysystem", "icu_admit_source", "apache_2_bodysystem",
    "icu_type", "icu_stay_type", "ethnicity", "gender"
]

ALL_FIELDS = TOP_N_NUMERIC_FIELDS + TOP_CATEGORICAL_FIELDS

# 查找该患者数据
row = df[df["encounter_id"] == target_id]

# 仅保留所需特征列
if not row.empty:
    input_data = row[ALL_FIELDS]
    print("提取的特征数据如下：")
    print(input_data.T)  # 转置方便查看
else:
    print(f"未找到 encounter_id 为 {target_id} 的数据行")
