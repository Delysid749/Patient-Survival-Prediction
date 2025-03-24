import sys
from pathlib import Path

# 设置项目根目录导入
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import streamlit as st
import pandas as pd
import cloudpickle
from io import StringIO
from src.model.Ensemble import predict_new_patients
from src.config.path_config import MODEL_DIR
from src.common.variable_array import final_numerical_arr, categorical_array

st.set_page_config(page_title="Patient Mortality Prediction", layout="wide")
st.title("住院患者死亡风险预测系统")

# 加载模型
with open(MODEL_DIR / "base_models.pkl", "rb") as f:
    base_models = cloudpickle.load(f)

with open(MODEL_DIR / "meta_model.pkl", "rb") as f:
    meta_model = cloudpickle.load(f)

# 精简字段设置
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

DEFAULT_NUMERIC_VALUES = {
    col: 0.0 for col in final_numerical_arr if col not in TOP_N_NUMERIC_FIELDS
}
DEFAULT_CATEGORICAL_VALUES = {
    col: "missing" for col in categorical_array if col not in TOP_CATEGORICAL_FIELDS
}

st.header("请选择数据输入方式")
input_mode = st.radio("输入方式", ["手动输入单个患者", "上传CSV文件批量预测"])

if input_mode == "手动输入单个患者":
    st.subheader("请输入患者特征信息")
    input_data = {}

    for field in TOP_N_NUMERIC_FIELDS:
        input_data[field] = st.number_input(f"{field}", value=0.0)

    for field in TOP_CATEGORICAL_FIELDS:
        input_data[field] = st.text_input(f"{field}", value="missing")

    input_data.update(DEFAULT_NUMERIC_VALUES)
    input_data.update(DEFAULT_CATEGORICAL_VALUES)
    df = pd.DataFrame([input_data])

    if st.button("预测患者死亡风险"):
        pred, prob = predict_new_patients(df, base_models, meta_model)
        st.success("预测结果：" + ("死亡" if pred[0] == 1 else "存活"))
        st.info(f"死亡概率：{prob[0]:.4f}")

else:
    st.subheader("上传患者数据CSV文件（包含模型需要的字段）")
    file = st.file_uploader("选择CSV文件", type=["csv"])
    if file:
        try:
            df = pd.read_csv(file, encoding="utf-8")
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(file, encoding="gbk")
            except Exception as e:
                st.error(f"文件读取失败：{e}")
                st.stop()

        required_columns = final_numerical_arr + categorical_array
        missing_cols = [col for col in required_columns if col not in df.columns]

        if missing_cols:
            st.error(f"CSV文件缺失以下必要字段:\n{missing_cols}")
        else:
            st.success("字段校验通过，正在进行预测...")
            pred, prob = predict_new_patients(df, base_models, meta_model)
            df["预测结果"] = ["死亡" if p == 1 else "存活" for p in pred]
            df["死亡概率"] = prob
            st.subheader("预测结果展示")
            st.dataframe(df)

            # 添加导出按钮
            csv_buffer = StringIO()
            df.to_csv(csv_buffer, index=False)
            st.download_button(
                label="下载预测结果 CSV",
                data=csv_buffer.getvalue(),
                file_name="prediction_result.csv",
                mime="text/csv"
            )
