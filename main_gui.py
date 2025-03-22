import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
import joblib
data = pd.read_csv('dataSet/dataset.csv')
load_pipeline = joblib.load('xgb_pipeline.joblib')
# 假设你已经训练好了这些模型，并且已经保存为joblib文件

"""
如下方式保存模型，可以使pipeline,也可以是xgb.XGBClassifier()等其他模型
import joblib
joblib.dump(xgb_pipeline, 'xgb_pipeline.joblib')

"""
# 获取相关性最高的特征
# top_features = ['elective_surgery', 'ethnicity', 'gender', 'apache_3j_bodysystem', 'icu_admit_source']
top_features = [
    'apache_4a_hospital_death_prob', 'apache_4a_icu_death_prob', 'd1_spo2_min',
    'ventilated_apache', 'd1_sysbp_noninvasive_min', 'd1_sysbp_min',
]
# 创建主窗口
root = tk.Tk()
root.title("Hospital Death Prediction")

# 设置窗口大小
root.geometry("400x450")
root.config(bg="#f5f5f5")

# 设置标题
title_label = ttk.Label(root, text="Predict Patient's Outcome", font=("Helvetica", 16), anchor="center")
title_label.pack(pady=10)

# 创建标签和输入框
labels = {}
entries = {}

frame = ttk.Frame(root, padding="10")
frame.pack(padx=20, pady=20)

for feature in top_features:
    # 为每个特征创建标签
    label = ttk.Label(frame, text=feature, font=("Helvetica", 10), anchor="w")
    label.grid(row=top_features.index(feature), column=0, pady=5, sticky="w")

    # 为每个特征创建输入框
    entry = ttk.Entry(frame, width=25, font=("Helvetica", 10))
    entry.grid(row=top_features.index(feature), column=1, pady=5)
    entries[feature] = entry


# 定义预测函数
def predict():
    input_data = {}

    # 获取用户输入的数据
    for feature in top_features:
        value = entries[feature].get()
        if value:
            input_data[feature] = value
        else:
            # 如果输入为空，则用众数代替
            input_data[feature] = data[feature].mode()[0]  # 使用训练数据中的众数

    # 补齐所有模型需要的特征
    all_features = ['age', 'bmi', 'elective_surgery', 'ethnicity',
                    'gender', 'height', 'icu_admit_source', 'icu_stay_type', 'icu_type', 'pre_icu_los_days',
                    'weight', 'apache_2_diagnosis', 'apache_3j_diagnosis', 'apache_post_operative', 'arf_apache',
                    'gcs_eyes_apache', 'gcs_motor_apache', 'gcs_unable_apache', 'gcs_verbal_apache',
                    'heart_rate_apache',
                    'intubated_apache', 'map_apache', 'resprate_apache', 'temp_apache', 'ventilated_apache',
                    'd1_diasbp_max',
                    'd1_diasbp_min', 'd1_diasbp_noninvasive_max', 'd1_diasbp_noninvasive_min', 'd1_heartrate_max',
                    'd1_heartrate_min', 'd1_mbp_max', 'd1_mbp_min', 'd1_mbp_noninvasive_max',
                    'd1_mbp_noninvasive_min',
                    'd1_resprate_max', 'd1_resprate_min', 'd1_spo2_max', 'd1_spo2_min', 'd1_sysbp_max',
                    'd1_sysbp_min',
                    'd1_sysbp_noninvasive_max', 'd1_sysbp_noninvasive_min', 'd1_temp_max', 'd1_temp_min',
                    'h1_diasbp_max',
                    'h1_diasbp_min', 'h1_diasbp_noninvasive_max', 'h1_diasbp_noninvasive_min', 'h1_heartrate_max',
                    'h1_heartrate_min', 'h1_mbp_max', 'h1_mbp_min', 'h1_mbp_noninvasive_max',
                    'h1_mbp_noninvasive_min',
                    'h1_resprate_max', 'h1_resprate_min', 'h1_spo2_max', 'h1_spo2_min', 'h1_sysbp_max',
                    'h1_sysbp_min',
                    'h1_sysbp_noninvasive_max', 'h1_sysbp_noninvasive_min', 'd1_glucose_max', 'd1_glucose_min',
                    'd1_potassium_max', 'd1_potassium_min', 'apache_4a_hospital_death_prob',
                    'apache_4a_icu_death_prob',
                    'aids', 'cirrhosis', 'diabetes_mellitus', 'hepatic_failure', 'immunosuppression', 'leukemia',
                    'lymphoma', 'solid_tumor_with_metastasis', 'apache_3j_bodysystem', 'apache_2_bodysystem']

    # 填补所有缺失的特征（包括非用户输入的特征），用众数填补
    for feature in all_features:
        if feature not in input_data:  # 如果用户没有提供该特征
            input_data[feature] = data[feature].mode()[0]  # 用训练数据中的众数填充

    # 转换输入数据为DataFrame
    input_df = pd.DataFrame([input_data])

    prediction = load_pipeline.predict(input_df)
    prediction_proba = load_pipeline.predict_proba(input_df)[:, 1]

    # 显示预测结果
    result = "Death" if prediction[0] == 1 else "Survival"
    prob = prediction_proba[0]
    messagebox.showinfo("Prediction Result", f"Predicted Outcome: {result}\nProbability: {prob:.2f}")


# 创建预测按钮
predict_button = ttk.Button(root, text="Predict", command=predict, width=20)
predict_button.pack(pady=20)

# 运行主循环
root.mainloop()


