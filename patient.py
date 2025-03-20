import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
from sklearn.metrics import (accuracy_score, roc_auc_score, 
                             confusion_matrix, classification_report,
                             RocCurveDisplay)
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

# 1. 数据加载
data = pd.read_csv('dataset.csv')  # 替换为你的数据路径

# 2. 数据预处理
# 删除不相关的ID列
data = data.drop(['encounter_id', 'patient_id', 'hospital_id', 'icu_id'], axis=1)
data = data.drop(columns=['Unnamed: 83'], errors='ignore')
# 处理缺失值（示例处理，需根据实际数据调整）
# 数值列用中位数填充
numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())

# 分类列用 'missing' 填充
categorical_cols = data.select_dtypes(include=['object']).columns
data = data.dropna(subset=categorical_cols)
 

# 3. 特征工程
# 定义分类特征和数值特征
categorical_features = ['elective_surgery', 'ethnicity', 'gender', 
                       'icu_admit_source', 'icu_stay_type', 'icu_type',
                       'apache_3j_bodysystem', 'apache_2_bodysystem']
numeric_features = [col for col in data.columns 
                   if col not in categorical_features + ['hospital_death']]

# 4. 创建预处理管道
# 这里加入SimpleImputer来处理缺失值
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),  # 对数值列进行中位数填充
            ('scaler', StandardScaler())  # 标准化
        ]), numeric_features),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),  # 对分类列进行'missing'填充
            ('encoder', OneHotEncoder(handle_unknown='ignore'))  # 独热编码
        ]), categorical_features)
    ])

# 5. 划分数据集
X = data.drop('hospital_death', axis=1)
y = data['hospital_death']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# 6. 创建模型管道（去掉 class_weight 参数）
logreg_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(class_weight='balanced', max_iter=1000))
])

tree_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(class_weight='balanced', max_depth=5))
])

gb_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', GradientBoostingClassifier())  # 去掉 class_weight
])

xgb_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', xgb.XGBClassifier(
        device = "cuda",
        scale_pos_weight=1, 
        use_label_encoder=False, 
        eval_metric='logloss'))
])


# 7. 训练和评估函数
def train_evaluate_model(pipeline, model_name):
    # 训练模型
    pipeline.fit(X_train, y_train)
    
    # 预测
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    
    # 评估指标
    print(f"\n{model_name} 评估结果:")
    print("准确率:", accuracy_score(y_test, y_pred))
    print("AUC:", roc_auc_score(y_test, y_proba))
    print("\n分类报告:")
    print(classification_report(y_test, y_pred))
    
    # 绘制ROC曲线
    RocCurveDisplay.from_estimator(pipeline, X_test, y_test)
    plt.title(f'{model_name} ROC Curve')
    plt.show()
    
    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    print("混淆矩阵:")
    print(cm)

# 8. 训练和评估模型
train_evaluate_model(logreg_pipeline, "logisticsRegression")
train_evaluate_model(tree_pipeline, "decisionTree")
train_evaluate_model(gb_pipeline, "gradient")
train_evaluate_model(xgb_pipeline, "XGBoost")

# 9. 特征重要性分析（以决策树为例）
# 提取预处理后的特征名称
preprocessor.fit(X)
feature_names = (numeric_features + 
                list(preprocessor.named_transformers_['cat']
                     .get_feature_names_out(categorical_features)))

# 获取特征重要性
importances = tree_pipeline.named_steps['classifier'].feature_importances_
indices = np.argsort(importances)[-10:]  # 取最重要的10个特征

# 绘制特征重要性
plt.figure(figsize=(10, 6))
plt.title('top 10 features')
plt.barh(range(len(indices)), importances[indices], align='center')
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel('feature importance')
plt.show()

import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
import joblib

# 假设你已经训练好了这些模型，并且已经保存为joblib文件
#logreg_pipeline = joblib.load('logreg_pipeline.joblib')  # 加载保存的模型

# 获取相关性最高的特征
top_features = ['elective_surgery', 'ethnicity', 'gender', 'apache_3j_bodysystem', 'icu_admit_source']

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
    all_features = [ 'age', 'bmi', 'elective_surgery', 'ethnicity', 
                    'gender', 'height', 'icu_admit_source',  'icu_stay_type', 'icu_type', 'pre_icu_los_days',
                    'weight', 'apache_2_diagnosis', 'apache_3j_diagnosis', 'apache_post_operative', 'arf_apache',
                    'gcs_eyes_apache', 'gcs_motor_apache', 'gcs_unable_apache', 'gcs_verbal_apache', 'heart_rate_apache',
                    'intubated_apache', 'map_apache', 'resprate_apache', 'temp_apache', 'ventilated_apache', 'd1_diasbp_max',
                    'd1_diasbp_min', 'd1_diasbp_noninvasive_max', 'd1_diasbp_noninvasive_min', 'd1_heartrate_max',
                    'd1_heartrate_min', 'd1_mbp_max', 'd1_mbp_min', 'd1_mbp_noninvasive_max', 'd1_mbp_noninvasive_min',
                    'd1_resprate_max', 'd1_resprate_min', 'd1_spo2_max', 'd1_spo2_min', 'd1_sysbp_max', 'd1_sysbp_min',
                    'd1_sysbp_noninvasive_max', 'd1_sysbp_noninvasive_min', 'd1_temp_max', 'd1_temp_min', 'h1_diasbp_max',
                    'h1_diasbp_min', 'h1_diasbp_noninvasive_max', 'h1_diasbp_noninvasive_min', 'h1_heartrate_max',
                    'h1_heartrate_min', 'h1_mbp_max', 'h1_mbp_min', 'h1_mbp_noninvasive_max', 'h1_mbp_noninvasive_min',
                    'h1_resprate_max', 'h1_resprate_min', 'h1_spo2_max', 'h1_spo2_min', 'h1_sysbp_max', 'h1_sysbp_min',
                    'h1_sysbp_noninvasive_max', 'h1_sysbp_noninvasive_min', 'd1_glucose_max', 'd1_glucose_min',
                    'd1_potassium_max', 'd1_potassium_min', 'apache_4a_hospital_death_prob', 'apache_4a_icu_death_prob',
                    'aids', 'cirrhosis', 'diabetes_mellitus', 'hepatic_failure', 'immunosuppression', 'leukemia',
                    'lymphoma', 'solid_tumor_with_metastasis', 'apache_3j_bodysystem', 'apache_2_bodysystem', 'hospital_death']
    
    # 填补所有缺失的特征（包括非用户输入的特征），用众数填补
    for feature in all_features:
        if feature not in input_data:  # 如果用户没有提供该特征
            input_data[feature] = data[feature].mode()[0]  # 用训练数据中的众数填充

    # 转换输入数据为DataFrame
    input_df = pd.DataFrame([input_data])

    # 数据预处理
    input_processed = logreg_pipeline.named_steps['preprocessor'].transform(input_df)

    # 进行预测
    prediction = logreg_pipeline.predict(input_processed)
    prediction_proba = logreg_pipeline.predict_proba(input_processed)[:, 1]

    # 显示预测结果
    result = "Death" if prediction[0] == 1 else "Survival"
    prob = prediction_proba[0]
    messagebox.showinfo("Prediction Result", f"Predicted Outcome: {result}\nProbability: {prob:.2f}")



# 创建预测按钮
predict_button = ttk.Button(root, text="Predict", command=predict, width=20)
predict_button.pack(pady=20)

# 运行主循环
root.mainloop()



from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
from sklearn.model_selection import cross_val_score,cross_validate


# 1. 创建基础模型（第一层模型）
base_learners = [
    ('tree', Pipeline(steps=[
        ('preprocessor', preprocessor),  # 使用同样的预处理管道
        ('classifier', DecisionTreeClassifier(class_weight='balanced', max_depth=5))
    ])),
    ('gb', Pipeline(steps=[
        ('preprocessor', preprocessor),  # 使用同样的预处理管道
        ('classifier', GradientBoostingClassifier())
    ])),
    ('xgb', Pipeline(steps=[
        ('preprocessor', preprocessor),  # 使用同样的预处理管道
        ('classifier', xgb.XGBClassifier(
            tree_method='gpu_hist',  # 使用GPU加速
            gpu_id=0,  # 选择GPU设备（如果有多个GPU可以选择其他编号）
            scale_pos_weight=1, 
            use_label_encoder=False, 
            eval_metric='logloss'))
    ]))
]

# 2. 创建元模型（第二层模型）
meta_model = LogisticRegression(class_weight='balanced', max_iter=1000)

# 3. 创建堆叠法模型
stacking_pipeline = StackingClassifier(
    estimators=base_learners, 
    final_estimator=meta_model
)

# 4. 使用交叉验证对堆叠法模型进行评估
cv_scores = cross_val_score(stacking_pipeline, X, y, cv=5, scoring='accuracy')
print(f"cross_validate_accuracy: {cv_scores}")
print(f"Average accuracy: {cv_scores.mean()}")


# 5. 使用 cross_validate 进行详细的交叉验证评估
cv_results = cross_validate(stacking_pipeline, X, y, cv=5, 
                            scoring=['accuracy', 'roc_auc'], return_train_score=False)
print(f"cross_validate_accuracy: {cv_results['test_accuracy']}")
print(f"cross_validate AUC: {cv_results['test_roc_auc']}")
print(f"Average accuracy: {cv_results['test_accuracy'].mean()}")
print(f"Average AUC: {cv_results['test_roc_auc'].mean()}")


# 绘制交叉验证准确率的箱型图
plt.boxplot(cv_scores, vert=False)
plt.title('cross_validate_accuracy')
plt.xlabel('accuracy')
plt.show()

# 绘制交叉验证 AUC 的箱型图
plt.boxplot(cv_results['test_roc_auc'], vert=False)
plt.title('cross_validate AUC')
plt.xlabel('AUC')
plt.show()
