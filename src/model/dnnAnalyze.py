import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (roc_auc_score, accuracy_score, 
                             confusion_matrix, classification_report)
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization  
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# 1. 数据加载与预处理（保持与之前相同的预处理流程）
data = pd.read_csv('dataset.csv')
data = data.drop(['encounter_id', 'patient_id', 'hospital_id', 'icu_id'], axis=1)
# 区分数值列和类别列
numerical_cols = data.select_dtypes(include=['number']).columns
categorical_cols = data.select_dtypes(include=['object']).columns

# 只删除空的类别数据
data = data.dropna(subset=categorical_cols)


# 2. 特征工程
categorical_features = ['elective_surgery', 'ethnicity', 'gender', 
                       'icu_admit_source', 'icu_stay_type', 'icu_type',
                       'apache_3j_bodysystem', 'apache_2_bodysystem']
numeric_features = [col for col in data.columns 
                   if col not in categorical_features + ['hospital_death']]

# 3. 创建预处理管道
# 定义预处理步骤
preprocessor = ColumnTransformer(
    transformers=[
        # 数值特征的处理
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),  # 中位数填充
            ('scaler', StandardScaler())  # 标准化
        ]), numeric_features),
        
        # 分类特征的处理
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),  # 填充缺失值
            ('encoder', OneHotEncoder(handle_unknown='ignore'))  # 独热编码
        ]), categorical_features)
    ])

# 4. 数据划分
X = data.drop('hospital_death', axis=1)
y = data['hospital_death']
X_processed = preprocessor.fit_transform(X)  # 先进行预处理
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42, stratify=y)

# 5. 处理类别不平衡
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# 6. 构建DNN模型
def build_dnn(input_shape):
    model = Sequential([
        Dense(256, activation='relu', input_shape=(input_shape,)),
        BatchNormalization(),
        Dropout(0.5),
        
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(1, activation='sigmoid')
    ])
    
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy', 
                           tensorflow.keras.metrics.AUC(name='auc')])
    return model

# 7. 模型训练
input_shape = X_train_res.shape[1]
model = build_dnn(input_shape)

callbacks = [
    EarlyStopping(patience=10, monitor='val_auc', mode='max', verbose=1),
    ReduceLROnPlateau(factor=0.1, patience=5, min_lr=1e-6, verbose=1)
]

history = model.fit(
    X_train_res, y_train_res,
    validation_split=0.2,
    epochs=100,
    batch_size=64,
    callbacks=callbacks,
    class_weight=None  # 根据数据不平衡程度调整权重
)

# 8. 模型评估
def evaluate_model(model, X_test, y_test):
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    y_proba = model.predict(X_test)
    
    print("\nDNN 评估结果:")
    print("准确率:", accuracy_score(y_test, y_pred))
    print("AUC:", roc_auc_score(y_test, y_proba))
    print("\n分类报告:")
    print(classification_report(y_test, y_pred))
    
    # 绘制训练曲线
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['auc'], label='Train AUC')
    plt.plot(history.history['val_auc'], label='Validation AUC')
    plt.title('AUC曲线')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('损失曲线')
    plt.legend()
    plt.show()
    
    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    print("混淆矩阵:")
    print(cm)

evaluate_model(model, X_test, y_test)

# 9. 特征重要性分析（使用Permutation Importance）
from sklearn.inspection import permutation_importance

feature_names = (numeric_features + 
                list(preprocessor.named_transformers_['cat']
                     .get_feature_names_out(categorical_features)))

result = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=42)

sorted_idx = result.importances_mean.argsort()[-10:]

plt.figure(figsize=(10, 6))
plt.title("Top 10 重要特征 (Permutation Importance)")
plt.barh(range(len(sorted_idx)), 
        result.importances_mean[sorted_idx], 
        xerr=result.importances_std[sorted_idx])
plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
plt.xlabel("重要性得分")
plt.show()