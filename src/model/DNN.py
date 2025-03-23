import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (roc_auc_score, accuracy_score,
                             confusion_matrix, classification_report)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from imblearn.over_sampling import SMOTE

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow

from src.config.path_config import DATA_FILE, DNN_VIS_DIR
from src.common.variable_array import final_numerical_arr, categorical_array
from src.visualization.plot_utils import plot_confusion_matrix, plot_roc_curve


# 封装 DNN 模型主流程为函数
def DNNModel(df):
    # 1. 特征定义：数值型和类别型
    numeric_features = final_numerical_arr
    categorical_features = categorical_array

    # 2. 创建数据预处理管道
    preprocessor = ColumnTransformer(transformers=[
        # 数值特征预处理：缺失值填充为中位数 + 标准化
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numeric_features),

        # 类别特征预处理：填充缺失为"missing" + 独热编码
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_features)
    ])

    # 3. 数据分离与处理
    X = df.drop('hospital_death', axis=1)  # 特征
    y = df['hospital_death']  # 标签
    X_processed = preprocessor.fit_transform(X)  # 执行预处理

    # 4. 划分训练集与验证集（按原始类别比例 stratify）
    X_train, X_val, y_train, y_val = train_test_split(
        X_processed, y, test_size=0.2, random_state=42, stratify=y)

    # 5. SMOTE 过采样处理类别不平衡（提升对少数类 recall）
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # 6. 构建 DNN 网络结构
    input_shape = X_train_res.shape[1]  # 输入特征维度
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

        Dense(1, activation='sigmoid')  # 输出层：sigmoid 输出死亡概率
    ])

    # 7. 编译模型：使用 Adam 优化器，二分类交叉熵损失，评估指标 AUC
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy', tensorflow.keras.metrics.AUC(name='auc')])

    # 8. 设置训练回调：早停 + 自适应学习率调整
    callbacks = [
        EarlyStopping(patience=10, monitor='val_auc', mode='max', verbose=1),
        ReduceLROnPlateau(factor=0.1, patience=5, min_lr=1e-6, verbose=1)
    ]

    # 9. 模型训练
    history = model.fit(
        X_train_res, y_train_res,
        validation_split=0.2,
        epochs=100,
        batch_size=64,
        callbacks=callbacks,
        verbose=1
    )

    # 10. 模型评估
    y_pred = (model.predict(X_val) > 0.5).astype(int)  # 二分类输出
    y_proba = model.predict(X_val)  # 概率输出

    print("Model: Deep Neural Network (Keras)")
    print("Accuracy:", accuracy_score(y_val, y_pred))
    print("AUC-ROC:", roc_auc_score(y_val, y_proba))
    print(classification_report(y_val, y_pred))

    # 11. 绘制模型 ROC 曲线与混淆矩阵
    plot_roc_curve(y_val, y_proba, DNN_VIS_DIR / "dnn_roc_curve.png")
    print("ROC 曲线图已保存至：", DNN_VIS_DIR / "dnn_roc_curve.png")

    plot_confusion_matrix(y_val, y_pred, DNN_VIS_DIR / "dnn_confusion_matrix.png")
    print("混淆矩阵图已保存至：", DNN_VIS_DIR / "dnn_confusion_matrix.png")

    # 12. 可视化训练过程：AUC 曲线与 Loss 曲线
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['auc'], label='Train AUC')
    plt.plot(history.history['val_auc'], label='Validation AUC')
    plt.title('AUC 曲线')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss 曲线')
    plt.legend()
    plt.savefig(DNN_VIS_DIR / "dnn_training_curve.png")
    print("训练曲线图已保存至：", DNN_VIS_DIR / "dnn_training_curve.png")


if __name__ == '__main__':
    df = pd.read_csv(DATA_FILE)
    DNNModel(df)
