
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


def DNNModel(df):
    # 特征定义
    numeric_features = final_numerical_arr
    categorical_features = categorical_array

    # 创建预处理管道
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), numeric_features),
            ('cat', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('encoder', OneHotEncoder(handle_unknown='ignore'))
            ]), categorical_features)
        ])

    # 分离 X 和 y
    X = df.drop('hospital_death', axis=1)
    y = df['hospital_death']
    X_processed = preprocessor.fit_transform(X)

    # 数据划分
    X_train, X_val, y_train, y_val = train_test_split(
        X_processed, y, test_size=0.2, random_state=42, stratify=y)

    # 处理类别不平衡
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # 构建模型
    input_shape = X_train_res.shape[1]
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
                  metrics=['accuracy', tensorflow.keras.metrics.AUC(name='auc')])

    # 设置回调
    callbacks = [
        EarlyStopping(patience=10, monitor='val_auc', mode='max', verbose=1),
        ReduceLROnPlateau(factor=0.1, patience=5, min_lr=1e-6, verbose=1)
    ]

    # 模型训练
    history = model.fit(
        X_train_res, y_train_res,
        validation_split=0.2,
        epochs=100,
        batch_size=64,
        callbacks=callbacks,
        verbose=1
    )

    # 预测与评估
    y_pred = (model.predict(X_val) > 0.5).astype(int)
    y_proba = model.predict(X_val)

    print("Model: Deep Neural Network (Keras)")
    print("Accuracy:", accuracy_score(y_val, y_pred))
    print("AUC-ROC:", roc_auc_score(y_val, y_proba))
    print(classification_report(y_val, y_pred))

    plot_roc_curve(y_val, y_proba, DNN_VIS_DIR / "dnn_roc_curve.png")
    print("ROC 曲线图已保存至：", DNN_VIS_DIR / "dnn_roc_curve.png")

    plot_confusion_matrix(y_val, y_pred, DNN_VIS_DIR / "dnn_confusion_matrix.png")
    print("混淆矩阵图已保存至：", DNN_VIS_DIR / "dnn_confusion_matrix.png")

    # 可视化训练过程
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['auc'], label='Train AUC')
    plt.plot(history.history['val_auc'], label='Validation AUC')
    plt.title('AUC curve')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('LOSS curve')
    plt.legend()
    plt.savefig(DNN_VIS_DIR / "dnn_training_curve.png")
    print("训练曲线图已保存至：", DNN_VIS_DIR / "dnn_training_curve.png")


if __name__ == '__main__':
    df = pd.read_csv(DATA_FILE)
    DNNModel(df)
