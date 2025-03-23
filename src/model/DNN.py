# DNN.py
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from src.config.path_config import DATA_FILE, LOG_DIR, DNN_VIS_DIR
from src.common.variable_array import final_numerical_arr, categorical_array
import os
from sklearn.metrics import classification_report, accuracy_score
from src.visualization.plot_utils import plot_roc_curve, plot_confusion_matrix, plot_loss_curve
from sklearn.model_selection import train_test_split

class DNNModel(BaseEstimator, ClassifierMixin):
    def __init__(self, epochs=100, batch_size=64):
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.preprocessor = None

    def fit(self, X, y):
        # 定义预处理器
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]), final_numerical_arr),
                ('cat', Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
                ]), categorical_array)
            ], sparse_threshold=0
        )
        # 预处理特征
        X_processed = self.preprocessor.fit_transform(X)
        if hasattr(X_processed, "toarray"):
            X_processed = X_processed.toarray()

        # SMOTE 过采样
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X_processed, y)

        input_shape = X_res.shape[1]
        # 构建 DNN 网络结构
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(256, activation='relu', input_shape=(input_shape,)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.model.compile(optimizer=optimizer,
                           loss='binary_crossentropy',
                           metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])

        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=10, monitor='val_auc', mode='max', verbose=1),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.1, patience=5, min_lr=1e-6, verbose=1)
        ]
        # 模型训练（validation_split 用于内部验证，不做外部数据分割）
        history = self.model.fit(X_res, y_res, epochs=self.epochs, batch_size=self.batch_size,
                       validation_split=0.2, callbacks=callbacks, verbose=0)
        self.history = history

        return self

    def predict_proba(self, X):
        # 预处理输入数据
        X_processed = self.preprocessor.transform(X)
        if hasattr(X_processed, "toarray"):
            X_processed = X_processed.toarray()
        # 得到预测概率，模型输出 shape 为 (n_samples, 1)
        proba = self.model.predict(X_processed)
        # 将概率转换为两列：[1 - p, p] 形式，以符合 scikit-learn 规范
        proba = np.hstack((1 - proba, proba))
        return proba

    def predict(self, X):
        proba = self.predict_proba(X)[:, 1]
        return (proba >= 0.5).astype(int)


if __name__ == '__main__':
    # 读取数据
    df = pd.read_csv(DATA_FILE)
    X = df.drop("hospital_death", axis=1)
    y = df["hospital_death"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # 初始化并训练 DNN 模型
    model = DNNModel(epochs=100, batch_size=64)
    print("开始训练 DNN 模型...")
    model.fit(X_train, y_train)

    # 模型预测与评估
    print("开始预测...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # 准确率与报告
    print("准确率:", accuracy_score(y_test, y_pred))
    print("分类报告:")
    print(classification_report(y_test, y_pred))

    ROC_PATH = os.path.join(DNN_VIS_DIR, "dnn_roc_curve.png")
    CM_PATH = os.path.join(DNN_VIS_DIR, "dnn_confusion_matrix.png")
    LOSS_PATH = os.path.join(DNN_VIS_DIR, "loss_curve.png")

    plot_roc_curve(y_test, y_prob, ROC_PATH)
    print(f"ROC 曲线已保存到: {ROC_PATH}")

    plot_confusion_matrix(y_test, y_pred, CM_PATH)
    print(f"混淆矩阵图已保存到: {CM_PATH}")

    plot_loss_curve(model.history, LOSS_PATH)
    print(f"Loss 曲线图已保存到: {LOSS_PATH}")
