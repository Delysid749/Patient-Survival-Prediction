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
from sklearn.model_selection import StratifiedKFold
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
    # 1. 读取数据
    df = pd.read_csv(DATA_FILE)
    X = df[final_numerical_arr + categorical_array]
    y = df["hospital_death"]

    # 2. 创建目录
    os.makedirs(DNN_VIS_DIR, exist_ok=True)

    # 3. 设置5折交叉验证
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 4. 初始化收集容器
    all_y_true = []
    all_y_pred = []
    all_y_prob = []

    loss_records = []
    val_loss_records = []

    accuracy_list = []
    auc_list = []

    fold = 1
    for train_idx, test_idx in skf.split(X, y):
        print(f"\n========== 第 {fold} 折训练 ==========")
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # 训练新模型
        model = DNNModel(epochs=100, batch_size=64)
        model.fit(X_train, y_train)

        # 预测
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # 收集评估
        acc = accuracy_score(y_test, y_pred)
        auc = tf.keras.metrics.AUC()(y_test, y_prob).numpy()
        print(f"准确率：{acc:.4f} | AUC值：{auc:.4f}")
        print("分类报告：")
        print(classification_report(y_test, y_pred))

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        all_y_prob.extend(y_prob)

        accuracy_list.append(acc)
        auc_list.append(auc)

        loss_records.append(model.history.history['loss'])
        val_loss_records.append(model.history.history['val_loss'])

        fold += 1

    # 5. 绘制结果路径
    ROC_PATH = os.path.join(DNN_VIS_DIR, "dnn_mean_roc.png")
    CM_PATH = os.path.join(DNN_VIS_DIR, "dnn_mean_cm.png")
    LOSS_PATH = os.path.join(DNN_VIS_DIR, "dnn_mean_loss.png")

    # 6. 绘图输出
    plot_roc_curve(all_y_true, all_y_prob, ROC_PATH)
    plot_confusion_matrix(all_y_true, all_y_pred, CM_PATH)

    # 7. 汇总 loss 曲线（对齐长度，取最小长度）
    min_len = min(len(loss) for loss in loss_records)
    mean_train_loss = np.mean([loss[:min_len] for loss in loss_records], axis=0)
    mean_val_loss = np.mean([val[:min_len] for val in val_loss_records], axis=0)


    # 构造一个假的 history 对象用于绘图
    class DummyHistory:
        def __init__(self, train_loss, val_loss):
            self.history = {
                'loss': train_loss,
                'val_loss': val_loss
            }


    dummy_history = DummyHistory(mean_train_loss, mean_val_loss)
    plot_loss_curve(dummy_history, LOSS_PATH)

    # 8. 汇总打印
    print("\n========== 交叉验证汇总 ==========")
    print(f"平均准确率: {np.mean(accuracy_list):.4f} ± {np.std(accuracy_list):.4f}")
    print(f"平均 AUC: {np.mean(auc_list):.4f} ± {np.std(auc_list):.4f}")
    print(f"\nROC 曲线已保存到: {ROC_PATH}")
    print(f"混淆矩阵图已保存到: {CM_PATH}")
    print(f"Loss 曲线图已保存到: {LOSS_PATH}")
