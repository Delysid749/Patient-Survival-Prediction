import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone
from sklearn.metrics import accuracy_score, classification_report
from src.config.path_config import DATA_FILE,LOG_DIR
# 导入重构后的模型
from DNN import DNNModel
from LogisticRegression import LogisticModel
from Xgboost import XGBoostModel


def get_oof_predictions(model, X, y, X_test, n_splits=5):
    """
    生成基模型的 Out-Of-Fold（OOF）预测，用于堆叠集成。

    参数：
        model: 基模型实例（需实现 fit 与 predict_proba 方法）
        X: 训练特征（DataFrame）
        y: 训练标签（Series）
        X_test: 测试特征（DataFrame）
        n_splits: KFold 折数

    返回：
        oof_train: 训练集上每个样本的正类预测概率，形状为 (n_samples, 1)
        oof_test: 测试集上各折预测的均值，形状为 (n_samples, 1)
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_train = np.zeros(X.shape[0])
    oof_test = np.zeros((X_test.shape[0], n_splits))

    for i, (train_idx, val_idx) in enumerate(kf.split(X)):
        # 重新克隆模型，确保每一折独立训练
        instance = clone(model)
        X_train_fold = X.iloc[train_idx]
        y_train_fold = y.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]

        instance.fit(X_train_fold, y_train_fold)
        # 获取验证集预测的正类概率
        oof_train[val_idx] = instance.predict_proba(X_val_fold)[:, 1]
        # 测试集上预测
        oof_test[:, i] = instance.predict_proba(X_test)[:, 1]

    # 对测试集预测取平均，形成稳定输出
    oof_test_mean = oof_test.mean(axis=1)
    return oof_train.reshape(-1, 1), oof_test_mean.reshape(-1, 1)


def stacking_ensemble(X, y, X_test):
    """
    构建堆叠集成模型，将基模型的预测作为新特征，训练 Meta 模型。

    参数：
        X: 训练特征（DataFrame）
        y: 训练标签（Series）
        X_test: 测试特征（DataFrame）

    返回：
        final_pred: Meta 模型对测试集的最终预测结果
        meta_model: 训练好的 Meta 模型
    """
    # 初始化基模型列表
    base_models = [DNNModel(), LogisticModel(), XGBoostModel()]
    meta_train_features = []
    meta_test_features = []

    # 生成每个基模型的 OOF 预测
    for model in base_models:
        train_pred, test_pred = get_oof_predictions(model, X, y, X_test, n_splits=5)
        meta_train_features.append(train_pred)
        meta_test_features.append(test_pred)

    # 拼接所有基模型的预测作为 Meta 层的特征矩阵
    X_meta_train = np.hstack(meta_train_features)
    X_meta_test = np.hstack(meta_test_features)

    print("Meta training feature shape:", X_meta_train.shape)
    print("Meta testing feature shape:", X_meta_test.shape)

    # 使用逻辑回归作为 Meta 模型
    meta_model = LogisticRegression()
    meta_model.fit(X_meta_train, y)
    final_pred = meta_model.predict(X_meta_test)

    return final_pred, meta_model


if __name__ == '__main__':
    # 加载数据（请替换为您的实际数据路径）
    df = pd.read_csv(DATA_FILE)

    # 分离特征与标签，确保特征名称与各模型重构时一致
    X = df.drop("hospital_death", axis=1)
    y = df["hospital_death"]

    # 将数据拆分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 运行堆叠集成
    final_predictions, meta_model = stacking_ensemble(X_train, y_train, X_test)

    # 输出评估结果
    print("Stacking Ensemble Accuracy:", accuracy_score(y_test, final_predictions))
    print("Classification Report:")
    print(classification_report(y_test, final_predictions))
