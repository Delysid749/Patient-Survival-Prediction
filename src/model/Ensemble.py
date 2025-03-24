import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from src.config.path_config import DATA_FILE, LOG_DIR, EN_VIS_DIR, MODEL_DIR
from src.visualization.plot_utils import plot_roc_curve, plot_confusion_matrix
from pathlib import Path
import cloudpickle
from src.common.variable_array import final_numerical_arr, categorical_array
# 导入重构后的模型
from src.model.DNN import DNNModel
from src.model.LogisticRegression import LogisticModel
from src.model.Xgboost import XGBoostModel

def predict_new_patients(new_data, base_models, meta_model):
    """
    使用训练好的堆叠模型对新病人数据进行预测。
    """
    base_outputs = []
    for model in base_models:
        prob = model.predict_proba(new_data)[:, 1]
        base_outputs.append(prob.reshape(-1, 1))

    meta_input = np.hstack(base_outputs)
    final_proba = meta_model.predict_proba(meta_input)[:, 1]
    final_prediction = (final_proba >= 0.5).astype(int)

    return final_prediction, final_proba

def get_oof_predictions(model, X, y, X_val, n_splits=5):
    """
    为一个基模型生成训练集的 OOF（Out-of-Fold）和验证集的预测概率。
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_train = np.zeros(X.shape[0])
    oof_val_pred = np.zeros(X_val.shape[0])

    for train_idx, valid_idx in kf.split(X):
        instance = clone(model)
        X_train_fold = X.iloc[train_idx]
        y_train_fold = y.iloc[train_idx]
        X_valid_fold = X.iloc[valid_idx]

        instance.fit(X_train_fold, y_train_fold)
        oof_train[valid_idx] = instance.predict_proba(X_valid_fold)[:, 1]
        oof_val_pred += instance.predict_proba(X_val)[:, 1] / n_splits

    return oof_train.reshape(-1, 1), oof_val_pred.reshape(-1, 1)

def cross_validate_stacking_ensemble(X, y, n_splits=5):
    """
    对堆叠模型进行外层交叉验证，统计各折表现，并保存 AUC 最佳的模型。
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accuracy_list = []
    auc_list = []

    all_y_true = []
    all_y_pred = []
    all_y_prob = []

    best_auc = 0
    best_models = None
    best_meta_model = None

    fold = 1
    for train_index, val_index in skf.split(X, y):
        print(f"\n========== 外层第 {fold} 折 ==========")
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        # 构建并训练每个基模型
        base_models = []
        for model_class in [DNNModel, LogisticModel, XGBoostModel]:
            model = model_class()
            model.fit(X_train, y_train)
            base_models.append(model)

        meta_train_features = []
        meta_val_features = []

        # 获取每个基模型的 OOF 特征和验证集预测
        for model in base_models:
            oof_train, oof_val = get_oof_predictions(model, X_train, y_train, X_val, n_splits=5)
            meta_train_features.append(oof_train)
            meta_val_features.append(oof_val)

        # 构造用于 meta 层训练的特征
        X_meta_train = np.hstack(meta_train_features)
        X_meta_val = np.hstack(meta_val_features)

        # 训练 meta 模型（逻辑回归）
        meta_model = LogisticRegression()
        meta_model.fit(X_meta_train, y_train)
        y_val_pred = meta_model.predict(X_meta_val)
        y_val_prob = meta_model.predict_proba(X_meta_val)[:, 1]

        # 验证集评估
        acc = accuracy_score(y_val, y_val_pred)
        auc = roc_auc_score(y_val, y_val_prob)
        print(f"准确率: {acc:.4f}, AUC: {auc:.4f}")
        print("分类报告:")
        print(classification_report(y_val, y_val_pred))

        # 收集每一折的结果
        accuracy_list.append(acc)
        auc_list.append(auc)
        all_y_true.extend(y_val)
        all_y_pred.extend(y_val_pred)
        all_y_prob.extend(y_val_prob)

        # 记录当前最优模型
        if auc > best_auc:
            best_auc = auc
            best_models = base_models
            best_meta_model = meta_model

        fold += 1

    # 汇总整体表现
    print("\n========== 堆叠模型交叉验证汇总 ==========")
    print(f"平均准确率: {np.mean(accuracy_list):.4f} ± {np.std(accuracy_list):.4f}")
    print(f"平均 AUC: {np.mean(auc_list):.4f} ± {np.std(auc_list):.4f}")

    # 保存可视化图
    roc_path = EN_VIS_DIR / "ensemble_mean_roc.png"
    cm_path = EN_VIS_DIR / "ensemble_mean_cm.png"
    plot_roc_curve(all_y_true, all_y_prob, roc_path)
    plot_confusion_matrix(all_y_true, all_y_pred, cm_path)
    print(f"\nROC 曲线保存至: {roc_path}")
    print(f"混淆矩阵保存至: {cm_path}")

    # 保存最佳堆叠模型（使用 cloudpickle 以确保兼容性）
    MODEL_DIR.mkdir(exist_ok=True, parents=True)
    with open(MODEL_DIR / "base_models.pkl", "wb") as f:
        cloudpickle.dump(best_models, f)
    with open(MODEL_DIR / "meta_model.pkl", "wb") as f:
        cloudpickle.dump(best_meta_model, f)
    print(f"\n 已保存 AUC 最优折的堆叠模型（AUC={best_auc:.4f}）")

if __name__ == '__main__':
    # 读取训练数据
    df = pd.read_csv(DATA_FILE)
    X = df[final_numerical_arr + categorical_array]
    y = df["hospital_death"]

    # 执行 5 折交叉验证
    cross_validate_stacking_ensemble(X, y, n_splits=5)