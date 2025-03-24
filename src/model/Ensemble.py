import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from src.config.path_config import DATA_FILE, LOG_DIR, EN_VIS_DIR
from src.visualization.plot_utils import plot_roc_curve, plot_confusion_matrix
from pathlib import Path

# 导入重构后的模型
from DNN import DNNModel
from LogisticRegression import LogisticModel
from Xgboost import XGBoostModel

def get_oof_predictions(model, X, y, X_val, n_splits=5):
    """
    为一个基模型生成训练集的 OOF 和验证集的预测概率。
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
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accuracy_list = []
    auc_list = []

    all_y_true = []
    all_y_pred = []
    all_y_prob = []

    fold = 1
    for train_index, val_index in skf.split(X, y):
        print(f"\n========== 外层第 {fold} 折 ==========")
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        base_models = [DNNModel(), LogisticModel(), XGBoostModel()]
        meta_train_features = []
        meta_val_features = []

        for model in base_models:
            oof_train, oof_val = get_oof_predictions(model, X_train, y_train, X_val, n_splits=5)
            meta_train_features.append(oof_train)
            meta_val_features.append(oof_val)

        X_meta_train = np.hstack(meta_train_features)
        X_meta_val = np.hstack(meta_val_features)

        meta_model = LogisticRegression()
        meta_model.fit(X_meta_train, y_train)
        y_val_pred = meta_model.predict(X_meta_val)
        y_val_prob = meta_model.predict_proba(X_meta_val)[:, 1]

        acc = accuracy_score(y_val, y_val_pred)
        auc = roc_auc_score(y_val, y_val_prob)
        print(f"准确率: {acc:.4f}, AUC: {auc:.4f}")
        print("分类报告:")
        print(classification_report(y_val, y_val_pred))

        accuracy_list.append(acc)
        auc_list.append(auc)

        all_y_true.extend(y_val)
        all_y_pred.extend(y_val_pred)
        all_y_prob.extend(y_val_prob)

        fold += 1

    print("\n========== 堆叠模型交叉验证汇总 ==========")
    print(f"平均准确率: {np.mean(accuracy_list):.4f} ± {np.std(accuracy_list):.4f}")
    print(f"平均 AUC: {np.mean(auc_list):.4f} ± {np.std(auc_list):.4f}")

    # 绘图输出到 EN_VIS_DIR
    roc_path = EN_VIS_DIR / "ensemble_mean_roc.png"
    cm_path = EN_VIS_DIR / "ensemble_mean_cm.png"

    plot_roc_curve(all_y_true, all_y_prob, roc_path)
    plot_confusion_matrix(all_y_true, all_y_pred, cm_path)

    print(f"\nROC 曲线保存至: {roc_path}")
    print(f"混淆矩阵保存至: {cm_path}")

if __name__ == '__main__':
    df = pd.read_csv(DATA_FILE)
    X = df.drop("hospital_death", axis=1)
    y = df["hospital_death"]

    cross_validate_stacking_ensemble(X, y, n_splits=5)