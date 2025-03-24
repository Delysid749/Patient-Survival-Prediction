# LogisticRegression.py
from collections import defaultdict

import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif, chi2
from sklearn.model_selection import StratifiedKFold
from src.common.variable_array import final_numerical_arr, categorical_array
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report,roc_auc_score
from src.config.path_config import DATA_FILE, LR_VIS_DIR
from src.visualization.plot_utils import (
    plot_roc_curve,
    plot_confusion_matrix,
    plot_feature_importance
)
from src.common.variable_array import final_numerical_arr, categorical_array


class LogisticModel(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.pipeline = None

    def fit(self, X, y):
        # 数值型特征预处理
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('selector', SelectKBest(score_func=f_classif, k='all'))
        ])
        # 类别型特征预处理
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore')),
            ('selector', SelectKBest(score_func=chi2, k='all'))
        ])
        # 总预处理器
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, final_numerical_arr),
                ('cat', categorical_transformer, categorical_array)
            ]
        )
        # 构建包含预处理器与逻辑回归模型的流水线
        self.pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', LogisticRegression(max_iter=1000))
        ])
        self.pipeline.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)

    def predict(self, X):
        return self.pipeline.predict(X)


if __name__ == '__main__':
    # 1. 读取数据
    df = pd.read_csv(DATA_FILE)
    X = df.drop("hospital_death", axis=1)
    y = df["hospital_death"]

    # 2. 创建可视化输出目录
    os.makedirs(LR_VIS_DIR, exist_ok=True)

    # 3. 初始化模型 & 交叉验证器
    model = LogisticModel()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 4. 结果收集容器
    accuracy_list = []
    auc_list = []
    all_y_true = []
    all_y_pred = []
    all_y_prob = []

    feature_score_dict = defaultdict(list)  # 保存每一折的特征分数

    fold = 1
    for train_idx, test_idx in skf.split(X, y):
        print(f"\n========== 第 {fold} 折训练 ==========")
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # 训练模型
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # 评估指标
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        print(f"准确率：{acc:.4f}")
        print(f"AUC值：{auc:.4f}")
        print("分类报告：")
        print(classification_report(y_test, y_pred))

        # 汇总预测结果
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        all_y_prob.extend(y_prob)

        # 特征重要性收集
        selector_num = model.pipeline.named_steps['preprocessor'].named_transformers_['num'].named_steps['selector']
        selector_cat = model.pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['selector']
        encoder_cat = model.pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']

        num_features = final_numerical_arr
        num_scores = selector_num.scores_

        cat_features = encoder_cat.get_feature_names_out(categorical_array).tolist()
        cat_scores = selector_cat.scores_

        all_features = num_features + cat_features
        all_scores = np.concatenate([num_scores, cat_scores])

        for feat, score in zip(all_features, all_scores):
            feature_score_dict[feat].append(score)

        accuracy_list.append(acc)
        auc_list.append(auc)
        fold += 1

    # 5. 汇总评估
    print("\n========== 交叉验证汇总 ==========")
    print(f"平均准确率: {np.mean(accuracy_list):.4f} ± {np.std(accuracy_list):.4f}")
    print(f"平均 AUC: {np.mean(auc_list):.4f} ± {np.std(auc_list):.4f}")

    # 6. 可视化：ROC曲线 + 混淆矩阵 + 特征重要性（平均）

    roc_path = os.path.join(LR_VIS_DIR, "lr_mean_roc.png")
    cm_path = os.path.join(LR_VIS_DIR, "lr_mean_cm.png")
    fi_path = os.path.join(LR_VIS_DIR, "lr_mean_feature_importance.png")

    plot_roc_curve(all_y_true, all_y_prob, roc_path)
    plot_confusion_matrix(all_y_true, all_y_pred, cm_path)

    # 平均特征重要性
    avg_feature_scores = []
    for feat in all_features:
        avg_score = np.mean(feature_score_dict[feat])
        avg_feature_scores.append(avg_score)

    plot_feature_importance(all_features, avg_feature_scores, fi_path, top_k=20)

    print(f"\n已保存平均 ROC 曲线至：{roc_path}")
    print(f"已保存平均混淆矩阵至：{cm_path}")
    print(f"已保存平均特征重要性图至：{fi_path}")