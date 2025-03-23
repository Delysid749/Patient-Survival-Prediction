# Xgboost.py
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
from sklearn.feature_selection import SelectKBest, f_classif, chi2

from src.common.variable_array import final_numerical_arr, categorical_array
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from src.config.path_config import DATA_FILE, XG_VIS_DIR
from src.visualization.plot_utils import (
    plot_roc_curve,
    plot_confusion_matrix,
    plot_feature_importance
)


class XGBoostModel(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.pipeline = None

    def fit(self, X, y):
        # 数值型预处理
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('selector', SelectKBest(score_func=f_classif, k='all'))
        ])
        # 类别型预处理
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore')),
            ('selector', SelectKBest(score_func=chi2, k='all'))
        ])
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, final_numerical_arr),
                ('cat', categorical_transformer, categorical_array)
            ]
        )
        self.pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', XGBClassifier(eval_metric='logloss', n_estimators=100, max_depth=5, random_state=42))
        ])
        self.pipeline.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)

    def predict(self, X):
        return self.pipeline.predict(X)


if __name__ == "__main__":
    # 1. 加载数据
    df = pd.read_csv(DATA_FILE)
    X = df.drop("hospital_death", axis=1)
    y = df["hospital_death"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 2. 初始化并训练模型
    model = XGBoostModel()
    print("开始训练 XGBoost 模型...")
    model.fit(X_train, y_train)

    # 3. 模型预测
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # 4. 评估
    print("准确率:", accuracy_score(y_test, y_pred))
    print("分类报告:")
    print(classification_report(y_test, y_pred))

    # 5. 可视化目录
    os.makedirs(XG_VIS_DIR, exist_ok=True)
    roc_path = os.path.join(XG_VIS_DIR, "xgb_roc_curve.png")
    cm_path = os.path.join(XG_VIS_DIR, "xgb_confusion_matrix.png")
    fi_path = os.path.join(XG_VIS_DIR, "xgb_feature_importance.png")

    # 6. 绘制 ROC 曲线
    plot_roc_curve(y_test, y_prob, roc_path)
    print(f"ROC 曲线已保存至：{roc_path}")

    # 7. 绘制混淆矩阵
    plot_confusion_matrix(y_test, y_pred, cm_path)
    print(f"混淆矩阵图已保存至：{cm_path}")

    # 8. 提取 SelectKBest 分数进行特征可视化（重点）
    print("提取特征重要性...")
    # 提取预处理器中的 SelectKBest 分数
    # 获取特征分数
    selector_num = model.pipeline.named_steps['preprocessor'].named_transformers_['num'].named_steps['selector']
    selector_cat = model.pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['selector']
    encoder_cat = model.pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']

    # 获取数值型特征名与打分
    num_features = final_numerical_arr
    num_scores = selector_num.scores_

    # 获取 OneHot 展开后的类别型特征名与打分
    cat_features = encoder_cat.get_feature_names_out(categorical_array).tolist()
    cat_scores = selector_cat.scores_

    # 拼接总特征与分数
    all_features = num_features + cat_features
    all_scores = np.concatenate([num_scores, cat_scores])

    plot_feature_importance(all_features, all_scores, fi_path, top_k=20)
    print(f"特征重要性图已保存至：{fi_path}")
