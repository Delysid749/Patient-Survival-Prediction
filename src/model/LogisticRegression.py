# LogisticRegression.py
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif, chi2

from src.common.variable_array import final_numerical_arr, categorical_array
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
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
    df = pd.read_csv(DATA_FILE)
    X = df.drop("hospital_death", axis=1)
    y = df["hospital_death"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 2. 初始化并训练模型
    model = LogisticModel()
    print("开始训练逻辑回归模型...")
    model.fit(X_train, y_train)

    # 3. 模型预测
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # 4. 评估输出
    print("准确率:", accuracy_score(y_test, y_pred))
    print("分类报告:")
    print(classification_report(y_test, y_pred))

    # 5. 可视化目录与图像路径
    os.makedirs(LR_VIS_DIR, exist_ok=True)
    roc_path = os.path.join(LR_VIS_DIR, "lr_roc_curve.png")
    cm_path = os.path.join(LR_VIS_DIR, "lr_confusion_matrix.png")
    fi_path = os.path.join(LR_VIS_DIR, "lr_feature_importance.png")

    # 6. 绘制 ROC 曲线 & 混淆矩阵
    plot_roc_curve(y_test, y_prob, roc_path)
    print(f"ROC 曲线已保存至：{roc_path}")

    plot_confusion_matrix(y_test, y_pred, cm_path)
    print(f"混淆矩阵图已保存至：{cm_path}")

    # 7. 提取特征打分（包含 OneHot 后特征名）
    print("提取特征重要性...")
    selector_num = model.pipeline.named_steps['preprocessor'].named_transformers_['num'].named_steps['selector']
    selector_cat = model.pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['selector']
    encoder_cat = model.pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']

    num_features = final_numerical_arr
    num_scores = selector_num.scores_

    cat_features = encoder_cat.get_feature_names_out(categorical_array).tolist()
    cat_scores = selector_cat.scores_

    all_features = num_features + cat_features
    all_scores = np.concatenate([num_scores, cat_scores])

    plot_feature_importance(all_features, all_scores, fi_path, top_k=20)
    print(f"特征重要性图已保存至：{fi_path}")