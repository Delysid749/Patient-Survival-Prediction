import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score
from src.common.variable_array import final_numerical_arr, categorical_array
from src.config.path_config import DATA_FILE, LOG_DIR, XG_VIS_DIR
from sklearn.feature_selection import SelectKBest, f_classif, chi2
from src.visualization.plot_utils import (
    plot_roc_curve,
    plot_confusion_matrix,
    plot_feature_importance
)


# 定义XGBoost模型训练与验证函数
def XGBoostModel(df):
    numeric_features = final_numerical_arr
    categorical_features = categorical_array

    # 特征选择器
    num_selector = SelectKBest(score_func=f_classif, k='all')
    cat_selector = SelectKBest(score_func=chi2, k='all')

    # 数值型特征预处理
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('selector', num_selector)
    ])

    # 类别型特征预处理
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore')),
        ('selector', cat_selector)
    ])

    # 总预处理器
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features),
        ])

    # 定义模型流水线
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model',
         XGBClassifier(eval_metric='logloss', n_estimators=100, max_depth=5, random_state=42))
    ])

    # 拆分数据
    X = df.drop("hospital_death", axis=1)
    y = df["hospital_death"]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # 模型训练
    model_pipeline.fit(X_train, y_train)

    # 预测与评估
    y_pred = model_pipeline.predict(X_val)
    y_pred_prob = model_pipeline.predict_proba(X_val)[:, 1]

    print("Model: XGBoostClassifier")
    print("Accuracy:", model_pipeline.score(X_val, y_val))
    print(classification_report(y_val, y_pred))
    print("AUC-ROC:", roc_auc_score(y_val, y_pred_prob))

    # 5折交叉验证
    cv_auc_scores = cross_val_score(model_pipeline, X, y, cv=5, scoring='roc_auc')
    print("5-Fold Cross-Validated AUC Scores:", cv_auc_scores)
    print("Mean AUC from CV:", cv_auc_scores.mean())

    # 可视化输出
    plot_roc_curve(y_val, y_pred_prob, XG_VIS_DIR / "roc_curve.png")
    print("ROC 曲线图已保存至：", XG_VIS_DIR / "roc_curve.png")

    plot_confusion_matrix(y_val, y_pred, XG_VIS_DIR / "confusion_matrix.png")
    print("混淆矩阵图已保存至：", XG_VIS_DIR / "confusion_matrix.png")

    # 特征重要性可视化
    selector_num = model_pipeline.named_steps['preprocessor'].named_transformers_['num'].named_steps['selector']
    scores_num = selector_num.scores_
    plot_feature_importance(numeric_features, scores_num, XG_VIS_DIR / "num_feature_importance.png", 10)
    print("数值特征重要性图已保存至：", XG_VIS_DIR / "num_feature_importance.png")

    selector_cat = model_pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['selector']
    encoder = model_pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
    cat_feature_names = encoder.get_feature_names_out(categorical_features)
    scores_cat = selector_cat.scores_
    plot_feature_importance(cat_feature_names, scores_cat, XG_VIS_DIR / "cat_feature_importance.png", 10)
    print("分类特征重要性图已保存至：", XG_VIS_DIR / "cat_feature_importance.png")


# 主程序入口
if __name__ == '__main__':
    df = pd.read_csv(DATA_FILE)
    XGBoostModel(df)
