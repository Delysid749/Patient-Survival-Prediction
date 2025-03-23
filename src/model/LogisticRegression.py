# 导入所需的库
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression, LinearRegression
from src.common.variable_array import final_numerical_arr, categorical_array
from src.config.path_config import DATA_FILE, LOG_DIR, LR_VIS_DIR
from sklearn.metrics import classification_report, roc_auc_score, roc_curve,confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif,chi2
import matplotlib.pyplot as plt
import seaborn as sns

from src.visualization.plot_utils import (
    plot_roc_curve,
    plot_confusion_matrix,
    plot_feature_importance
)


# 定义逻辑回归模型训练与验证函数
def LogisticRegressionModel(df):
    numeric_features = final_numerical_arr  # 数值特征
    categorical_features = categorical_array  # 类别特征
    # 数值型特征选择器
    num_selector = SelectKBest(score_func=f_classif, k='all')
    # 类别型特征选择器
    cat_selector = SelectKBest(score_func=chi2, k='all')

    # 数值型特征预处理管道：缺失值填充（中位数）+ 标准化
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),  # 缺失值填充为中位数
        ('scaler', StandardScaler()),  # 标准化处理
        ('selector', num_selector)
    ])

    # 类别型特征预处理管道：缺失值填充为'missing' + 独热编码
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),  # 填充空值为'missing'
        ('onehot', OneHotEncoder(handle_unknown='ignore')),  # 独热编码
        ('selector', cat_selector)
    ])

    # 总的预处理器：将数值与类别特征的预处理组合起来
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),  # 数值特征处理方式
            ('cat', categorical_transformer, categorical_features),  # 类别特征处理方式
        ])

    # 定义完整模型管道：预处理器 + 逻辑回归分类器
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),  # 第一步：预处理
        ('model', LogisticRegression(max_iter=1000))  # 第二步：训练逻辑回归模型
    ])

    # 拆分特征与标签
    X = df.drop(["hospital_death"], axis=1)  # 特征（删除目标变量列）
    y = df['hospital_death']  # 目标变量（住院是否死亡）

    # 拆分训练集和验证集（80%训练，20%验证）
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=42)

    # 模型拟合训练数据
    model_pipeline.fit(X_train, y_train)

    # 在验证集上打分（输出准确率）
    accuracy = model_pipeline.score(X_val, y_val)

    # 预测验证集的结果
    y_pred = model_pipeline.predict(X_val)
    y_pred_prob = model_pipeline.predict_proba(X_val)[:, 1]

    # 打印模型名称与准确率
    print("Model: LogisticRegression")
    print("Accuracy:", accuracy)
    print(classification_report(y_val, y_pred))
    print("AUC-ROC:", roc_auc_score(y_val, y_pred_prob))

    # 添加 5 折交叉验证
    from sklearn.model_selection import cross_val_score
    cv_auc_scores = cross_val_score(model_pipeline, X, y, cv=5, scoring='roc_auc')
    print("5-Fold Cross-Validated AUC Scores:", cv_auc_scores)
    print("Mean AUC from CV:", cv_auc_scores.mean())

    plot_roc_curve(y_val, y_pred_prob, LR_VIS_DIR / "roc_curve.png")
    print("ROC 曲线图已保存至：", LR_VIS_DIR / "roc_curve.png")

    plot_confusion_matrix(y_val, y_pred, LR_VIS_DIR / "confusion_matrix.png")
    print("混淆矩阵图已保存至：", LR_VIS_DIR / "confusion_matrix.png")

    # 数值型特征重要性可视化
    selector_num = model_pipeline.named_steps['preprocessor'] \
        .named_transformers_['num'] \
        .named_steps['selector']
    scores_num = selector_num.scores_
    plot_feature_importance(numeric_features, scores_num, LR_VIS_DIR / "num_feature_importance.png",10)
    print("数值特征重要性图已保存至：", LR_VIS_DIR / "num_feature_importance.png")

    # 分类特征重要性可视化
    selector_cat = model_pipeline.named_steps['preprocessor'] \
        .named_transformers_['cat'] \
        .named_steps['selector']
    encoder = model_pipeline.named_steps['preprocessor'] \
        .named_transformers_['cat'] \
        .named_steps['onehot']
    cat_feature_names = encoder.get_feature_names_out(categorical_features)
    scores_cat = selector_cat.scores_
    plot_feature_importance(cat_feature_names, scores_cat, LR_VIS_DIR / "cat_feature_importance.png",10)
    print("分类特征重要性图已保存至：", LR_VIS_DIR / "cat_feature_importance.png")

if __name__ == '__main__':
    df = pd.read_csv(DATA_FILE)

    LogisticRegressionModel(df)
