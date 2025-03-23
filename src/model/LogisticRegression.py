# 导入所需的库
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression, LinearRegression
from src.common.variable_array import final_numerical_arr, categorical_array
from src.config.path_config import DATA_FILE, LOG_DIR
from sklearn.metrics import classification_report, roc_auc_score, roc_curve


# 定义逻辑回归模型训练与验证函数
def LogisticRegressionModel(df):
    numeric_features = final_numerical_arr  # 数值特征
    categorical_features = categorical_array  # 类别特征

    # 数值型特征预处理管道：缺失值填充（中位数）+ 标准化
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),   # 缺失值填充为中位数
        ('scaler', StandardScaler())                     # 标准化处理
    ])

    # 类别型特征预处理管道：缺失值填充为'missing' + 独热编码
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),  # 填充空值为'missing'
        ('onehot', OneHotEncoder(handle_unknown='ignore'))                      # 独热编码
    ])

    # 总的预处理器：将数值与类别特征的预处理组合起来
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),  # 数值特征处理方式
            ('cat', categorical_transformer, categorical_features),  # 类别特征处理方式
        ])

    # 定义完整模型管道：预处理器 + 逻辑回归分类器
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),         # 第一步：预处理
        ('model', LogisticRegression())         # 第二步：训练逻辑回归模型
    ])

    # 拆分特征与标签
    X = df.drop(["hospital_death"], axis=1)    # 特征（删除目标变量列）
    y = df['hospital_death']                   # 目标变量（住院是否死亡）

    # 拆分训练集和验证集（75%训练，25%验证）
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42)

    # 模型拟合训练数据
    model_pipeline.fit(X_train, y_train)

    # 在验证集上打分（输出准确率）
    accuracy = model_pipeline.score(X_val, y_val)

    # 预测验证集的结果
    y_pred = model_pipeline.predict(X_val)

    # 打印模型名称与准确率
    print("Model: LogisticRegression")
    print("Accuracy:", accuracy)

# 主程序入口
if __name__ == '__main__':
    df = pd.read_csv(DATA_FILE)

    LogisticRegressionModel(df)
