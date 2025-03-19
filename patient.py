import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
from sklearn.metrics import (accuracy_score, roc_auc_score, 
                             confusion_matrix, classification_report,
                             RocCurveDisplay)
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

# 1. 数据加载
data = pd.read_csv('dataset.csv')  # 替换为你的数据路径

# 2. 数据预处理
# 删除不相关的ID列
data = data.drop(['encounter_id', 'patient_id', 'hospital_id', 'icu_id'], axis=1)

# 处理缺失值（示例处理，需根据实际数据调整）
# 数值列用中位数填充
numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())

# 分类列用 'missing' 填充
categorical_cols = data.select_dtypes(include=['object']).columns
data = data.dropna(subset=categorical_cols)
 

# 3. 特征工程
# 定义分类特征和数值特征
categorical_features = ['elective_surgery', 'ethnicity', 'gender', 
                       'icu_admit_source', 'icu_stay_type', 'icu_type',
                       'apache_3j_bodysystem', 'apache_2_bodysystem']
numeric_features = [col for col in data.columns 
                   if col not in categorical_features + ['hospital_death']]

# 4. 创建预处理管道
# 这里加入SimpleImputer来处理缺失值
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),  # 对数值列进行中位数填充
            ('scaler', StandardScaler())  # 标准化
        ]), numeric_features),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),  # 对分类列进行'missing'填充
            ('encoder', OneHotEncoder(handle_unknown='ignore'))  # 独热编码
        ]), categorical_features)
    ])

# 5. 划分数据集
X = data.drop('hospital_death', axis=1)
y = data['hospital_death']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# 6. 创建模型管道（去掉 class_weight 参数）
logreg_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(class_weight='balanced', max_iter=1000))
])

tree_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(class_weight='balanced', max_depth=5))
])

gb_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', GradientBoostingClassifier())  # 去掉 class_weight
])

xgb_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', xgb.XGBClassifier(
        device = "cuda",
        scale_pos_weight=1, 
        use_label_encoder=False, 
        eval_metric='logloss'))
])


# 7. 训练和评估函数
def train_evaluate_model(pipeline, model_name):
    # 训练模型
    pipeline.fit(X_train, y_train)
    
    # 预测
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    
    # 评估指标
    print(f"\n{model_name} 评估结果:")
    print("准确率:", accuracy_score(y_test, y_pred))
    print("AUC:", roc_auc_score(y_test, y_proba))
    print("\n分类报告:")
    print(classification_report(y_test, y_pred))
    
    # 绘制ROC曲线
    RocCurveDisplay.from_estimator(pipeline, X_test, y_test)
    plt.title(f'{model_name} ROC Curve')
    plt.show()
    
    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    print("混淆矩阵:")
    print(cm)

# 8. 训练和评估模型
train_evaluate_model(logreg_pipeline, "logisticsRegression")
train_evaluate_model(tree_pipeline, "decisionTree")
train_evaluate_model(gb_pipeline, "gradient")
train_evaluate_model(xgb_pipeline, "XGBoost")

# 9. 特征重要性分析（以决策树为例）
# 提取预处理后的特征名称
preprocessor.fit(X)
feature_names = (numeric_features + 
                list(preprocessor.named_transformers_['cat']
                     .get_feature_names_out(categorical_features)))

# 获取特征重要性
importances = tree_pipeline.named_steps['classifier'].feature_importances_
indices = np.argsort(importances)[-10:]  # 取最重要的10个特征

# 绘制特征重要性
plt.figure(figsize=(10, 6))
plt.title('top 10 features')
plt.barh(range(len(indices)), importances[indices], align='center')
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel('feature importance')
plt.show()



from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
from sklearn.model_selection import cross_val_score,cross_validate


# 1. 创建基础模型（第一层模型）
base_learners = [
    ('logreg', Pipeline(steps=[
        ('preprocessor', preprocessor),  # 使用同样的预处理管道
        ('classifier', LogisticRegression(class_weight='balanced', max_iter=1000))
    ])),
    ('tree', Pipeline(steps=[
        ('preprocessor', preprocessor),  # 使用同样的预处理管道
        ('classifier', DecisionTreeClassifier(class_weight='balanced', max_depth=5))
    ])),
    ('gb', Pipeline(steps=[
        ('preprocessor', preprocessor),  # 使用同样的预处理管道
        ('classifier', GradientBoostingClassifier())
    ])),
    ('xgb', Pipeline(steps=[
        ('preprocessor', preprocessor),  # 使用同样的预处理管道
        ('classifier', xgb.XGBClassifier(
            tree_method='gpu_hist',  # 使用GPU加速
            gpu_id=0,  # 选择GPU设备（如果有多个GPU可以选择其他编号）
            scale_pos_weight=1, 
            use_label_encoder=False, 
            eval_metric='logloss'))
    ]))
]

# 2. 创建元模型（第二层模型）
meta_model = LogisticRegression(class_weight='balanced', max_iter=1000)

# 3. 创建堆叠法模型
stacking_pipeline = StackingClassifier(
    estimators=base_learners, 
    final_estimator=meta_model
)

# 4. 使用交叉验证对堆叠法模型进行评估
cv_scores = cross_val_score(stacking_pipeline, X, y, cv=5, scoring='accuracy')
print(f"cross_validate_accuracy: {cv_scores}")
print(f"Average accuracy: {cv_scores.mean()}")


# 5. 使用 cross_validate 进行详细的交叉验证评估
cv_results = cross_validate(stacking_pipeline, X, y, cv=5, 
                            scoring=['accuracy', 'roc_auc'], return_train_score=False)
print(f"cross_validate_accuracy: {cv_results['test_accuracy']}")
print(f"cross_validate AUC: {cv_results['test_roc_auc']}")
print(f"Average accuracy: {cv_results['test_accuracy'].mean()}")
print(f"Average AUC: {cv_results['test_roc_auc'].mean()}")


# 绘制交叉验证准确率的箱型图
plt.boxplot(cv_scores, vert=False)
plt.title('cross_validate_accuracy')
plt.xlabel('accuracy')
plt.show()

# 绘制交叉验证 AUC 的箱型图
plt.boxplot(cv_results['test_roc_auc'], vert=False)
plt.title('cross_validate AUC')
plt.xlabel('AUC')
plt.show()
