from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import cross_val_score, cross_validate, train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

# 1. 数据加载
data = pd.read_csv('dataSet/dataset.csv')  # 替换为你的数据路径

# 2. 数据预处理
# 删除不相关的ID列
data = data.drop(['encounter_id', 'patient_id', 'hospital_id', 'icu_id'], axis=1)
data = data.drop(columns=['Unnamed: 83'], errors='ignore')

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


# 1. 创建基础模型（第一层模型）
base_learners = [
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
            # tree_method='gpu_hist',  # 使用GPU加速
            # gpu_id=0,  # 选择GPU设备（如果有多个GPU可以选择其他编号）
            scale_pos_weight=1,
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
X = data.drop('hospital_death', axis=1)
y = data['hospital_death']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# 4. 使用交叉验证对堆叠法模型进行评估
cv_scores = cross_val_score(stacking_pipeline, X, y, cv=5, scoring='accuracy')
print(f"cross_validate_accuracy: {cv_scores}")
print(f"Average accuracy: {cv_scores.mean()}")
# 绘制交叉验证准确率的箱型图
plt.boxplot(cv_scores, vert=False)
plt.title('cross_validate_accuracy')
plt.xlabel('accuracy')
plt.show()

# 5. 使用 cross_validate 进行详细的交叉验证评估
cv_results = cross_validate(stacking_pipeline, X, y, cv=5,
                            scoring=['accuracy', 'roc_auc'], return_train_score=False)
print(f"cross_validate_accuracy: {cv_results['test_accuracy']}")
print(f"cross_validate AUC: {cv_results['test_roc_auc']}")
print(f"Average accuracy: {cv_results['test_accuracy'].mean()}")
print(f"Average AUC: {cv_results['test_roc_auc'].mean()}")

# 绘制交叉验证 AUC 的箱型图
plt.boxplot(cv_results['test_roc_auc'], vert=False)
plt.title('cross_validate AUC')
plt.xlabel('AUC')
plt.show()