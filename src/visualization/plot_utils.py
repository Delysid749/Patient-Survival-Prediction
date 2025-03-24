import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, confusion_matrix, roc_auc_score
import pandas as pd

"""
绘制ROC曲线并计算AUC值

参数：
- y_true：真实标签（1维数组或列表）
- y_score：预测概率值（1维数组或列表）
- save_path：保存图像的文件路径

输出：
- 保存一张ROC曲线图，包含AUC值
"""
def plot_roc_curve(y_true, y_score, save_path):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc_score = roc_auc_score(y_true, y_score)

    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')  # 对角线参考线
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve with AUC")
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

"""
绘制混淆矩阵（Confusion Matrix）

参数：
- y_true：真实标签（1维数组或列表）
- y_pred：预测标签（1维数组或列表）
- save_path：保存图像的文件路径

输出：
- 保存一张混淆矩阵热力图，展示分类模型的预测表现
"""
def plot_confusion_matrix(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

"""
绘制特征重要性条形图（适用于SelectKBest评分）

参数：
- features：特征名称列表
- scores：每个特征对应的评分（如SelectKBest打分）
- save_path：图像保存路径
- top_k：可选，仅显示前 top_k 个特征（按评分降序）

输出：
- 保存一张条形图，展示特征的重要性评分
"""
def plot_feature_importance(features, scores, save_path, top_k=None):
    # 构建 DataFrame
    feature_scores = pd.DataFrame({
        'Feature': features,
        'Score': scores
    })

    # 按照分数从高到低排序
    feature_scores = feature_scores.sort_values(by='Score', ascending=False)

    # 只显示前 top_k 个特征（如果指定）
    if top_k is not None:
        feature_scores = feature_scores.head(top_k)

    # 绘制条形图
    plt.figure(figsize=(10, 0.5 * len(feature_scores)))  # 动态调整图像高度
    sns.barplot(
        x='Score',
        y='Feature',
        hue='Feature',
        data=feature_scores,
        palette='crest',
        legend=False
    )
    plt.title(f"Feature Importance (SelectKBest Scores) TOP{top_k}")
    plt.xlabel("Score")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

"""
绘制训练过程中的损失函数变化曲线（适用于Keras模型）

参数：
- history：Keras训练过程的 history 对象（history.history 应包含 'loss' 和 'val_loss'）
- save_path：图像保存路径

输出：
- 保存一张曲线图，展示训练集与验证集的损失变化趋势
"""
def plot_loss_curve(history, save_path):
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(train_loss))

    plt.figure()
    plt.plot(epochs, train_loss, label='Train Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.title("Loss Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
