import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, confusion_matrix,roc_auc_score
import pandas as pd

"""
定义AUC、ROC曲線、混淆矩阵的绘制函数
"""
def plot_roc_curve(y_true, y_score, save_path):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc_score = roc_auc_score(y_true, y_score)

    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve with AUC")
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

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

def plot_feature_importance(features, scores, save_path, top_k=None):
    # 构建 DataFrame
    feature_scores = pd.DataFrame({
        'Feature': features,
        'Score': scores
    })

    # 排序
    feature_scores = feature_scores.sort_values(by='Score', ascending=False)

    # 只显示前 top_k 个特征
    if top_k is not None:
        feature_scores = feature_scores.head(top_k)

    # 绘图
    plt.figure(figsize=(10, 0.5 * len(feature_scores)))  # 动态调整高度
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

def plot_loss_curve(history, save_path):
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(train_loss))

    plt.figure()
    plt.plot(epochs, train_loss, label='Train Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.title("Loss curve")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

