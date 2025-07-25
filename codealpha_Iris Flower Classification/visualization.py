import seaborn as sns
import matplotlib.pyplot as plt

def plot_pairplot(data):
    sns.pairplot(data, hue="Species", diag_kind="hist")
    plt.suptitle("Pairplot of Iris Features", y=1.02)
    plt.show()

def plot_confusion_matrix(cm, encoder):
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=encoder.classes_,
                yticklabels=encoder.classes_)
    plt.title("Confusion Matrix")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()