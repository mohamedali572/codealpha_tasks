import matplotlib.pyplot as plt
import seaborn as sns

def plot_correlation(df):
    # Plot correlation heatmap
    plt.figure(figsize=(8,6))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title("Correlation Matrix")
    plt.show()

def plot_feature_importance(model, features):
    # Plot feature importance for models that support it (e.g., RandomForest)
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        plt.bar(features, importance)
        plt.title("Feature Importance")
        plt.show()
