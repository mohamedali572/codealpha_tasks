import matplotlib.pyplot as plt
import seaborn as sns

def plot_correlation(df):
    """
    Create and return correlation matrix figure for displaying inside GUI.
    
    Parameters:
        df (DataFrame): Raw dataset
    
    Returns:
        fig (Figure): Matplotlib figure of correlation matrix
    """
    # Calculate correlation
    corr = df.corr()

    # Create figure
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    ax.set_title("Correlation Matrix")

    # Close figure to prevent external window popup
    plt.close(fig)
    return fig


def plot_feature_importance(model, feature_names):
    """
    Create and return feature importance figure for displaying inside GUI.

    Parameters:
        model: Trained model (must have feature_importances_ attribute)
        feature_names (list): Names of the features used for training
    
    Returns:
        fig (Figure): Matplotlib figure of feature importances
    """
    # Check if model supports feature_importances_
    if not hasattr(model, "feature_importances_"):
        # If not supported, return empty figure
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.text(0.5, 0.5, "Feature importance not available",
                ha="center", va="center", fontsize=12)
        ax.axis("off")
        plt.close(fig)
        return fig

    # Get importances
    importances = model.feature_importances_

    # Create bar plot
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(feature_names, importances, color='orange')
    ax.set_title("Feature Importance")
    ax.set_ylabel("Importance")
    ax.set_xlabel("Features")

    # Close figure to prevent popup
    plt.close(fig)
    return fig

