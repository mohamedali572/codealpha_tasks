import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(Advertising):
    # Load dataset
    df = pd.read_csv(Advertising)
    
    # Drop missing values if any
    df.dropna(inplace=True)
    
    # Separate features and target correctly
    X = df[['TV', 'Radio', 'Newspaper']]  # Explicitly select only 3 features
    y = df["Sales"]

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test, scaler

