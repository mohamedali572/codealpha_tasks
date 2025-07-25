from src.data_preprocessing import load_and_preprocess
from src.model_training import train_model
from src.evaluate import evaluate_model

def main():
    # 1. Load & preprocess data
    X_train, X_test, y_train, y_test = load_and_preprocess(r"C:\Users\mS\Documents\CodeAlpha\Data Science\TASK 3 Car Price Prediction with Machine Learning\cardata.csv")

    # 2. Train model
    model = train_model(X_train, y_train)

    # 3. Evaluate model
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()
