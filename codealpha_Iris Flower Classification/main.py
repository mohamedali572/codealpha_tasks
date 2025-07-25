from data_processing import load_data, split_data
from model import train_model, evaluate_model, predict_species
from visualization import plot_pairplot, plot_confusion_matrix
import numpy as np


X, y, encoder, data = load_data(
    r"C:\Users\mS\Documents\CodeAlpha\Data Science\TASK 1 Iris Flower Classification\Iris.csv"
)   

X_train, X_test, y_train, y_test = split_data(X, y)

# 3. تدريب النموذج
model = train_model(X_train, y_train)

# 4. التقييم
accuracy, report, cm = evaluate_model(model, X_test, y_test, encoder)
print("Accuracy:", accuracy)
print(report)

# 5. رسم البيانات
plot_pairplot(data)
plot_confusion_matrix(cm, encoder)

# 6. تجربة النموذج على بيانات جديدة
new_data = np.array([[5.1, 3.5, 1.4, 0.2]])
print("Predicted Species:", predict_species(model, new_data, encoder))