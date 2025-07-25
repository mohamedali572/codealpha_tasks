import tkinter as tk
from tkinter import messagebox
from data_processing import load_data, split_data
from model import train_model, evaluate_model, predict_species
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns


def plot_confusion_matrix_in_gui(cm, encoder, parent_frame):
    
    for widget in parent_frame.winfo_children():
        widget.destroy()

    
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=encoder.classes_,
                yticklabels=encoder.classes_, ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')

    
    canvas = FigureCanvasTkAgg(fig, master=parent_frame)
    canvas.draw()
    canvas.get_tk_widget().pack()

def run_model():
    try:
        
        filepath = "Iris.csv"  
        X, y, encoder, data = load_data(filepath)
        X_train, X_test, y_train, y_test = split_data(X, y)

        
        model = train_model(X_train, y_train)
        accuracy, report, cm = evaluate_model(model, X_test, y_test, encoder)
        
        lbl_accuracy.config(text=f"Accuracy: {accuracy:.2f}")
        txt_report.delete("1.0", tk.END)
        txt_report.insert(tk.END, report)

        
        plot_confusion_matrix_in_gui(cm, encoder, frame_plot)

        
        app_data['model'] = model
        app_data['encoder'] = encoder

    except Exception as e:
        messagebox.showerror("Error", str(e))

def predict_flower():
    try:
        model = app_data.get('model')
        encoder = app_data.get('encoder')
        if model is None or encoder is None:
            messagebox.showwarning("Warning")
            return

        
        values = [
            float(entry_sepal_length.get()),
            float(entry_sepal_width.get()),
            float(entry_petal_length.get()),
            float(entry_petal_width.get())
        ]

        prediction = predict_species(model, np.array([values]), encoder)
        messagebox.showinfo("Prediction", f"Predicted Species: {prediction[0]}")

    except ValueError:
        messagebox.showerror("Error")


app_data = {}

root = tk.Tk()
root.title("Iris Classification GUI")
root.geometry("600x700")


btn_run = tk.Button(root, text="Run Model", command=run_model)
btn_run.pack(pady=10)

# Accuracy
lbl_accuracy = tk.Label(root, text="Accuracy: -", font=("Arial", 12))
lbl_accuracy.pack()

# Classification Report
txt_report = tk.Text(root, height=10, width=60)
txt_report.pack(pady=10)


frame_plot = tk.Frame(root)
frame_plot.pack(pady=10)


frame_inputs = tk.Frame(root)
frame_inputs.pack(pady=10)

tk.Label(frame_inputs, text="Sepal Length").grid(row=0, column=0)
entry_sepal_length = tk.Entry(frame_inputs)
entry_sepal_length.grid(row=0, column=1)

tk.Label(frame_inputs, text="Sepal Width").grid(row=1, column=0)
entry_sepal_width = tk.Entry(frame_inputs)
entry_sepal_width.grid(row=1, column=1)

tk.Label(frame_inputs, text="Petal Length").grid(row=2, column=0)
entry_petal_length = tk.Entry(frame_inputs)
entry_petal_length.grid(row=2, column=1)

tk.Label(frame_inputs, text="Petal Width").grid(row=3, column=0)
entry_petal_width = tk.Entry(frame_inputs)
entry_petal_width.grid(row=3, column=1)

# Predict
btn_predict = tk.Button(root, text="Predict Flower", command=predict_flower)
btn_predict.pack(pady=10)

root.mainloop()
