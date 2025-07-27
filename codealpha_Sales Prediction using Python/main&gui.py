import tkinter as tk
from tkinter import filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd

# Import project modules
from data_preprocessing import load_and_preprocess_data
from model_training import train_and_save_model
from prediction import predict_sales
from visualization import plot_correlation, plot_feature_importance

# Global variables
trained_models = {}      # Holds trained models and metrics
data_scaler = None       # Holds fitted scaler

# ----------------- Helper Functions -----------------

def select_file():
    """Allow user to select CSV file and show path in entry."""
    file_path = filedialog.askopenfilename(
        title="Select CSV file",
        filetypes=[("CSV Files", "*.csv")]
    )
    if file_path:
        file_entry.delete(0, tk.END)
        file_entry.insert(0, file_path)

def run_pipeline():
    """
    Load data, preprocess, train models, display metrics and plots.
    """
    file_path = file_entry.get()
    if not file_path:
        messagebox.showwarning("No file", "Please select a CSV file first.")
        return

    try:
        # Load and preprocess data
        X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data(file_path)

        # Train models
        global trained_models
        trained_models = train_and_save_model(X_train, y_train, X_test, y_test, save=False)

        # Display metrics in GUI
        metrics_text = "\n".join([
            f"{name} → R²: {metrics['r2']:.2f}, RMSE: {metrics['rmse']:.2f}"
            for name, metrics in trained_models.items()
        ])
        metrics_label.config(text=metrics_text)

        # Show plots
        raw_df = pd.read_csv(file_path)
        corr_fig = plot_correlation(raw_df)
        feat_fig = plot_feature_importance(trained_models["Random Forest"]["model"], ['TV', 'Radio', 'Newspaper'])
        display_plots(corr_fig, feat_fig)

        # Save scaler globally
        global data_scaler
        data_scaler = scaler

        messagebox.showinfo("Training Complete", "Models trained successfully!")

    except Exception as e:
        messagebox.showerror("Error", str(e))

def predict_new():
    """
    Predict sales using selected model and user inputs.
    """
    if not trained_models or data_scaler is None:
        messagebox.showwarning("No model", "Please train models first.")
        return

    try:
        # Get user input
        tv = float(tv_entry.get())
        radio = float(radio_entry.get())
        newspaper = float(newspaper_entry.get())
        new_data = [tv, radio, newspaper]

        # Get selected model
        selected_name = selected_model_name.get()
        model = trained_models[selected_name]["model"]

        # Predict
        prediction = predict_sales(new_data, data_scaler, model)
        result_label.config(text=f"Predicted Sales: {prediction:.2f}")

    except ValueError:
        messagebox.showwarning("Invalid Input", "Please enter valid numbers for all fields.")
    except Exception as e:
        messagebox.showerror("Error", str(e))

def display_plots(corr_fig, feat_fig):
    """
    Display correlation and feature importance plots side by side.
    """
    for widget in plot_frame.winfo_children():
        widget.destroy()

    # Correlation plot
    corr_canvas = FigureCanvasTkAgg(corr_fig, master=plot_frame)
    corr_canvas.draw()
    corr_canvas.get_tk_widget().grid(row=0, column=0, padx=10)

    # Feature importance plot
    feat_canvas = FigureCanvasTkAgg(feat_fig, master=plot_frame)
    feat_canvas.draw()
    feat_canvas.get_tk_widget().grid(row=0, column=1, padx=10)

# ----------------- GUI Layout -----------------

root = tk.Tk()
root.title("Sales Prediction with Machine Learning")
root.geometry("1000x800")
root.configure(bg="black")

# Selected model variable
selected_model_name = tk.StringVar(value="Linear Regression")

# File selection
file_frame = tk.Frame(root, bg="black")
file_frame.pack(pady=10)

file_entry = tk.Entry(file_frame, width=50, font=("Arial", 12))
file_entry.pack(side=tk.LEFT, padx=5)

browse_button = tk.Button(file_frame, text="Browse CSV", command=select_file,
                          bg="#4CAF50", fg="white", font=("Arial", 12, "bold"))
browse_button.pack(side=tk.LEFT)

# Train button
train_button = tk.Button(root, text="Train Models", command=run_pipeline,
                         bg="#2196F3", fg="white", font=("Arial", 14, "bold"), padx=10, pady=5)
train_button.pack(pady=10)

# Metrics display
metrics_label = tk.Label(root, text="Metrics will appear here after training.",
                         bg="black", fg="white", font=("Arial", 12))
metrics_label.pack(pady=10)

# Model selection dropdown
model_dropdown = tk.OptionMenu(root, selected_model_name, "Linear Regression", "Random Forest")
model_dropdown.config(font=("Arial", 12), bg="#FF9800", fg="white")
model_dropdown.pack(pady=5)

# Input frame
input_frame = tk.Frame(root, bg="black")
input_frame.pack(pady=20)

tk.Label(input_frame, text="TV:", bg="black", fg="white", font=("Arial", 12)).grid(row=0, column=0, padx=5, pady=5)
tv_entry = tk.Entry(input_frame, width=10, font=("Arial", 12))
tv_entry.grid(row=0, column=1, padx=5)

tk.Label(input_frame, text="Radio:", bg="black", fg="white", font=("Arial", 12)).grid(row=0, column=2, padx=5, pady=5)
radio_entry = tk.Entry(input_frame, width=10, font=("Arial", 12))
radio_entry.grid(row=0, column=3, padx=5)

tk.Label(input_frame, text="Newspaper:", bg="black", fg="white", font=("Arial", 12)).grid(row=0, column=4, padx=5, pady=5)
newspaper_entry = tk.Entry(input_frame, width=10, font=("Arial", 12))
newspaper_entry.grid(row=0, column=5, padx=5)

# Predict button
predict_button = tk.Button(root, text="Predict Sales", command=predict_new,
                           bg="#FF9800", fg="white", font=("Arial", 14, "bold"), padx=10, pady=5)
predict_button.pack(pady=10)

# Prediction result
result_label = tk.Label(root, text="Predicted Sales: -", bg="black", fg="white", font=("Arial", 16, "bold"))
result_label.pack(pady=10)

# Plot frame
plot_frame = tk.Frame(root, bg="black")
plot_frame.pack(pady=20)

root.mainloop()



