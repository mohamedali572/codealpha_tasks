import tkinter as tk
from tkinter import filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from src.data_preprocessing import load_and_preprocess
from src.model_training import train_model
from src.evaluate import evaluate_model

# Clear old plots
def clear_plots():
    for widget in plot_frame.winfo_children():
        widget.destroy()

# Main pipeline
def run_pipeline():
    file_path = filedialog.askopenfilename(
        title="Select CSV file",
        filetypes=[("CSV Files", "*.csv")]
    )

    if not file_path:
        messagebox.showwarning("No file", "Please select a CSV file.")
        return

    try:
        # 1. Load & preprocess
        X_train, X_test, y_train, y_test = load_and_preprocess(file_path)

        # 2. Train
        model = train_model(X_train, y_train)

        # 3. Evaluate (returns metrics + figures)
        metrics, figures = evaluate_model(model, X_test, y_test)

        mae, rmse, r2 = metrics
        result_label.config(
            text=f"MAE: {mae:.2f}\nRMSE: {rmse:.2f}\nR²: {r2:.2f}"
        )

        # Display plots in GUI
        clear_plots()
        for fig in figures:
            canvas = FigureCanvasTkAgg(fig, master=plot_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(side=tk.LEFT, padx=5, pady=5)

    except Exception as e:
        messagebox.showerror("Error", str(e))

# ------------------ GUI ------------------
root = tk.Tk()
root.title("Car Price Prediction")
root.geometry("1200x600")
root.configure(bg="black")

# Button
run_button = tk.Button(
    root,
    text="Select CSV & Run",
    command=run_pipeline,
    font=("Arial", 16, "bold"),
    bg="#4CAF50",
    fg="white",
    padx=20,
    pady=10
)
run_button.pack(pady=20)

# Result label
result_label = tk.Label(
    root,
    text="Results will appear here",
    font=("Arial", 16),
    bg="black",
    fg="white",
    justify="center"
)
result_label.pack(pady=10)

# Frame for plots
plot_frame = tk.Frame(root, bg="black")
plot_frame.pack(pady=20)

root.mainloop()

