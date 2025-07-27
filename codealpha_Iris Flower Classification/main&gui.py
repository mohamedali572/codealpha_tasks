import pandas as pd
from tkinter import Tk, Button, Label, filedialog, messagebox, StringVar, Entry, Frame
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np


class IrisGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Iris Classification GUI")
        self.root.geometry("700x800")
        self.root.configure(bg="black")  # خلفية سوداء

        self.model = None
        self.X_test = None
        self.y_test = None
        self.result_var = StringVar()

        # Styles for buttons and labels
        btn_style = {"bg": "#444", "fg": "white", "font": ("Arial", 14, "bold"), "width": 25, "height": 2}
        label_style = {"bg": "black", "fg": "white", "font": ("Arial", 12, "bold")}

        # Button to choose CSV file
        Button(root, text="Choose Iris.csv file", command=self.choose_file, **btn_style).pack(pady=10)

        # Label for accuracy
        Label(root, textvariable=self.result_var, **label_style).pack(pady=5)

        # Input fields for manual prediction
        self.entries = []
        fields = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
        self.entry_frame = Frame(root, bg="black")
        self.entry_frame.pack(pady=10)
        for field in fields:
            Label(self.entry_frame, text=field, **label_style).pack()
            entry = Entry(self.entry_frame, font=("Arial", 14), width=20, justify='center')
            entry.pack(pady=3)
            self.entries.append(entry)

        # Buttons to fill sample values
        Button(root, text="Fill Sample (Setosa)", command=lambda: self.fill_sample([5.1, 3.5, 1.4, 0.2]), **btn_style).pack(pady=5)
        Button(root, text="Fill Sample (Versicolor)", command=lambda: self.fill_sample([6.0, 2.9, 4.5, 1.5]), **btn_style).pack(pady=5)
        Button(root, text="Fill Sample (Virginica)", command=lambda: self.fill_sample([6.5, 3.0, 5.2, 2.0]), **btn_style).pack(pady=5)

        # Button to predict
        Button(root, text="Predict", command=self.predict_flower, **btn_style).pack(pady=10)

        # Frame for Pie Chart
        self.pie_frame = Frame(root, bg="black")
        self.pie_frame.pack(pady=10)

        # Button to show Confusion Matrix
        Button(root, text="Show Confusion Matrix", command=self.show_confusion_matrix, **btn_style).pack(pady=10)

        # Frame for Confusion Matrix
        self.cm_frame = Frame(root, bg="black")
        self.cm_frame.pack(pady=10)

    def fill_sample(self, values):
        """Fill entries with sample numeric values."""
        for entry, val in zip(self.entries, values):
            entry.delete(0, 'end')
            entry.insert(0, str(val))

    def choose_file(self):
        """Load CSV, clean columns, train model, and display accuracy."""
        filepath = filedialog.askopenfilename(
            title="Choose data file",
            filetypes=[("CSV files", "*.csv")]
        )
        if filepath:
            try:
                # Load data
                data = pd.read_csv(filepath)

                # Remove 'Id' column if present
                if 'Id' in data.columns:
                    data = data.drop(['Id'], axis=1)

                # Prepare features and labels
                X = data.drop('Species', axis=1)
                y = data['Species'].str.strip()

                # Train/test split
                X_train, self.X_test, y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Train logistic regression model
                self.model = LogisticRegression(max_iter=200)
                self.model.fit(X_train, y_train)

                # Show accuracy
                acc = accuracy_score(self.y_test, self.model.predict(self.X_test))
                self.result_var.set(f"Accuracy: {acc:.2f}")

            except Exception as e:
                messagebox.showerror("Error", f"An error occurred:\n{e}")

    def predict_flower(self):
        """Predict based on input values and show Pie Chart."""
        if self.model is None:
            messagebox.showwarning("Warning", "Choose file and train model first.")
            return

        # Collect inputs
        raw_inputs = [entry.get().strip() for entry in self.entries]

        # Check for empty fields
        if any(val == "" for val in raw_inputs):
            messagebox.showerror("Error", "Please fill all fields with numbers.")
            return

        try:
            # Convert to float
            values = [float(val) for val in raw_inputs]

            # Predict
            prediction = self.model.predict([values])[0]

            # Show result
            messagebox.showinfo("Prediction", f"Predicted class: {prediction}")
            self.show_prediction_pie(prediction)

        except ValueError as e:
            messagebox.showerror("Error", f"Error converting input to float:\n{raw_inputs}\nDetails: {e}")

    def show_prediction_pie(self, predicted_class):
        """Display an improved Pie Chart (Donut) for predicted class."""

        # Clear previous chart
        for widget in self.pie_frame.winfo_children():
            widget.destroy()

        # Classes and prediction values
        classes = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
        values = [1 if c == predicted_class else 0 for c in classes]

        # Colors
        colors = ['#4CAF50', '#2196F3', '#FF7043']

        fig, ax = plt.subplots(figsize=(5, 4))

        # Donut chart
        wedges, texts, autotexts = ax.pie(
            values,
            labels=classes,
            colors=colors,
            startangle=90,
            counterclock=False,
            autopct=lambda p: f'{p:.0f}%' if p > 0 else '',
            wedgeprops=dict(width=0.4, edgecolor='white')
        )

        # Highlight predicted class
        for i, text in enumerate(autotexts):
            if classes[i] == predicted_class:
                text.set_color('black')
                text.set_fontsize(14)
                text.set_weight('bold')

        # Add title
        ax.set_title(f"Predicted Class: {predicted_class}",
                     fontsize=16, fontweight='bold', color='#333')

        # Add legend outside
        ax.legend(wedges, classes, title="Classes", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

        # Embed chart in Tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.pie_frame)
        canvas.draw()
        canvas.get_tk_widget().pack()

        plt.close(fig)

    def show_confusion_matrix(self):
        """Display improved Confusion Matrix for model performance."""
        if self.model is None or self.X_test is None:
            messagebox.showwarning("Warning", "Train the model first to view Confusion Matrix.")
            return

        # Clear previous CM chart
        for widget in self.cm_frame.winfo_children():
            widget.destroy()

        cm = confusion_matrix(self.y_test, self.model.predict(self.X_test))
        classes = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]

        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(cm, cmap='Blues')

        # Add numbers to cells
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, cm[i, j], ha='center', va='center', color='black', fontsize=12, fontweight='bold')

        # Set ticks and labels
        ax.set_xticks(np.arange(len(classes)))
        ax.set_yticks(np.arange(len(classes)))
        ax.set_xticklabels(classes)
        ax.set_yticklabels(classes)
        ax.set_xlabel("Predicted", fontsize=12)
        ax.set_ylabel("Actual", fontsize=12)
        ax.set_title("Confusion Matrix", fontsize=16, fontweight='bold')

        # Embed in Tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.cm_frame)
        canvas.draw()
        canvas.get_tk_widget().pack()

        plt.close(fig)


if __name__ == "__main__":
    root = Tk()
    app = IrisGUI(root)
    root.mainloop()
