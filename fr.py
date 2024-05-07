import tkinter as tk
from tkinter import ttk
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd


class FraudDetector:
    def __init__(self):
        self.model = None
        self.X = None
        self.y = None

    def load_data(self):
        try:
            # Load your dataset here
            data = pd.read_csv("fraudTrain.csv")
            # sample a subset of the data
            data=data.sample(frac=0.1, random_state=42)
            # specify columns for one-hot encoding
            categorical_cols=["trans_date_trans_time","cc_num","merchant","category","first", "last","gender", "street","state","job","dob","trans_num","unix_time"]
            # Preprocess data: One-hot encoding for categorical variables
            data = pd.get_dummies(data, columns=categorical_cols ,prefix="cat")

            self.X = data.drop(columns=["is_fraud"])
            print(self.X.shape)
            self.y = data["is_fraud"]
        except Exception as e:
            print("Error loading data:", e)

    def train_model(self):
        try:
            # Split dataset into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

            # Initialize Random Forest classifier
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)

            # Train the model
            self.model.fit(X_train, y_train)

            # Evaluate model accuracy on test set
            y_pred = self.model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            print(f"Model Accuracy: {acc}")

            # Display confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            print("Confusion Matrix:")
            print(cm)
        except Exception as e:
            print("Error training model:", e)

    def predict(self, data):
        try:
            if self.model:
                # Predict fraud labels for new data
                return self.model.predict(data)
            else:
                print("Model not trained yet.")
                return None
        except Exception as e:
            print("Error predicting:", e)


def detect_fraud():
    # Load data and train model
    fraud_detector = FraudDetector()
    fraud_detector.load_data()
    fraud_detector.train_model()

    # Predict fraud for new data
    new_data = [[0, 1, 0, ...], [1, 0, 1, ...], ...]  # Example new data
    predictions = fraud_detector.predict(new_data)

    # Display predictions if available
    if predictions is not None:
        result_text.config(text="Fraud Predictions:\n" + "\n".join(str(pred) for pred in predictions))


# Create Tkinter window
window = tk.Tk()
window.title("Fraud Detection System")
window.geometry("400x300")

# Create heading label
heading_label = ttk.Label(window, text="Fraud Detection System", font=("Helvetica", 18))
heading_label.pack(pady=10)

# Create button to detect fraud
detect_button = ttk.Button(window, text="Detect Fraud", command=detect_fraud)
detect_button.pack(pady=10)

# Create label to display results
result_text = tk.Label(window, text="", font=("Helvetica", 12), wraplength=300, justify="left")
result_text.pack(pady=10)

# Run the Tkinter event loop
window.mainloop()
