# =====================================================
# MNIST Neural Network Comparison
# 6 Hidden Layers – Sigmoid vs Tanh
# Epochs + Tables SHOWN
# PNGs + Results file GENERATED
# =====================================================

import os
import sys

# --- Force UTF-8 (prevents Windows Unicode errors)
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# -----------------------------------------------------
# 1. Load MNIST dataset locally
# -----------------------------------------------------
with np.load("mnist.npz") as data:
    x_train = data["x_train"]
    y_train = data["y_train"]
    x_test = data["x_test"]
    y_test = data["y_test"]

# -----------------------------------------------------
# 2. Preprocess data
# -----------------------------------------------------
x_train = x_train.reshape(60000, 784).astype("float32") / 255
x_test = x_test.reshape(10000, 784).astype("float32") / 255

y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

# -----------------------------------------------------
# 3. Model builder (6 hidden layers)
# -----------------------------------------------------
def build_model(hidden_activation):
    model = Sequential()
    model.add(Input(shape=(784,)))

    # Six hidden layers
    model.add(Dense(128, activation=hidden_activation))
    model.add(Dense(128, activation=hidden_activation))
    model.add(Dense(128, activation=hidden_activation))
    model.add(Dense(128, activation=hidden_activation))
    model.add(Dense(128, activation=hidden_activation))
    model.add(Dense(128, activation=hidden_activation))

    # Output layer
    model.add(Dense(10, activation="softmax"))

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

# -----------------------------------------------------
# 4. Train SIGMOID model (epochs shown)
# -----------------------------------------------------
print("\nTraining model with SIGMOID activation\n")
sigmoid_model = build_model("sigmoid")

print("Sigmoid Model Summary:")
sigmoid_model.summary()

history_sigmoid = sigmoid_model.fit(
    x_train,
    y_train_cat,
    epochs=10,
    batch_size=128,
    validation_split=0.1,
    verbose=2
)

# -----------------------------------------------------
# 5. Train TANH model (epochs shown)
# -----------------------------------------------------
print("\nTraining model with TANH activation\n")
tanh_model = build_model("tanh")

print("Tanh Model Summary:")
tanh_model.summary()

history_tanh = tanh_model.fit(
    x_train,
    y_train_cat,
    epochs=10,
    batch_size=128,
    validation_split=0.1,
    verbose=2
)

# -----------------------------------------------------
# 6. Evaluate models
# -----------------------------------------------------
sigmoid_loss, sigmoid_acc = sigmoid_model.evaluate(x_test, y_test_cat, verbose=0)
tanh_loss, tanh_acc = tanh_model.evaluate(x_test, y_test_cat, verbose=0)

print("\nTest Accuracy Results")
print(f"Sigmoid Model Accuracy: {sigmoid_acc:.4f}")
print(f"Tanh Model Accuracy:    {tanh_acc:.4f}")

# -----------------------------------------------------
# 7. Classification reports (tables)
# -----------------------------------------------------
y_pred_sigmoid = np.argmax(sigmoid_model.predict(x_test, verbose=0), axis=1)
y_pred_tanh = np.argmax(tanh_model.predict(x_test, verbose=0), axis=1)

print("\nSigmoid Classification Report\n")
print(classification_report(y_test, y_pred_sigmoid))

print("\nTanh Classification Report\n")
print(classification_report(y_test, y_pred_tanh))

# -----------------------------------------------------
# 8. Save validation accuracy comparison PNG
# -----------------------------------------------------
plt.figure()
plt.plot(history_sigmoid.history["val_accuracy"], label="Sigmoid")
plt.plot(history_tanh.history["val_accuracy"], label="Tanh")
plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy")
plt.title("Validation Accuracy: Sigmoid vs Tanh")
plt.legend()
plt.savefig("fig_val_accuracy_compare.png", dpi=300)
plt.show()

# -----------------------------------------------------
# 9. Save validation loss comparison PNG
# -----------------------------------------------------
plt.figure()
plt.plot(history_sigmoid.history["val_loss"], label="Sigmoid")
plt.plot(history_tanh.history["val_loss"], label="Tanh")
plt.xlabel("Epoch")
plt.ylabel("Validation Loss")
plt.title("Validation Loss: Sigmoid vs Tanh")
plt.legend()
plt.savefig("fig_val_loss_compare.png", dpi=300)
plt.show()

# -----------------------------------------------------
# 10. Confusion matrices (PNG)
# -----------------------------------------------------
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred_sigmoid
)
plt.title("Confusion Matrix – Sigmoid")
plt.savefig("fig_confusion_sigmoid.png", dpi=300)
plt.show()

ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred_tanh
)
plt.title("Confusion Matrix – Tanh")
plt.savefig("fig_confusion_tanh.png", dpi=300)
plt.show()

# -----------------------------------------------------
# 11. Save results summary text file
# -----------------------------------------------------
with open("results_summary.txt", "w", encoding="utf-8") as f:
    f.write("MNIST Neural Network Comparison\n")
    f.write("6 Hidden Layers – Sigmoid vs Tanh\n")
    f.write("================================\n\n")
    f.write(f"Sigmoid Test Accuracy: {sigmoid_acc:.4f}\n")
    f.write(f"Tanh Test Accuracy:    {tanh_acc:.4f}\n\n")

    f.write("Sigmoid Classification Report:\n")
    f.write(classification_report(y_test, y_pred_sigmoid))
    f.write("\n\n")

    f.write("Tanh Classification Report:\n")
    f.write(classification_report(y_test, y_pred_tanh))

print("\nALL FILES GENERATED SUCCESSFULLY:")
print(" - fig_val_accuracy_compare.png")
print(" - fig_val_loss_compare.png")
print(" - fig_confusion_sigmoid.png")
print(" - fig_confusion_tanh.png")
print(" - results_summary.txt")
