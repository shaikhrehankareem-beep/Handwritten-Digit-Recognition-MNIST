# LOAD LIBRARIES
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler
import matplotlib.pyplot as plt

# ============================================================
# CONFIG
# ============================================================
TRAIN_PATH = r"C:\Users\HP\OneDrive\Desktop\Handwritten Digit Recognition (MNIST)\train.csv"
TEST_PATH = r"C:\Users\HP\OneDrive\Desktop\Handwritten Digit Recognition (MNIST)\test.csv"

# ============================================================
# LOAD THE DATA
# ============================================================
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

# ============================================================
# PREPARE DATA FOR NEURAL NETWORK
# ============================================================
Y_train = train["label"]
X_train = train.drop(labels=["label"], axis=1)
X_train = X_train / 255.0
X_test = test / 255.0
X_train = X_train.values.reshape(-1, 28, 28, 1)
X_test = X_test.values.reshape(-1, 28, 28, 1)
Y_train = to_categorical(Y_train, num_classes=10)

# ============================================================
# PREVIEW IMAGES (saved to file, no popup window)
# ============================================================
plt.figure(figsize=(15, 4.5))
for i in range(30):
    plt.subplot(3, 10, i + 1)
    plt.imshow(X_train[i].reshape((28, 28)), cmap=plt.cm.binary)
    plt.axis('off')
plt.subplots_adjust(wspace=-0.1, hspace=-0.1)
plt.savefig("preview_images.png")
plt.close()
print("Saved preview_images.png")

# ============================================================
# DATA AUGMENTATION
# ============================================================
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.10,
    width_shift_range=0.1,
    height_shift_range=0.1
)

# PREVIEW AUGMENTED IMAGES (saved to file, no popup window)
X_train3 = X_train[9, ].reshape((1, 28, 28, 1))
Y_train3 = Y_train[9, ].reshape((1, 10))
plt.figure(figsize=(15, 4.5))
for i in range(30):
    plt.subplot(3, 10, i + 1)
    X_train2, Y_train2 = next(datagen.flow(X_train3, Y_train3))
    plt.imshow(X_train2[0].reshape((28, 28)), cmap=plt.cm.binary)
    plt.axis('off')
    if i == 9:
        X_train3 = X_train[11, ].reshape((1, 28, 28, 1))
    if i == 19:
        X_train3 = X_train[18, ].reshape((1, 28, 28, 1))
plt.subplots_adjust(wspace=-0.1, hspace=-0.1)
plt.savefig("augmented_preview.png")
plt.close()
print("Saved augmented_preview.png")

# ============================================================
# BUILD CNN MODELS (ENSEMBLE)
# ============================================================
# NOTE: nets=15, epochs=45 bahut heavy hai (GPU chahiye, ghanto lag sakte hain).
# CPU pe pehle yeh chota version chalao (1-2 min lagega), sab sahi lage to badhao.
nets = 3
epochs = 8

model = [0] * nets
for j in range(nets):
    model[j] = Sequential()

    model[j].add(Conv2D(32, kernel_size=3, activation='relu', input_shape=(28, 28, 1)))
    model[j].add(BatchNormalization())
    model[j].add(Conv2D(32, kernel_size=3, activation='relu'))
    model[j].add(BatchNormalization())
    model[j].add(Conv2D(32, kernel_size=5, strides=2, padding='same', activation='relu'))
    model[j].add(BatchNormalization())
    model[j].add(Dropout(0.4))

    model[j].add(Conv2D(64, kernel_size=3, activation='relu'))
    model[j].add(BatchNormalization())
    model[j].add(Conv2D(64, kernel_size=3, activation='relu'))
    model[j].add(BatchNormalization())
    model[j].add(Conv2D(64, kernel_size=5, strides=2, padding='same', activation='relu'))
    model[j].add(BatchNormalization())
    model[j].add(Dropout(0.4))

    model[j].add(Conv2D(128, kernel_size=4, activation='relu'))
    model[j].add(BatchNormalization())
    model[j].add(Flatten())
    model[j].add(Dropout(0.4))
    model[j].add(Dense(10, activation='softmax'))

    model[j].compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# DECREASE LEARNING RATE EACH EPOCH
annealer = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x)

# ============================================================
# TRAIN NETWORKS
# ============================================================
history = [0] * nets
for j in range(nets):
    X_train2, X_val2, Y_train2, Y_val2 = train_test_split(X_train, Y_train, test_size=0.1)
    history[j] = model[j].fit(
        datagen.flow(X_train2, Y_train2, batch_size=64),
        epochs=epochs,
        steps_per_epoch=X_train2.shape[0] // 64,
        validation_data=(X_val2, Y_val2),
        callbacks=[annealer],
        verbose=1
    )
    print("CNN {0:d}: Epochs={1:d}, Train accuracy={2:.5f}, Validation accuracy={3:.5f}".format(
        j + 1, epochs,
        max(history[j].history['accuracy']),
        max(history[j].history['val_accuracy'])
    ))

# ============================================================
# ENSEMBLE PREDICTIONS AND SUBMIT
# ============================================================
results = np.zeros((X_test.shape[0], 10))
for j in range(nets):
    results = results + model[j].predict(X_test)
results = np.argmax(results, axis=1)
results = pd.Series(results, name="Label")
submission = pd.concat([pd.Series(range(1, len(results) + 1), name="ImageId"), results], axis=1)
submission.to_csv("MNIST-CNN-ENSEMBLE.csv", index=False)
print("Saved MNIST-CNN-ENSEMBLE.csv")

# ============================================================
# PREVIEW PREDICTIONS (saved to file, no popup window)
# ============================================================
plt.figure(figsize=(15, 6))
for i in range(40):
    plt.subplot(4, 10, i + 1)
    plt.imshow(X_test[i].reshape((28, 28)), cmap=plt.cm.binary)
    plt.title("predict=%d" % results[i], y=0.9)
    plt.axis('off')
plt.subplots_adjust(wspace=0.3, hspace=-0.1)
plt.savefig("predictions_preview.png")
plt.close()
print("Saved predictions_preview.png")

print("\nALL DONE. Check these files in your project folder:")
print(" - preview_images.png")
print(" - augmented_preview.png")
print(" - predictions_preview.png")
print(" - MNIST-CNN-ENSEMBLE.csv")
