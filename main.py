import os
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import kagglehub

IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 15
SEED = 42

tf.random.set_seed(SEED)
np.random.seed(SEED)

print("Downloading dataset...")

dataset_path = kagglehub.dataset_download(
    "navoneel/brain-mri-images-for-brain-tumor-detection"
)

dataset_path = Path(dataset_path)
print("Original dataset path:", dataset_path)

brain_dataset_path = None

for root, dirs, files in os.walk(dataset_path):
    if "yes" in dirs and "no" in dirs:
        brain_dataset_path = Path(root)
        break

if brain_dataset_path is None:
    raise FileNotFoundError("Could not find folders named yes and no.")

print("Using dataset folder:", brain_dataset_path)

def load_dataset(folder_path):
    image_paths = []
    labels = []

    valid_extensions = [".jpg", ".jpeg", ".png", ".bmp"]

    for class_name in ["yes", "no"]:
        class_folder = folder_path / class_name

        if not class_folder.exists():
            raise FileNotFoundError(f"Folder not found: {class_folder}")

        for file in os.listdir(class_folder):
            file_path = class_folder / file

            if file_path.suffix.lower() in valid_extensions:
                image_paths.append(str(file_path))

                if class_name == "yes":
                    labels.append(1)   # Tumor
                else:
                    labels.append(0)   # No Tumor

    return image_paths, labels


images, labels = load_dataset(brain_dataset_path)

print("Total images:", len(images))
print("Tumor images:", sum(labels))
print("No tumor images:", len(labels) - sum(labels))

train_images, temp_images, train_labels, temp_labels = train_test_split(
    images,
    labels,
    test_size=0.30,
    random_state=SEED,
    stratify=labels
)

valid_images, test_images, valid_labels, test_labels = train_test_split(
    temp_images,
    temp_labels,
    test_size=0.50,
    random_state=SEED,
    stratify=temp_labels
)

print("Training images:", len(train_images))
print("Validation images:", len(valid_images))
print("Testing images:", len(test_images))

def load_image(image_path, label):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.cast(image, tf.float32)

    return image, tf.cast(label, tf.float32)


def make_dataset(image_paths, labels, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(buffer_size=len(image_paths), seed=SEED)

    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


train_ds = make_dataset(train_images, train_labels, shuffle=True)
valid_ds = make_dataset(valid_images, valid_labels)
test_ds = make_dataset(test_images, test_labels)

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.10),
    layers.RandomContrast(0.10),
])

base_model = EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

base_model.trainable = False

inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

x = data_augmentation(inputs)
x = tf.keras.applications.efficientnet.preprocess_input(x)
x = base_model(x, training=False)

x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.35)(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.25)(x)

outputs = layers.Dense(1, activation="sigmoid")(x)

model = models.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall")
    ]
)

model.summary()

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        "best_brain_tumor_model.keras",
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    ),

    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=2,
        min_lr=1e-7,
        verbose=1
    )
]

print("\nTraining started...\n")

history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

print("\nFine tuning started...\n")

base_model.trainable = True

for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00005),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall")
    ]
)

history_fine = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=8,
    callbacks=callbacks
)

print("\nTesting model...\n")

test_loss, test_acc, test_precision, test_recall = model.evaluate(test_ds)

print("\nFinal Test Results")
print("Accuracy :", round(test_acc * 100, 2), "%")
print("Precision:", round(test_precision * 100, 2), "%")
print("Recall   :", round(test_recall * 100, 2), "%")

y_true = np.array(test_labels)

y_prob = model.predict(test_ds)
y_pred = (y_prob > 0.5).astype(int).flatten()

print("\nClassification Report:")
print(
    classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["No Tumor", "Tumor"]
    )
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred, labels=[0, 1]))

model.save("brain_tumor_detection_model.h5")
model.save("brain_tumor_detection_model.keras")

print("\nModel saved successfully:")
print("brain_tumor_detection_model.h5")
print("brain_tumor_detection_model.keras")
print("best_brain_tumor_model.keras")

def predict_single_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.cast(image, tf.float32)
    image = tf.expand_dims(image, axis=0)

    prediction = model.predict(image)[0][0]

    if prediction > 0.5:
        result = "Tumor Detected"
        confidence = prediction * 100
    else:
        result = "No Tumor Detected"
        confidence = (1 - prediction) * 100

    print("\nSingle Image Prediction Result")
    print("Image:", image_path)
    print("Result:", result)
    print("Confidence:", round(float(confidence), 2), "%")

def predict_single_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.cast(image, tf.float32)
    image = tf.expand_dims(image, axis=0)

    prediction = model.predict(image)[0][0]

    if prediction > 0.5:
        result = "Tumor Detected"
        confidence = prediction * 100
    else:
        result = "No Tumor Detected"
        confidence = (1 - prediction) * 100

    print("\n==============================")
    print("Single Image Prediction Result")
    print("==============================")
    print("Image:", image_path)
    print("Result:", result)
    print("Confidence:", round(float(confidence), 2), "%")
    print("==============================\n")

predict_single_image("sample_mri.jpg")
