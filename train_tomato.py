import os
import pathlib
import tensorflow as tf
# Configuration
DATASET_DIR = os.environ.get("TOMATO_DATASET_DIR", r"C:\\Users\\arunk\\Downloads\\TomatoDataset")
OUTPUT_DIR = pathlib.Path("models")
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_STAGE1 = 8  # initial frozen base training
EPOCHS_STAGE2 = 4  # fine-tuning with top layers unfrozen
VAL_SPLIT = 0.2
SEED = 1337
def build_datasets(dataset_dir: str):
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        dataset_dir,
        validation_split=VAL_SPLIT,
        subset="training",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
    )
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        dataset_dir,
        validation_split=VAL_SPLIT,
        subset="validation",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
    )
    class_names = train_ds.class_names
    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.shuffle(1000).prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)
    return train_ds, val_ds, class_names


def build_model(num_classes: int) -> tf.keras.Model:
    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.05),
            tf.keras.layers.RandomZoom(0.1),
        ]
    )
    base = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet"
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = data_augmentation(x)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    return model, base

def train_and_export():
    print(f"Using dataset: {DATASET_DIR}")
    OUTPUT_DIR.mkdir(exist_ok=True)

    train_ds, val_ds, class_names = build_datasets(DATASET_DIR)
    num_classes = len(class_names)
    print("Classes:", class_names)

    model, base = build_model(num_classes)

    # Stage 1: frozen base
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_STAGE1)

    # Stage 2: fine-tune top layers
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_STAGE2)

    # Save labels
    (OUTPUT_DIR / "labels.txt").write_text("\n".join(class_names), encoding="utf-8")
    print("Saved labels to:", OUTPUT_DIR / "labels.txt")

    # Export TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    (OUTPUT_DIR / "mobilenet_v2.tflite").write_bytes(tflite_model)
    print("Saved TFLite model to:", OUTPUT_DIR / "mobilenet_v2.tflite")
if __name__ == "__main__":
    train_and_export()


