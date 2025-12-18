r"""
train_svm_from_cnn.py

Windows'ta yerel çalışmak üzere hazırlanmış script.
- Örnek çalıştırma:
  python train_svm_from_cnn.py --train_path "C:\Users\ataka\Desktop\MachineLearningProject\training_set\training_set" --test_path "C:\Users\ataka\Desktop\MachineLearningProject\test_set\test_set" --epochs 20 --backbone mobilenetv2 --fine_tune

Not:
- train_path ve test_path, klasör içindeki class'lara göre (class alt klasörleri) organize edilmiş olmalı.
- Varsayılan backbone: MobileNetV2 (hafif ve hızlı). İsterseniz --backbone vgg16 veya efficientnetb0 seçebilirsiniz.
- Eğer GPU'nuz varsa (RTX 3060) TensorFlow otomatik olarak kullanacaktır (CUDA/cuDNN kurulumu varsa).
"""

import os
import argparse
import numpy as np
from pathlib import Path
import joblib
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model

from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

AUTOTUNE = tf.data.AUTOTUNE

BACKBONES = {
    "mobilenetv2": {
        "builder": tf.keras.applications.MobileNetV2,
        "preprocess": tf.keras.applications.mobilenet_v2.preprocess_input,
        "input_size": (224, 224)
    },
    "vgg16": {
        "builder": tf.keras.applications.VGG16,
        "preprocess": tf.keras.applications.vgg16.preprocess_input,
        "input_size": (224, 224)
    },
    "efficientnetb0": {
        "builder": tf.keras.applications.EfficientNetB0,
        "preprocess": tf.keras.applications.efficientnet.preprocess_input,
        "input_size": (224, 224)
    }
}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_path", default=r"C:\Users\ataka\Desktop\MachineLearningProject\training_set\training_set", help="Path to training folder (with class subfolders).")
    p.add_argument("--test_path", default=r"C:\Users\ataka\Desktop\MachineLearningProject\test_set\test_set", help="Path to test folder (with class subfolders).")
    p.add_argument("--backbone", default="mobilenetv2", choices=BACKBONES.keys(), help="Backbone model.")
    p.add_argument("--img_size", type=int, default=224, help="Image size (square).")
    p.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    p.add_argument("--epochs", type=int, default=20, help="Epochs for fine-tuning (if enabled).")
    p.add_argument("--fine_tune", action="store_true", help="If set, fine-tune the backbone for given epochs before extracting features.")
    p.add_argument("--fine_tune_at", type=int, default=100, help="If fine-tune: layer index from which to unfreeze (negative allowed). Default 100 (will be adjusted based on model).")
    p.add_argument("--svm_kernel", default="linear", help="SVM kernel (linear, rbf, poly...). Linear çok daha hızlı.")
    p.add_argument("--output_dir", default="output_models", help="Where to save trained models and reports.")
    p.add_argument("--validation_split", type=float, default=0.2, help="Validation split for fine-tuning (only used if --fine_tune).")
    return p.parse_args()

def prepare_datasets(train_path, test_path, img_size, batch_size, validation_split=0.2, fine_tune=False, seed=123):
    # For feature extraction (full training set):
    train_ds_full = tf.keras.preprocessing.image_dataset_from_directory(
        train_path,
        labels="inferred",
        label_mode="int",
        image_size=(img_size, img_size),
        batch_size=batch_size,
        shuffle=False
    )

    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        test_path,
        labels="inferred",
        label_mode="int",
        image_size=(img_size, img_size),
        batch_size=batch_size,
        shuffle=False
    )

    class_names = train_ds_full.class_names

    if fine_tune:
        # Create a train/val split for fine-tuning
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            train_path,
            labels="inferred",
            label_mode="int",
            image_size=(img_size, img_size),
            batch_size=batch_size,
            shuffle=True,
            validation_split=validation_split,
            subset="training",
            seed=seed
        )
        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            train_path,
            labels="inferred",
            label_mode="int",
            image_size=(img_size, img_size),
            batch_size=batch_size,
            shuffle=True,
            validation_split=validation_split,
            subset="validation",
            seed=seed
        )

        # Prefetch
        train_ds = train_ds.prefetch(AUTOTUNE)
        val_ds = val_ds.prefetch(AUTOTUNE)
        train_ds_full = train_ds_full.prefetch(AUTOTUNE)
        test_ds = test_ds.prefetch(AUTOTUNE)

        return train_ds, val_ds, train_ds_full, test_ds, class_names
    else:
        train_ds_full = train_ds_full.prefetch(AUTOTUNE)
        test_ds = test_ds.prefetch(AUTOTUNE)
        return None, None, train_ds_full, test_ds, class_names

def build_backbone(name, input_shape, weights="imagenet", pooling="avg"):
    info = BACKBONES[name]
    builder = info["builder"]
    base_model = builder(include_top=False, weights=weights, input_shape=input_shape, pooling=pooling)
    return base_model, info["preprocess"]

def extract_features_from_dataset(feature_extractor, preprocess_fn, dataset):
    features = []
    labels = []
    for batch_images, batch_labels in dataset:
        # Preprocess
        x = preprocess_fn(tf.cast(batch_images, tf.float32).numpy())
        feats = feature_extractor.predict(x, verbose=0)
        features.append(feats)
        labels.append(batch_labels.numpy())
    features = np.vstack(features)
    labels = np.concatenate(labels)
    return features, labels

def build_and_train_classifier(X_train, y_train, kernel="rbf"):
    scaler = StandardScaler()
    svc = SVC(kernel=kernel, probability=True, class_weight='balanced')
    pipeline = Pipeline([("scaler", scaler), ("svc", svc)])
    pipeline.fit(X_train, y_train)
    return pipeline

def evaluate_and_report(pipeline, X_test, y_test, class_names, out_dir):
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=class_names, digits=4)
    cm = confusion_matrix(y_test, y_pred)
    # Save
    with open(os.path.join(out_dir, "svm_report.txt"), "w", encoding="utf-8") as f:
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write("Classification report:\n")
        f.write(report + "\n\n")
        f.write("Confusion matrix:\n")
        f.write(np.array2string(cm))
    return acc, report, cm

def plot_history(history, out_dir):
    # history is a Keras History object
    if history is None:
        return
    plt.figure(figsize=(8,4))
    plt.plot(history.history.get("loss", []), label="train_loss")
    plt.plot(history.history.get("val_loss", []), label="val_loss")
    plt.legend(); plt.title("Loss")
    plt.savefig(os.path.join(out_dir, "loss.png"))
    plt.close()

    plt.figure(figsize=(8,4))
    # accuracy could be named accuracy or sparse_categorical_accuracy depending on compile
    acc_key = "accuracy" if "accuracy" in history.history else "sparse_categorical_accuracy"
    val_acc_key = "val_" + acc_key
    if acc_key in history.history:
        plt.plot(history.history.get(acc_key, []), label="train_acc")
    if val_acc_key in history.history:
        plt.plot(history.history.get(val_acc_key, []), label="val_acc")
    plt.legend(); plt.title("Accuracy")
    plt.savefig(os.path.join(out_dir, "accuracy.png"))
    plt.close()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("Script configuration:", args)

    # Prepare datasets
    print("Preparing datasets...")
    if args.fine_tune:
        train_ds, val_ds, train_ds_full, test_ds, class_names = prepare_datasets(
            args.train_path, args.test_path, args.img_size, args.batch_size, validation_split=args.validation_split, fine_tune=True)
    else:
        train_ds, val_ds, train_ds_full, test_ds, class_names = prepare_datasets(
            args.train_path, args.test_path, args.img_size, args.batch_size, fine_tune=False)

    num_classes = len(class_names)
    print(f"Found classes: {class_names} (num_classes={num_classes})")

    # Build backbone
    input_shape = (args.img_size, args.img_size, 3)
    print(f"Building backbone {args.backbone}...")
    base_model, preprocess_fn = build_backbone(args.backbone, input_shape=input_shape)
    base_model.trainable = False
    print("Backbone summary:")
    base_model.summary()

    fine_tuned_model = None
    history = None

    if args.fine_tune:
        # Build a small classification head on top of base_model for fine-tuning
        print("Preparing model for fine-tuning...")
        inputs = keras.Input(shape=input_shape)
        x = preprocess_fn(inputs)
        x = base_model(x, training=False)
        x = layers.Dense(256, activation="relu")(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(num_classes, activation="softmax")(x)
        model = Model(inputs, outputs)
        # Unfreeze last layers: compute an index to unfreeze from
        base_model.trainable = True
        # Freeze all then unfreeze last N layers
        for layer in base_model.layers:
            layer.trainable = False
        # fine_tune_at could be provided relative; ensure it's not out of range
        ft_at = args.fine_tune_at
        if ft_at < 0:
            ft_at = len(base_model.layers) + ft_at
        ft_at = max(0, min(ft_at, len(base_model.layers)-1))
        for layer in base_model.layers[ft_at:]:
            layer.trainable = True

        model.compile(optimizer=keras.optimizers.Adam(1e-4),
                      loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"])
        model.summary()

        print(f"Starting fine-tuning for {args.epochs} epochs (validation_split={args.validation_split})...")
        history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs)
        plot_history(history, args.output_dir)

        # After fine-tuning, use the fine-tuned backbone for feature extraction
        # Create a new model that outputs the pooled features from the (now fine-tuned) base
        # We need to build a feature extractor: inputs -> preprocess -> base_model -> output
        inputs2 = keras.Input(shape=input_shape)
        x2 = preprocess_fn(inputs2)
        x2 = base_model(x2, training=False)
        feature_extractor = Model(inputs2, x2)
        fine_tuned_model = model
        # Save fine-tuned classification model for reference
        model.save(os.path.join(args.output_dir, "fine_tuned_classification_model.h5"))
        print("Saved fine-tuned Keras model to", os.path.join(args.output_dir, "fine_tuned_classification_model.h5"))
    else:
        # Use frozen base_model for feature extraction
        inputs2 = keras.Input(shape=input_shape)
        x2 = preprocess_fn(inputs2)
        x2 = base_model(x2, training=False)
        feature_extractor = Model(inputs2, x2)

    # Extract features for train (use train_ds_full which contains all training images)
    print("Extracting features for training set...")
    X_train, y_train = extract_features_from_dataset(feature_extractor, preprocess_fn, train_ds_full)
    print("Training features shape:", X_train.shape, "labels shape:", y_train.shape)

    # Train SVM
    print("Training SVM classifier (kernel=%s) ..." % args.svm_kernel)
    svm_pipeline = build_and_train_classifier(X_train, y_train, kernel=args.svm_kernel)
    joblib.dump(svm_pipeline, os.path.join(args.output_dir, "svm_pipeline.joblib"))
    print("Saved SVM pipeline to", os.path.join(args.output_dir, "svm_pipeline.joblib"))

    # Extract features for test set and evaluate
    print("Extracting features for test set...")
    X_test, y_test = extract_features_from_dataset(feature_extractor, preprocess_fn, test_ds)
    print("Test features shape:", X_test.shape, "labels shape:", y_test.shape)

    acc, report, cm = evaluate_and_report(svm_pipeline, X_test, y_test, class_names, args.output_dir)
    print(f"Test accuracy (SVM): {acc:.4f}")
    print("Classification report:\n", report)
    print("Confusion matrix:\n", cm)
    print("Detailed report saved to", os.path.join(args.output_dir, "svm_report.txt"))

    # Done
    print("All done. Outputs in directory:", args.output_dir)

if __name__ == "__main__":
    main()