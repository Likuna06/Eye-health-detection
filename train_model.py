"""
train_model.py – Eye Disease model trainer (TensorFlow/Keras)

Run:
    python train_model.py
"""

import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Reduce TF log noise
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# =========================
# CONFIG – UPDATE PATH HERE
# =========================
DATA_DIR = r"C:\Users\USER\OneDrive\Desktop\data\dataset"  # <-- Liku's dataset path
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
TEST_SIZE = 0.1
VAL_SIZE = 0.1
RANDOM_STATE = 42


# =========================
# DATASET CLASS
# =========================
class EyeDiseaseDataset:
    def __init__(
        self,
        data_dir,
        img_exts=(".jpg", ".jpeg", ".png"),
        test_size=TEST_SIZE,
        val_size=VAL_SIZE,
        random_state=RANDOM_STATE,
    ):
        if not os.path.isdir(data_dir):
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        self.data_dir = data_dir
        self.img_exts = img_exts
        self.test_size = float(test_size)
        self.val_size = float(val_size)
        self.random_state = random_state
        self.df = self._build_df()

    def _build_df(self):
        rows = []
        for cls in os.listdir(self.data_dir):
            cls_path = os.path.join(self.data_dir, cls)
            if not os.path.isdir(cls_path):
                continue
            for fname in os.listdir(cls_path):
                if fname.lower().endswith(self.img_exts):
                    rows.append({"filepath": os.path.join(cls_path, fname), "label": cls})
        if not rows:
            raise RuntimeError(f"No images found under {self.data_dir}. Check folder structure.")
        return pd.DataFrame(rows)

    def split_(self):
        df = self.df.copy()

        # test split
        train_val_df, test_df = train_test_split(
            df,
            test_size=self.test_size,
            stratify=df["label"],
            random_state=self.random_state,
            shuffle=True,
        )

        # validation relative to remaining
        rel_val = self.val_size / (1.0 - self.test_size)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=rel_val,
            stratify=train_val_df["label"],
            random_state=self.random_state,
            shuffle=True,
        )

        return (
            train_df.reset_index(drop=True),
            val_df.reset_index(drop=True),
            test_df.reset_index(drop=True),
        )

    def split(self):
        return self.split_()


# =========================
# BUILD DATA GENERATORS
# =========================
def build_generators(train_df, val_df):
    datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col="filepath",
        y_col="label",
        target_size=IMG_SIZE,
        class_mode="categorical",
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    val_gen = datagen.flow_from_dataframe(
        dataframe=val_df,
        x_col="filepath",
        y_col="label",
        target_size=IMG_SIZE,
        class_mode="categorical",
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    return train_gen, val_gen


# =========================
# MODEL
# =========================
def build_model(num_classes: int):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(128, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax'),
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# =========================
# TRAINING + SAVE + PLOTS
# =========================
def main():
    print("\n" + "="*72)
    print("🚀 STARTING EYE DISEASE TRAINING")
    print("="*72)
    print(f"📂 DATA_DIR: {DATA_DIR}")

    if not os.path.isdir(DATA_DIR):
        raise SystemExit(f"❌ Directory not found: {DATA_DIR}")

    dataset = EyeDiseaseDataset(DATA_DIR)
    train_df, val_df, test_df = dataset.split()

    print(f"📊 Total images: {len(dataset.df)}")
    print(f"   Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    print("   Classes:", sorted(train_df['label'].unique()))
    print("-"*72)

    train_gen, val_gen = build_generators(train_df, val_df)

    # Correct num-classes
    num_classes = len(train_gen.class_indices)
    print(f"🧾 num_classes = {num_classes}  (from train_gen.class_indices)")

    model = build_model(num_classes)
    model.summary(print_fn=lambda s: print("   " + s))

    print("\n🧠 Training...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        verbose=1,
    )
    # ====================
    # SAVE MODEL + LABELS
    # ====================
    model.save("model.h5")
    print("💾 Saved model.h5")

    with open("class_indices.txt", "w", encoding="utf-8") as f:
        for label, idx in sorted(train_gen.class_indices.items(), key=lambda x: x[1]):
            f.write(f"{idx}\t{label}\n")
    print("💾 Saved class_indices.txt")

    # Save training history
    with open("history.pkl", "wb") as f:
        pickle.dump(history.history, f)
    print("💾 Saved history.pkl")
    # ====================
    # PLOTS
    # ====================
    train_acc = history.history.get('accuracy', [])
    val_acc = history.history.get('val_accuracy', [])
    train_loss = history.history.get('loss', [])
    val_loss = history.history.get('val_loss', [])
    print(f"\nFinal Training Accuracy: {train_acc[-1] if train_acc else 'NA'}")
    print(f"Final Validation Accuracy: {val_acc[-1] if val_acc else 'NA'}")
    # Loss plot
    plt.figure(figsize=(8, 4))
    plt.plot(train_loss, label='Training Loss')
    if val_loss: plt.plot(val_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig("loss_curve.png", dpi=150)
    plt.close()
    print("📈 Saved loss_curve.png")
    # Accuracy plot
    plt.figure(figsize=(8, 4))
    plt.plot(train_acc, label='Training Accuracy')
    if val_acc: plt.plot(val_acc, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig("accuracy_curve.png", dpi=150)
    plt.close()
    print("📈 Saved accuracy_curve.png")
    print("\n✅ DONE.")
if __name__ == "__main__":
    main()