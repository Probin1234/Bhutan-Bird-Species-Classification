import json
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import StratifiedKFold
from config import *

keras.utils.set_random_seed(SEED)

# Load dataset and class names
df = pd.read_csv(DATA / "manifest.csv")
names = json.loads((DATA / "classes.json").read_text())

n = len(names)

# Combine training and validation data
# Keep the original test dataset untouched
pool = df[df["split"].isin(
    ["train", "valid"]
)].reset_index(drop=True)


# Prepare TensorFlow dataset
def dataset(frame, training=False):

    d = tf.data.Dataset.from_tensor_slices(
        (
            frame.path.tolist(),
            frame.label.astype("int32").tolist()
        )
    )

    if training:
        d = d.shuffle(len(frame), seed=SEED)

    def load(path, y):

        x = tf.io.decode_image(
            tf.io.read_file(path),
            channels=3,
            expand_animations=False
        )

        x = tf.image.resize(x, SIZE)

        return x, tf.one_hot(y, n)

    return d.map(
        load,
        num_parallel_calls=tf.data.AUTOTUNE
    ).batch(BATCH).prefetch(tf.data.AUTOTUNE)


# Build a fresh EfficientNet model for each fold
def build():

    inp = keras.Input((*SIZE, 3))

    x = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomContrast(0.15)
    ])(inp)

    base = keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(*SIZE, 3)
    )

    base.trainable = False

    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)

    out = layers.Dense(
        n,
        activation="softmax"
    )(x)

    model = keras.Model(inp, out)

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# Stratified five-fold cross-validation
skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=SEED
)

scores = []

for fold, (train_idx, valid_idx) in enumerate(
    skf.split(pool, pool.label),
    start=1
):

    print(f"\n========== FOLD {fold}/5 ==========")

    keras.backend.clear_session()

    train_fold = pool.iloc[train_idx]
    valid_fold = pool.iloc[valid_idx]

    print("Training images:", len(train_fold))
    print("Validation images:", len(valid_fold))

    model = build()

    model.fit(
        dataset(train_fold, training=True),
        epochs=3,
        verbose=2
    )

    loss, accuracy = model.evaluate(
        dataset(valid_fold),
        verbose=0
    )

    scores.append({
        "fold": fold,
        "accuracy": accuracy,
        "loss": loss
    })

    print(f"Fold {fold} Accuracy: {accuracy:.2%}")

    pd.DataFrame(scores).to_csv(
        RESULTS / "cv.csv",
        index=False
    )

    del model


# Final cross-validation results
results = pd.DataFrame(scores)

mean_accuracy = results["accuracy"].mean()
std_accuracy = results["accuracy"].std(ddof=0)

print("\n========== FINAL RESULTS ==========")
print(results)

print(f"Mean Accuracy: {mean_accuracy:.2%}")
print(f"Standard Deviation: {std_accuracy:.2%}")

summary = {
    "folds": 5,
    "epochs_per_fold": 3,
    "mean_accuracy": float(mean_accuracy),
    "std_accuracy": float(std_accuracy)
}

(RESULTS / "cv_summary.json").write_text(
    json.dumps(summary, indent=4)
)

print("\nCross-validation completed successfully.")