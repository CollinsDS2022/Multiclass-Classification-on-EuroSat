import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras

(train_ds, val_ds, test_ds), info = tfds.load(
    "eurosat",
    split=["train[:80%]", "train[80%:90%]", "train[90%:]"],
    #     as_supervised=True,
    with_info=True,
)


def count_occurences(dataset, info):
    counter = dict.fromkeys(range(10), 0)
    for item in dataset:
        counter[int(item["label"])] = counter[int(item["label"])] + 1
    num_occurance = np.array(list(counter.values()))
    return num_occurance


train_occurences = count_occurences(train_ds, info)
print(f"train_ds:\t {train_occurences}")
val_occurences = count_occurences(val_ds, info)
print(f"val_ds:\t \t {val_occurences}")
test_occurences = count_occurences(test_ds, info)
print(f"test_ds:\t {test_occurences}")


def preprocess(item):
    return (item["image"] / 255, item["label"])


train_ds = train_ds.map(preprocess)
val_ds = val_ds.map(preprocess)
test_ds = test_ds.map(preprocess)

batch_size = 32
train_ds = train_ds.repeat()
train_ds = train_ds.shuffle(buffer_size=1024, seed=0)
train_ds = train_ds.batch(batch_size=batch_size)
train_ds = train_ds.prefetch(buffer_size=1)
##Val_Ds
val_ds = val_ds.batch(batch_size=batch_size)
val_ds = val_ds.prefetch(buffer_size=1)
##Test_DS
test_ds = test_ds.batch(batch_size=1)
test_ds = test_ds.prefetch(buffer_size=1)

epochs = 100

results = pd.DataFrame(
    columns=[
        "start_learning_rate",
        "width",
        "depth",
        "l2_weight",
        "train_loss",
        "val_loss",
        "train_acc",
        "val_acc",
    ]
)

start_learning_rates = [1e-5, 1e-4, 1e-3]
widths = [256, 512, 1024]
depths = [1, 2, 3]
l2_weights = [0, 1e-5, 1e-4]

for start_learning_rate in start_learning_rates:
    for width in widths:
        for depth in depths:
            for l2_weight in l2_weights:
                model = keras.models.Sequential()
                # your code (add input layer)
                model.add(
                    tf.keras.layers.InputLayer(input_shape=info.features["image"].shape)
                )
                model.add(tf.keras.layers.Flatten())
                for _ in range(depth):
                    model.add(
                        keras.layers.Dense(
                            units=width,
                            activation="relu",
                            kernel_regularizer=keras.regularizers.l2(l2_weight),
                        )
                    )
                # your code (add output layer)
                model.add(
                    tf.keras.layers.Dense(10, activation=tf.keras.activations.softmax)
                )
                scheduler = tf.keras.optimizers.schedules.PolynomialDecay(
                    start_learning_rate,
                    epochs * sum(train_occurences) // batch_size,
                    1e-8,
                    power=1.0,
                )
                model.compile(
                    loss=keras.losses.sparse_categorical_crossentropy,
                    optimizer=keras.optimizers.Adam(learning_rate=scheduler),
                    metrics=[keras.metrics.sparse_categorical_accuracy],
                )
                # your code (call fit function with verbose=0)
                model.fit(
                    train_ds,
                    epochs=epochs,
                    steps_per_epoch=math.ceil(sum(train_occurences) / batch_size),
                    validation_data=val_ds,
                    verbose=0,
                )
                train_loss, train_acc = model.evaluate(
                    train_ds, steps=np.sum(train_occurences) // batch_size
                )
                val_loss, val_acc = model.evaluate(val_ds)
                results_tmp = np.array(
                    [
                        start_learning_rate,
                        width,
                        depth,
                        l2_weight,
                        train_loss,
                        val_loss,
                        train_acc,
                        val_acc,
                    ]
                ).reshape(1, -1)
                results = results.append(
                    pd.DataFrame(data=results_tmp, columns=results.columns),
                    ignore_index=True,
                )
results.to_csv("results.csv")
