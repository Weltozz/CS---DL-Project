import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.src.layers import BatchNormalization

from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models, Input, Model, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dropout
from tcn import TCN
from sklearn.metrics import r2_score, mean_absolute_error

def r2_metric(y_true, y_pred):
    ss_res = tf.reduce_sum(tf.square(y_true - y_pred))
    ss_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    return 1 - ss_res / ss_tot


def accuracy_5_metric(y_true, y_pred):
    # Erreur relative : |y_true - y_pred| / (|y_true| + epsilon) on evite les divisions par 0
    error = tf.abs((y_true - y_pred) / (tf.abs(y_true)))
    correct = tf.cast(error <= 0.05, tf.float32)

    return tf.reduce_mean(correct)

def create_sequences(values, sequence_length=60):
    X, y = [], []
    for i in range(sequence_length, len(values)):
        X_window = values[i - sequence_length: i]
        y_value = values[i]  # pour prédire la valeur suivante
        X.append(X_window)
        y.append(y_value)
    X = np.array(X)
    y = np.array(y).reshape(-1, 1)
    return X, y

def build_model(input_shape, learning_rate=0.0005):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        TCN(
            nb_filters=6,
            kernel_size=1,
            nb_stacks=1,
            dilations=[1, 2, 4, 8],
            padding='causal',
            dropout_rate=0.2,
            return_sequences=True # True : pour avoir en sortie des sequences à donner à un second modèle
        ),
        BatchNormalization(), # éviter une explosion de la loss
        layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='mean_squared_error',
        metrics=['mean_absolute_error', r2_metric, accuracy_5_metric]
    )
    return model

def main():
    df = pd.read_csv("./NASDAQ_100.csv")
    close_prices = df['close'].values.reshape(-1, 1)

    scaler = StandardScaler()
    scaled_prices = scaler.fit_transform(close_prices)

    sequence_length = 2000 # nombre de features pour l'entrainement (nombre de jours d'entrée)
    X, y = create_sequences(scaled_prices, sequence_length=sequence_length)
    print("Forme de X :", X.shape, "Forme de y :", y.shape)

    train_size = int(len(X) * 0.7)
    val_size = int(len(X) * 0.15)
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size + val_size], y[train_size:train_size + val_size]
    X_test, y_test = X[train_size + val_size:], y[train_size + val_size:]
    print("Train :", X_train.shape, y_train.shape)
    print("Val   :", X_val.shape, y_val.shape)
    print("Test  :", X_test.shape, y_test.shape)

    model = build_model(input_shape=(sequence_length, 1), learning_rate=0.0005)
    model.summary()

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True) # on arrete l'entrainement si la loss ne diminue plus sur plusieurs epochs
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1) # quand la loss ne diminue plus, on baisse le learning rate
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[
            early_stopping,
            reduce_lr
        ]
    )

    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Perte entraînement')
    plt.plot(history.history['val_loss'], label='Perte validation')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.title("Évolution de la perte")
    plt.legend()
    plt.show()

    test_loss = model.evaluate(X_test, y_test)
    print("Test Loss :", test_loss)
    y_pred_scaled = model.predict(X_test)
    y_test_unscaled = scaler.inverse_transform(y_test)
    y_pred_unscaled = scaler.inverse_transform(y_pred_scaled)

    n_plot = 100
    plt.figure(figsize=(10, 5))
    plt.plot(y_test_unscaled[:n_plot], label='Prix réel')
    plt.plot(y_pred_unscaled[:n_plot], label='Prix prédit')
    plt.xlabel("Index d'échantillon de test")
    plt.ylabel("Prix de clôture")
    plt.title("Prédictions TCN vs. Prix réels (100 premiers échantillons)")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()