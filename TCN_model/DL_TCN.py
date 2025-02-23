import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import ta

from keras.src.layers import BatchNormalization
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras import layers, models, Input, Model, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dropout, Conv1D, BatchNormalization
from tcn import TCN
from sklearn.metrics import r2_score, mean_absolute_error

def R2(y_true, y_pred):
    ss_res = tf.reduce_sum(
        tf.square(y_true - y_pred)
    )
    ss_tot = tf.reduce_sum(
        tf.square(y_true - tf.reduce_mean(y_true))
    )
    
    return 1 - ss_res / (ss_tot + tf.keras.backend.epsilon()) # sss_tot


def ACCURACY_5(y_true, y_pred):
    # Erreur relative : |y_true - y_pred| / (|y_true| + epsilon) on evite les divisions par 0
    error = tf.abs(
        (y_true - y_pred) / (tf.abs(y_true) + tf.keras.backend.epsilon())
    )
    correct = tf.cast(
        error <= 0.05,
        tf.float32
    )

    return tf.reduce_mean(correct)

def CREATE_SEQUENCES(data, sequence_length, target_column='close'):
    X, y, = [], []
    for i in range(sequence_length, len(data)):
        features = data.iloc[i - sequence_length:i].values
        X.append(features)

        targets = data.iloc[i][target_column]
        y.append(targets)

    return np.array(X), np.array(y)

def NN_MODEL(input_shape, learning_rate=5e-4):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        TCN(
            nb_filters=8,
            kernel_size=4,
            nb_stacks=1,
            dilations=[1, 2, 4, 8],
            padding='causal',
            dropout_rate=0.2,
            return_sequences=False
        ),
        BatchNormalization(),

        layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss='mean_squared_error',
        metrics=[
            'mean_absolute_error', 
            R2, 
            ACCURACY_5
        ]
    )
    
    return model

def main():
    df = pd.read_csv("Datasets/NASDAQ_100.csv")
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    df['bollinger_upper'] = df['sma_20'] + 2 * df['close'].rolling(window=20).std()
    df['bollinger_uower'] = df['sma_20'] - 2 * df['close'].rolling(window=20).std()

    df.ffill(inplace=True)
    df = df.drop(columns=['date'])

    scaler_features = MinMaxScaler()
    target_scaler = StandardScaler()

    target = pd.DataFrame(
        target_scaler.fit_transform(df['close'].values.reshape(-1, 1))
    )

    df = pd.concat(
        [
            pd.DataFrame(
                scaler_features.fit_transform(
                    df.filter(
                        items=[col for col in df.columns if col != 'close']
                    )
                ),
                columns=df.filter(
                    items=[col for col in df.columns if col != 'close']
                ).columns
            ),
            target,
        ], axis=1
    ).dropna()
    df.rename(
        columns={0: 'close'},
        inplace=True
    )

    sequence_length = 2000 # nombre de features pour l'entrainement (nombre de jours d'entrée)

    X, y = CREATE_SEQUENCES(df, sequence_length=sequence_length)
    print("X shape :", X.shape, "y shape :", y.shape)

    train_size = int(len(X) * 0.7)
    val_size = int(len(X) * 0.15)

    X_train = X[:train_size]
    X_val = X[train_size:train_size + val_size]
    X_test = X[train_size + val_size:]

    y_train = y[:train_size]
    y_val = y[train_size:train_size + val_size]
    y_test = y[train_size + val_size:]

    print("Train shapes :", X_train.shape, y_train.shape)
    print("Val shapes  :", X_val.shape, y_val.shape)
    print("Test shapes :", X_test.shape, y_test.shape)

    input_shape = (sequence_length, X.shape[-1])
    model = NN_MODEL(input_shape=input_shape)
    model.summary()

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True
    ) # on arrete l'entrainement si la loss ne diminue plus sur plusieurs epochs

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        verbose=1
    ) # quand la loss ne diminue plus, on baisse le learning rate

    history = model.fit(
        X_train, y_train,
        epochs=80,
        batch_size=16,
        validation_data=(X_val, y_val),
        callbacks=[
            early_stopping,
            reduce_lr
        ]
    )

    plt.figure(figsize=(10, 5))

    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history['val_loss'], label='Validation loss')

    plt.xlabel('Epoch')
    plt.ylabel('MSE')

    plt.title("Loss over epochs (TCN)")

    plt.legend()
    plt.show()

    test_loss = model.evaluate(X_test, y_test)
    print("Test Loss :", test_loss)

    y_pred_scaled = model.predict(X_test)
    y_test_unscaled = target_scaler.inverse_transform(y_test.reshape(-1,1))
    y_pred_unscaled = target_scaler.inverse_transform(y_pred_scaled.reshape(-1,1))

    n_plot = 100
    plt.figure(figsize=(10, 5))
    plt.plot(y_test_unscaled[:n_plot], label='y real')
    plt.plot(y_pred_unscaled[:n_plot], label='y predicted')
    plt.xlabel("Index")
    plt.ylabel("Close price")
    plt.title("TCN Predictions vs Real values")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()