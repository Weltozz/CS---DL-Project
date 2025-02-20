import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Input, models, layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import BatchNormalization
from tcn import TCN


def R2(y_true, y_pred):
    ss_res = tf.reduce_sum(
        tf.square(y_true - y_pred)
    )
    ss_tot = tf.reduce_sum(
        tf.square(y_true - tf.reduce_mean(y_true))
    )

    return 1 - ss_res / ss_tot


def ACCURACY_5(y_true, y_pred):
    # Erreur relative : |y_true - y_pred| / (|y_true| + epsilon) on evite les divisions par 0
    error = tf.abs(
        (y_true - y_pred) / (tf.abs(y_true))
    )
    correct = tf.cast(
        error <= 0.05,
        tf.float32
    )

    return tf.reduce_mean(correct)
def NN_MODEL(input_shape, learning_rate=0.0005):
    model = models.Sequential([
        layers.Input(shape=input_shape),

        TCN(
            nb_filters=2,
            kernel_size=1,
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

def create_sequences(values, sequence_length=60):
    X, y = [], []
    for i in range(sequence_length, len(values)):
        X_window = values[i - sequence_length : i]
        y_value = values[i]
        X.append(X_window)
        y.append(y_value)
    X = np.array(X)
    y = np.array(y).reshape(-1, 1)
    return X, y

def main():
    df = pd.read_csv("/Users/welto/Library/CloudStorage/OneDrive-CentraleSupelec/3A/DL/Project/NASDAQ_100.csv")
    close_prices = df['close'].values.reshape(-1, 1)
    
    scaler = MinMaxScaler()
    scaled_prices = scaler.fit_transform(close_prices)
    
    sequence_length = 2000
    X, y = create_sequences(scaled_prices, sequence_length=sequence_length)
    print("Shape X :", X.shape, "Shape y :", y.shape)

    train_size = int(len(X) * 0.7)
    val_size = int(len(X) * 0.15)
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
    X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]
    print("Train :", X_train.shape, y_train.shape)
    print("Val  :", X_val.shape, y_val.shape)
    print("Test        :", X_test.shape, y_test.shape)
    
    model = NN_MODEL(
        input_shape=(sequence_length, 1),
        learning_rate=0.0005
    )
    model.summary()
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=16,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, reduce_lr]
    )
    
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Train loss')
    plt.plot(history.history['val_loss'], label='Val loss')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.title('Loss over epochs')
    plt.legend()
    plt.show()
    
    test_loss = model.evaluate(X_test, y_test)
    print("Test Loss :", test_loss)
    
    y_pred_scaled = model.predict(X_test)
    y_test_unscaled = scaler.inverse_transform(y_test)
    y_pred_unscaled = scaler.inverse_transform(y_pred_scaled)
    
    n_plot = 100
    plt.figure(figsize=(10, 5))
    plt.plot(y_test_unscaled[:n_plot], label='Y test')
    plt.plot(y_pred_unscaled[:n_plot], label='Y pred')
    plt.xlabel("Index")
    plt.ylabel("Close price")
    plt.title("Y test vs Y pred")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()

