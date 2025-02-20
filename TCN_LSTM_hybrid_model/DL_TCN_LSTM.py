#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Input, Model, layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

def TCN_SimpleBlock(input_shape, num_filters=32, kernel_size=2, dropout_rate=0.2):
    inputs = tf.keras.Input(shape=input_shape)

    x = layers.Conv1D(
        filters=num_filters,
        kernel_size=kernel_size,
        padding='causal',
        activation='relu'
    )(inputs)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Conv1D(
        filters=num_filters,
        kernel_size=kernel_size,
        padding='causal',
        activation='relu'
    )(x)
    x = layers.Dropout(dropout_rate)(x)

    # Connexion résiduelle
    if inputs.shape[-1] != x.shape[-1]:
        inputs_res = layers.Conv1D(num_filters, kernel_size=1, padding='same')(inputs)
    else:
        inputs_res = inputs

    x = layers.Add()([inputs_res, x])
    x = layers.Activation('relu')(x)

    model = Model(inputs, x, name="TCN")
    return model

def build_hybrid_model(input_shape, tcn_filters=32, lstm_units=50, dropout_rate=0.2, learning_rate=0.0005):
    inputs = Input(shape=input_shape)

    tcn_out = TCN_SimpleBlock(input_shape=input_shape,
                              num_filters=tcn_filters,
                              kernel_size=2,
                              dropout_rate=dropout_rate)(inputs)

    lstm_out = layers.LSTM(lstm_units)(tcn_out)

    outputs = layers.Dense(1, activation='linear')(lstm_out)

    model = Model(inputs, outputs, name="Hybrid_TCN_LSTM_Model")
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
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
    df = pd.read_csv("/Datasets/NASDAQ_100.csv")
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
    
    model = build_hybrid_model(
        input_shape=(sequence_length, 1),
        tcn_filters=32,
        lstm_units=50,
        dropout_rate=0.2,
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