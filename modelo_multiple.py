import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def modelo_ventas_multiple():
# Cargar dataset
    df = pd.read_csv('data/dataset_sintetico.csv')
# Seleccionar features y target
    X = df[['Temperatura', 'Promocion', 'Fin_de_Semana']]
    y = df['Ventas']
    # Normalizar
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_x.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))
    # Split datos
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled,test_size=0.2)
    # Crear modelo
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(100, activation='relu', input_shape=(3,)),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])
    # Compilar y entrenar
    model.compile(optimizer='adam', loss='mean_squared_error')
    history = model.fit(X_train, y_train, epochs=100, batch_size=32,
    validation_split=0.2, verbose=1)
    return model, history, scaler_x, scaler_y, X_test, y_test
# EJECUTAR
modelo_multiple, hist_multiple, scaler_x, scaler_y, X_test, y_test = modelo_ventas_multiple()