import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


def calcular_metricas(y_real, y_pred, nombre_modelo):
    mae = mean_absolute_error(y_real, y_pred)
    rmse = np.sqrt(mean_squared_error(y_real, y_pred))

    return {
        "Modelo": nombre_modelo,
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4)
    }


def crear_secuencias(data, pasos=5):
    X, y = [], []

    for i in range(len(data) - pasos):
        X.append(data[i:i + pasos])
        y.append(data[i + pasos])

    return np.array(X), np.array(y)


def ejecutar_arima(train, test):
    modelo = ARIMA(train, order=(1, 1, 1))
    ajuste = modelo.fit()
    pred = ajuste.forecast(steps=len(test))
    return np.array(pred)


def ejecutar_holt_winters(train, test):
    modelo = ExponentialSmoothing(
        train,
        trend="add",
        seasonal=None
    )
    ajuste = modelo.fit()
    pred = ajuste.forecast(len(test))
    return np.array(pred)


def ejecutar_deep_learning(train, test, pasos=5, epochs=20):
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(np.array(train).reshape(-1, 1))

    X_train, y_train = crear_secuencias(train_scaled, pasos)

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

    modelo = Sequential()
    modelo.add(LSTM(50, activation="relu", input_shape=(pasos, 1)))
    modelo.add(Dense(1))

    modelo.compile(optimizer="adam", loss="mse")
    modelo.fit(X_train, y_train, epochs=epochs, verbose=0)

    entrada = train_scaled[-pasos:].flatten().tolist()
    predicciones = []

    for _ in range(len(test)):
        x_input = np.array(entrada[-pasos:]).reshape((1, pasos, 1))
        yhat = modelo.predict(x_input, verbose=0)[0][0]
        predicciones.append(yhat)
        entrada.append(yhat)

    predicciones = scaler.inverse_transform(
        np.array(predicciones).reshape(-1, 1)
    ).flatten()

    return predicciones


def mostrar_modelos_predictivos(df):
    st.subheader("Implementacion de modelos predictivos")
    st.caption("Compare el desempeño de ARIMA, Holt-Winters y Deep Learning sobre una serie numérica.")

    columnas_numericas = df.select_dtypes(include="number").columns.tolist()

    if not columnas_numericas:
        st.warning("El dataset no contiene columnas numéricas para aplicar modelos de series de tiempo.")
        return

    serie_col = st.selectbox(
        "Seleccione la columna de la serie temporal",
        columnas_numericas
    )

    serie = df[serie_col].dropna().reset_index(drop=True)

    if len(serie) < 20:
        st.warning("La serie debe tener al menos 20 registros para ejecutar los modelos.")
        return

    st.write(f"Cantidad de registros disponibles: {len(serie)}")

    test_size = st.slider(
        "Tamaño del conjunto de prueba",
        min_value=5,
        max_value=min(50, len(serie) // 2),
        value=10
    )

    modelos_seleccionados = st.multiselect(
        "Seleccione los modelos a ejecutar",
        ["ARIMA", "Holt-Winters", "Deep Learning"],
        default=["ARIMA", "Holt-Winters", "Deep Learning"]
    )

    pasos_lstm = st.slider(
        "Ventana de tiempo para Deep Learning",
        min_value=3,
        max_value=10,
        value=5
    )

    epochs_lstm = st.slider(
        "Epochs para Deep Learning",
        min_value=5,
        max_value=100,
        value=20
    )

    train = serie[:-test_size]
    test = serie[-test_size:]

    if st.button("🚀 Ejecutar modelos predictivos"):
        resultados = []
        predicciones = {}

        # ARIMA
        if "ARIMA" in modelos_seleccionados:
            try:
                pred_arima = ejecutar_arima(train, test)
                resultados.append(calcular_metricas(test, pred_arima, "ARIMA"))
                predicciones["ARIMA"] = pred_arima
            except Exception as e:
                st.error(f"Error al ejecutar ARIMA: {e}")

        # Holt-Winters
        if "Holt-Winters" in modelos_seleccionados:
            try:
                pred_hw = ejecutar_holt_winters(train, test)
                resultados.append(calcular_metricas(test, pred_hw, "Holt-Winters"))
                predicciones["Holt-Winters"] = pred_hw
            except Exception as e:
                st.error(f"Error al ejecutar Holt-Winters: {e}")

        # Deep Learning
        if "Deep Learning" in modelos_seleccionados:
            try:
                pred_dl = ejecutar_deep_learning(
                    train,
                    test,
                    pasos=pasos_lstm,
                    epochs=epochs_lstm
                )
                resultados.append(calcular_metricas(test, pred_dl, "Deep Learning"))
                predicciones["Deep Learning"] = pred_dl
            except Exception as e:
                st.error(f"Error al ejecutar Deep Learning: {e}")

        if not resultados:
            st.warning("No se pudo ejecutar ningún modelo.")
            return

        df_resultados = pd.DataFrame(resultados).sort_values(by="RMSE")

        st.subheader("Comparación de resultados")
        st.dataframe(df_resultados, use_container_width=True)

        mejor_modelo = df_resultados.iloc[0]["Modelo"]
        st.success(f"El modelo con mejor desempeño según RMSE fue: {mejor_modelo}")

        st.subheader("Predicción vs valores reales")

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(test.index, test.values, marker="o", label="Valores reales")

        for nombre, preds in predicciones.items():
            ax.plot(test.index, preds, marker="o", label=nombre)

        ax.set_title(f"Comparación de predicciones - {serie_col}")
        ax.set_xlabel("Índice")
        ax.set_ylabel("Valor")
        ax.legend()
        ax.grid(True, alpha=0.3)

        st.pyplot(fig)

        st.subheader("Conclusión automática")
        st.write(
            f"Se compararon {len(df_resultados)} modelos predictivos sobre la serie **{serie_col}**. "
            f"Según el criterio de **RMSE**, el mejor modelo fue **{mejor_modelo}**."
        )
