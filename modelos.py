import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


# =========================
# FUNCIONES GENERALES
# =========================
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


def mostrar_resultado_individual(nombre_modelo, test, predicciones, info_extra=None):
    st.markdown(f"## {nombre_modelo}")

    metricas = calcular_metricas(test, predicciones, nombre_modelo)

    col1, col2 = st.columns(2)
    col1.metric("MAE", metricas["MAE"])
    col2.metric("RMSE", metricas["RMSE"])

    if info_extra is not None:
        st.write("Configuración utilizada:")
        st.json(info_extra)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(test.index, test.values, marker="o", label="Valores reales")
    ax.plot(test.index, predicciones, marker="o", label=f"Predicción - {nombre_modelo}")
    ax.set_title(f"{nombre_modelo}: Predicción vs valores reales")
    ax.set_xlabel("Índice")
    ax.set_ylabel("Valor")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    st.divider()

    return metricas


# =========================
# SERIES DE TIEMPO
# =========================
def ejecutar_arima(train, test):
    orden = (1, 1, 1)
    modelo = ARIMA(train, order=orden)
    ajuste = modelo.fit()
    pred = ajuste.forecast(steps=len(test))
    return np.array(pred), {"order": orden}


def ejecutar_arima_calibrado(train, test):
    mejor_rmse = float("inf")
    mejor_orden = None
    mejor_pred = None

    combinaciones = [
        (0, 1, 1),
        (1, 1, 1),
        (1, 1, 2),
        (2, 1, 1),
        (2, 1, 2)
    ]

    for orden in combinaciones:
        try:
            modelo = ARIMA(train, order=orden)
            ajuste = modelo.fit()
            pred = ajuste.forecast(steps=len(test))
            rmse = np.sqrt(mean_squared_error(test, pred))

            if rmse < mejor_rmse:
                mejor_rmse = rmse
                mejor_orden = orden
                mejor_pred = np.array(pred)
        except Exception:
            continue

    if mejor_pred is None:
        raise ValueError("No se pudo calibrar ARIMA con las combinaciones definidas.")

    return mejor_pred, {"best_order": mejor_orden}


def ejecutar_holt_winters(train, test):
    config = {
        "trend": "add",
        "seasonal": None,
        "seasonal_periods": None
    }

    modelo = ExponentialSmoothing(
        train,
        trend=config["trend"],
        seasonal=config["seasonal"],
        seasonal_periods=config["seasonal_periods"]
    )
    ajuste = modelo.fit()
    pred = ajuste.forecast(len(test))
    return np.array(pred), config


def ejecutar_holt_winters_calibrado(train, test):
    mejor_rmse = float("inf")
    mejor_config = None
    mejor_pred = None

    configuraciones = [
        {"trend": "add", "seasonal": None, "seasonal_periods": None},
        {"trend": "add", "seasonal": "add", "seasonal_periods": 4},
        {"trend": "add", "seasonal": "add", "seasonal_periods": 6},
        {"trend": "add", "seasonal": "add", "seasonal_periods": 12},
        {"trend": "add", "seasonal": "mul", "seasonal_periods": 4},
        {"trend": "add", "seasonal": "mul", "seasonal_periods": 6},
        {"trend": "add", "seasonal": "mul", "seasonal_periods": 12},
    ]

    for config in configuraciones:
        try:
            if config["seasonal"] is not None and config["seasonal_periods"] is not None:
                if len(train) < 2 * config["seasonal_periods"]:
                    continue

            modelo = ExponentialSmoothing(
                train,
                trend=config["trend"],
                seasonal=config["seasonal"],
                seasonal_periods=config["seasonal_periods"]
            )
            ajuste = modelo.fit()
            pred = ajuste.forecast(len(test))
            rmse = np.sqrt(mean_squared_error(test, pred))

            if rmse < mejor_rmse:
                mejor_rmse = rmse
                mejor_config = config
                mejor_pred = np.array(pred)
        except Exception:
            continue

    if mejor_pred is None:
        raise ValueError("No se pudo calibrar Holt-Winters con las configuraciones definidas.")

    return mejor_pred, mejor_config


def ejecutar_deep_learning(train, test, pasos=5, epochs=20):
    try:
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense
    except Exception as e:
        raise ImportError(f"No se pudo importar TensorFlow/Keras. Detalle: {e}")

    train = np.array(train, dtype=float)
    test = np.array(test, dtype=float)

    if len(train) <= pasos:
        raise ValueError(
            f"No hay suficientes datos para Deep Learning. "
            f"Se requieren más de {pasos} registros de entrenamiento."
        )

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train.reshape(-1, 1))

    X_train, y_train = crear_secuencias(train_scaled, pasos)

    if len(X_train) == 0:
        raise ValueError("No se pudieron crear secuencias para LSTM.")

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

    modelo = Sequential()
    modelo.add(LSTM(50, input_shape=(pasos, 1)))
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

    return predicciones, {"pasos": pasos, "epochs": epochs}


def detectar_columnas_numericas(df):
    columnas_validas = []

    for col in df.columns:
        serie_convertida = pd.to_numeric(df[col], errors="coerce")
        if serie_convertida.notna().sum() >= max(10, int(len(df) * 0.5)):
            columnas_validas.append(col)

    return columnas_validas


def mostrar_modelos_predictivos(df, modelos_seleccionados):
    st.subheader("Implementación de modelos predictivos")
    st.caption("Benchmarking individual y comparativo de modelos de series de tiempo.")

    columnas_numericas = detectar_columnas_numericas(df)

    if not columnas_numericas:
        st.error("No se detectaron columnas aptas para series de tiempo.")
        st.write("Revise el formato del CSV y confirme que exista al menos una columna con valores numéricos.")
        return

    serie_col = st.selectbox(
        "Seleccione la columna de la serie temporal",
        columnas_numericas
    )

    serie = pd.to_numeric(df[serie_col], errors="coerce").dropna().reset_index(drop=True)

    if len(serie) < 20:
        st.warning("La serie debe tener al menos 20 registros para ejecutar los modelos.")
        return

    st.write(f"Cantidad de registros disponibles en la serie: {len(serie)}")

    test_size = st.slider(
        "Tamaño del conjunto de prueba",
        min_value=5,
        max_value=min(50, max(5, len(serie) // 2)),
        value=min(10, max(5, len(serie) // 4))
    )

    pasos_lstm = st.slider(
        "Ventana de tiempo para Deep Learning",
        min_value=3,
        max_value=min(10, max(3, len(serie) // 4)),
        value=5
    )

    epochs_lstm = st.slider(
        "Epochs para Deep Learning",
        min_value=5,
        max_value=100,
        value=20
    )

    ejecutar = st.button("▶ Ejecutar benchmarking de series de tiempo")

    if ejecutar:
        train = serie[:-test_size]
        test = serie[-test_size:]

        resultados = []

        if "Deep Learning" in modelos_seleccionados:
            try:
                pred_dl, info_dl = ejecutar_deep_learning(
                    train, test, pasos=pasos_lstm, epochs=epochs_lstm
                )
                resultados.append(
                    mostrar_resultado_individual("Deep Learning", test, pred_dl, info_dl)
                )
            except Exception as e:
                st.error(f"Error al ejecutar Deep Learning: {e}")

        if "Holt-Winters" in modelos_seleccionados:
            try:
                pred_hw, info_hw = ejecutar_holt_winters(train, test)
                resultados.append(
                    mostrar_resultado_individual("Holt-Winters", test, pred_hw, info_hw)
                )
            except Exception as e:
                st.error(f"Error al ejecutar Holt-Winters: {e}")

        if "Holt-Winters-Calibrado" in modelos_seleccionados:
            try:
                pred_hw_cal, info_hw_cal = ejecutar_holt_winters_calibrado(train, test)
                resultados.append(
                    mostrar_resultado_individual(
                        "Holt-Winters-Calibrado",
                        test,
                        pred_hw_cal,
                        info_hw_cal
                    )
                )
            except Exception as e:
                st.error(f"Error al ejecutar Holt-Winters-Calibrado: {e}")

        if "ARIMA" in modelos_seleccionados:
            try:
                pred_arima, info_arima = ejecutar_arima(train, test)
                resultados.append(
                    mostrar_resultado_individual("ARIMA", test, pred_arima, info_arima)
                )
            except Exception as e:
                st.error(f"Error al ejecutar ARIMA: {e}")

        if "ARIMA-Calibrado" in modelos_seleccionados:
            try:
                pred_arima_cal, info_arima_cal = ejecutar_arima_calibrado(train, test)
                resultados.append(
                    mostrar_resultado_individual(
                        "ARIMA-Calibrado",
                        test,
                        pred_arima_cal,
                        info_arima_cal
                    )
                )
            except Exception as e:
                st.error(f"Error al ejecutar ARIMA-Calibrado: {e}")

        if not resultados:
            st.warning("No se pudo ejecutar ningún modelo.")
            return

        df_resultados = pd.DataFrame(resultados).sort_values(by="RMSE")

        st.subheader("Resumen general del benchmarking")
        st.dataframe(df_resultados, use_container_width=True)

        mejor_modelo = df_resultados.iloc[0]["Modelo"]
        st.success(f"El mejor modelo según RMSE fue: {mejor_modelo}")


# =========================
# CLASIFICACIÓN
# =========================
def mostrar_modelos_clasificacion(df, modelos_seleccionados, kfold, threshold):
    st.subheader("Modelos de Clasificación")
    st.caption("Comparación de modelos utilizando AUC y validación cruzada.")

    columnas_numericas = df.select_dtypes(include="number").columns.tolist()

    if len(columnas_numericas) < 2:
        st.warning("Se necesitan al menos dos variables numéricas.")
        return

    target = st.selectbox(
        "Seleccione la variable objetivo",
        columnas_numericas
    )

    X = df[columnas_numericas].drop(columns=[target])
    y = df[target]
    y = (y > y.median()).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    modelos = {
        "Logistic regression": LogisticRegression(max_iter=1000),
        "Random forest": RandomForestClassifier(random_state=42),
        "SVM": SVC(probability=True)
    }

    resultados = []
    curvas_roc = {}

    for nombre in modelos_seleccionados:
        model = modelos[nombre]

        try:
            model.fit(X_train, y_train)
            y_prob = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_prob)

            scores = cross_val_score(
                model,
                X,
                y,
                cv=kfold,
                scoring="roc_auc"
            )

            resultados.append({
                "Modelo": nombre,
                "AUC": round(auc, 4),
                "CV_AUC": round(scores.mean(), 4)
            })

            fpr, tpr, _ = roc_curve(y_test, y_prob)
            curvas_roc[nombre] = (fpr, tpr)

        except Exception as e:
            st.error(f"Error al ejecutar {nombre}: {e}")

    if not resultados:
        st.warning("No se pudo ejecutar ningún modelo de clasificación.")
        return

    df_resultados = pd.DataFrame(resultados).sort_values(by="AUC", ascending=False)

    st.subheader("Comparación de modelos")
    st.dataframe(df_resultados, use_container_width=True)

    mejor = df_resultados.iloc[0]["Modelo"]
    st.success(f"El mejor modelo según AUC fue: {mejor}")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(df_resultados["Modelo"], df_resultados["AUC"])
    ax.set_xlabel("AUC")
    ax.set_title("Comparación de modelos")
    st.pyplot(fig)

    st.subheader("Curvas ROC")
    fig2, ax2 = plt.subplots(figsize=(8, 5))

    for nombre, (fpr, tpr) in curvas_roc.items():
        ax2.plot(fpr, tpr, label=nombre)

    ax2.plot([0, 1], [0, 1], "--")
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.set_title("Curvas ROC")
    ax2.legend()

    st.pyplot(fig2)
