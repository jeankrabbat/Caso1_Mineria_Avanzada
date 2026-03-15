import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from modelos import mostrar_modelos_predictivos, mostrar_modelos_clasificacion

st.set_page_config(layout="wide")

st.title("Plataforma de Benchmarking Analítico")
st.markdown("### Evaluación avanzada de modelos predictivos")
st.write(
    "Entorno interactivo para explorar datos, configurar algoritmos y comparar el "
    "desempeño de modelos de clasificación y series de tiempo."
)

st.info(
    "Seleccione un conjunto de datos para iniciar el análisis y configure el flujo "
    "de evaluación según el tipo de modelo requerido."
)

st.divider()

# ---------------- SIDEBAR ----------------
st.sidebar.header("Configuración del modelo")
archivo = st.sidebar.file_uploader("Cargue acá su dataset", type=["csv"])

if archivo is not None:
    df = pd.read_csv(archivo)

    # Intenta convertir columnas tipo texto a numéricas cuando sea posible
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")

    st.success("Dataset cargado correctamente")

    st.sidebar.subheader("Seleccione el tipo")
    tipo_modelo = st.sidebar.selectbox(
        "Seleccione el tipo",
        ["Clasificacion", "Series de tiempo"]
    )

    if tipo_modelo == "Clasificacion":
        modelos_disponibles = ["Logistic regression", "Random forest", "SVM"]
    else:
        modelos_disponibles = [
            "Deep Learning",
            "Holt-Winters",
            "Holt-Winters-Calibrado",
            "ARIMA",
            "ARIMA-Calibrado"
        ]

    st.sidebar.subheader("Seleccione los algoritmos")
    modelos_seleccionados = st.sidebar.multiselect(
        "Algoritmos",
        modelos_disponibles,
        default=modelos_disponibles
    )

    st.sidebar.subheader("Parámetros")
    if tipo_modelo == "Clasificacion":
        kfold = st.sidebar.slider("Número de folds", 2, 10, 5)
        threshold = st.sidebar.slider("Probabilidad de corte", 0.0, 1.0, 0.5)
    else:
        kfold = None
        threshold = None
        st.sidebar.info("Los parámetros de series de tiempo se configuran dentro de la pestaña de benchmarking.")

    tab1, tab2 = st.tabs([
        "Exploración de datos",
        "Benchmarking de modelos"
    ])

    with tab1:
        col1, col2, col3 = st.columns(3)

        columnas_numericas = df.select_dtypes(include="number").columns.tolist()

        col1.metric("Filas", df.shape[0])
        col2.metric("Columnas", df.shape[1])
        col3.metric("Variables numéricas", len(columnas_numericas))

        st.subheader("Vista previa del dataset")
        st.dataframe(df.head(), use_container_width=True)

        st.subheader("Tipos de datos")
        st.dataframe(
            pd.DataFrame({
                "Columna": df.columns,
                "Tipo": [str(df[col].dtype) for col in df.columns]
            }),
            use_container_width=True
        )

        if len(columnas_numericas) > 0:
            st.subheader("Estadísticas descriptivas")
            st.dataframe(df[columnas_numericas].describe(), use_container_width=True)

            columna_numerica = st.selectbox(
                "Seleccione una variable numérica",
                columnas_numericas
            )

            col_a, col_b = st.columns(2)

            with col_a:
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.hist(df[columna_numerica].dropna(), bins=20)
                ax.set_title(f"Distribución de {columna_numerica}")
                ax.set_xlabel(columna_numerica)
                ax.set_ylabel("Frecuencia")
                st.pyplot(fig)

            with col_b:
                fig2, ax2 = plt.subplots(figsize=(6, 4))
                ax2.boxplot(df[columna_numerica].dropna(), vert=False)
                ax2.set_title(f"Boxplot de {columna_numerica}")
                ax2.set_xlabel(columna_numerica)
                st.pyplot(fig2)

            if len(columnas_numericas) > 1:
                st.subheader("Matriz de correlación")
                corr = df[columnas_numericas].corr()
                fig3, ax3 = plt.subplots(figsize=(8, 6))
                sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax3)
                st.pyplot(fig3)
        else:
            st.warning("No se detectaron variables numéricas en el dataset.")

    with tab2:
        st.subheader("Benchmarking de modelos")

        if not modelos_seleccionados:
            st.warning("Seleccione al menos un algoritmo en el panel lateral.")
        else:
            if tipo_modelo == "Clasificacion":
                mostrar_modelos_clasificacion(
                    df=df,
                    modelos_seleccionados=modelos_seleccionados,
                    kfold=kfold,
                    threshold=threshold
                )
            else:
                mostrar_modelos_predictivos(
                    df=df,
                    modelos_seleccionados=modelos_seleccionados
                )

else:
    st.warning("Por favor, cargue un archivo CSV para comenzar.")
