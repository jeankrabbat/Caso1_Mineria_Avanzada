import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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


st.sidebar.header("Panel de configuracion")
st.sidebar.subheader("Carga de datos")
st.sidebar.caption("Importe un archivo en formato CSV para comenzar.")
archivo = st.sidebar.file_uploader("Cargue aca su dataset", type = ["csv"])

if archivo is not None:
    df = pd.read_csv(archivo)

    st.success("Dataset cargado correctamente")

    col1, col2, col3 = st.columns(3)

    col1.metric("Filas", df.shape[0])
    col2.metric("Columnas", df.shape[1])
    col3.metric("Variables numéricas", df.select_dtypes(include="number").shape[1])

    #Mostrar dataset
    st.subheader("Vista previa del dataset")
    st.caption("Se muestran las primeras filas del dataset cargado")
    st.dataframe(df.head(), use_container_width=True)

    st.subheader("Estadisticas descriptivas:")
    st.dataframe(df.describe(), use_container_width= True)

    st.subheader("Visualización de variables numéricas")
    st.caption("Seleccione una variable para explorar su distribución.")

    columnas_numericas = df.select_dtypes(include="number").columns

    col1, col2 = st.columns([2,3])

    with col1:
        columna_numerica = st.selectbox(
            "Seleccione una variable numérica",
            columnas_numericas
        )
    col1, col2 = st.columns(2)

    # HISTOGRAMA
    with col1:
        st.subheader("Distribución")

        fig, ax = plt.subplots(figsize=(5,3), facecolor="#0E1117")

        ax.hist(df[columna_numerica].dropna(), bins=20)

        ax.set_title(f"{columna_numerica}", color="white")
        ax.set_xlabel(columna_numerica, color="white")
        ax.set_ylabel("Frecuencia", color="white")

        ax.tick_params(colors="white")
        ax.set_facecolor("#0E1117")

        st.pyplot(fig)

    # BOXPLOT
    with col2:
        st.subheader("Outliers")

        fig2, ax2 = plt.subplots(figsize=(5,3), facecolor="#0E1117")

        ax2.boxplot(df[columna_numerica].dropna(), vert=False)

        ax2.set_title(f"{columna_numerica}", color="white")
        ax2.set_xlabel(columna_numerica, color="white")

        ax2.tick_params(colors="white")
        ax2.set_facecolor("#0E1117")

        st.pyplot(fig2)

    st.subheader("Matriz de correlación")

    corr = df.corr()

    fig, ax = plt.subplots(figsize=(8,6), facecolor="#0E1117")

    sns.heatmap(
        corr,
        annot=True,
        cmap="coolwarm",
        ax=ax
    )

    ax.tick_params(colors="white")
    ax.set_facecolor("#0E1117")

    st.pyplot(fig)

    #Tipo de modelo

    st.sidebar.divider()
    st.sidebar.subheader("Configuración del modelo")
    st.sidebar.caption("Seleccione el tipo de análisis y el algoritmo que desea utilizar.")

    tipo_modelo = st.sidebar.selectbox(
        "Seleccione el tipo",
        ["Clasificacion","Series de tiempo"]
    )

    #Despues de seleccionar el tipo de problema

    if tipo_modelo == "Clasificacion":
        modelo = st.sidebar.selectbox(
            "Seleccione el algoritmo",
            ["Logistic regression", "Random forest", "SVM"]
        )
    if tipo_modelo == "Series de tiempo":
        modelo = st.sidebar.selectbox(
            "Seleccione el algoritmo",
            ["ARIMA", "Holt-Winters", "Deep Learning"]
        )

    #Parametros
    st.sidebar.divider()
    st.sidebar.subheader("Parametros")

    kfold = st.sidebar.slider("Numero de folds",2, 10, 5)

    if tipo_modelo == "Clasificacion":
        threshold = st.sidebar.slider("Probabilidad de corte", 0.0, 1.0, 0.5)
    else:
        threshold = None
        st.sidebar.info("La probabilidad de corte aplica únicamente a modelos de clasificación.") 

    st.sidebar.divider()

    if st.sidebar.button("🚀 Ejecutar modelo"):
        st.success("Modelo ejecutado correctamente")
        st.subheader("Configuración seleccionada:")

        st.write("Tipo de problema:", tipo_modelo)
        st.write("Algortimo:", modelo)
        st.write("Numero de folds:", kfold)
        if threshold is not None:
            st.write("Probabilidad de corte:", threshold)
