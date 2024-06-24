import streamlit as st
import numpy as np
import pandas as pd
import pywt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Título de la aplicación
st.title("Clasificación de Muestras de Perforación con Teselación Wavelet")

# Carga de datos
uploaded_file = st.file_uploader("Sube tu archivo CSV con datos de perforación", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Datos cargados con éxito:")
    st.write(data.head())

    # Selección de variables para análisis
    variables = st.multiselect("Selecciona las variables para el análisis", data.columns)
    
    if variables:
        # Estandarización de los datos
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data[variables])

        # Aplicación de Wavelet Transform
        coeffs = pywt.wavedec(data_scaled, 'db1', level=2)
        wavelet_features = np.hstack(coeffs)

        # Selección del método de clustering
        method = st.selectbox("Selecciona el método de clustering", ["KMeans", "Agglomerative Clustering"])

        if method == "KMeans":
            n_clusters = st.slider("Número de clusters", 2, 10, 3)
            model = KMeans(n_clusters=n_clusters)
        elif method == "Agglomerative Clustering":
            n_clusters = st.slider("Número de clusters", 2, 10, 3)
            model = AgglomerativeClustering(n_clusters=n_clusters)

        # Ajuste del modelo y predicciones
        labels = model.fit_predict(wavelet_features)
        data['Cluster'] = labels

        # Visualización de resultados
        st.write("Resultados del clustering:")
        st.write(data[['Cluster']].value_counts())

        # Gráfico de resultados
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=data, x=variables[0], y=variables[1], hue='Cluster', palette='viridis')
        plt.title("Resultados de Clustering")
        st.pyplot(plt.gcf())
