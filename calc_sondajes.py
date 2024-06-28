import streamlit as st
import numpy as np
import plotly.graph_objects as go

# Título de la aplicación
st.title("Optimización de Costos de Sondajes")

# Función para calcular los costos de sondaje
def calcular_costos(tipo_sondaje, metros):
  """
  Calcula los costos de sondaje según el tipo y los metros.

  Args:
      tipo_sondaje (str): "Diamantina" o "Aire Reverso"
      metros (float): Metros a perforar

  Returns:
      float: Costo total del sondaje
  """
  costo_metro = {"Diamantina": 100, "Aire Reverso": 80}
  return costo_metro.get(tipo_sondaje, "Tipo de sondaje inválido") * metros

# Función para crear el gráfico
def graficar_costos(costo_diamantina, costo_aire_reverso, metros_rango):
  """Crea un gráfico interactivo para comparar costos."""

  fig = go.Figure()
  fig.add_trace(go.Scatter(x=metros_rango, y=costo_diamantina, 
                            mode='lines', name='Diamantina', 
                            line=dict(color='royalblue')))
  fig.add_trace(go.Scatter(x=metros_rango, y=costo_aire_reverso,
                            mode='lines', name='Aire Reverso',
                            line=dict(color='firebrick')))

  fig.update_layout(title="Comparación de Costos de Sondaje",
                    xaxis_title="Metros a Perforar",
                    yaxis_title="Costo total ($)",
                    legend_title="Tipo de Sondaje",
                    hovermode="x unified")
  st.plotly_chart(fig)

# Barra lateral para ingresar los datos
st.sidebar.header("Parámetros de Sondaje")

# Usar selectbox con opciones predefinidas
tipo_sondaje = st.sidebar.selectbox("Tipo de Sondaje", ["Diamantina", "Aire Reverso"])

# Usar slider para ingresar metros con rango y paso específico
metros = st.sidebar.slider("Metros a Perforar", min_value=10, max_value=500, value=50, step=10)

# Calcular los costos para un rango de metros
metros_rango = np.arange(10, metros + 10, 10)
costo_diamantina = [calcular_costos("Diamantina", m) for m in metros_rango]
costo_aire_reverso = [calcular_costos("Aire Reverso", m) for m in metros_rango]

# Mostrar los resultados
st.header("Resultados")
st.write(f"Costo total del sondaje: **${calcular_costos(tipo_sondaje, metros):.2f}**")

# Mostrar el gráfico interactivo
graficar_costos(costo_diamantina, costo_aire_reverso, metros_rango)

# Sección para análisis de optimización (simplificada)
st.header("Análisis de Costos")

st.write("En este ejemplo simple, no hay puntos de inflexión o curvas de optimización complejas, "
         "ya que los costos aumentan linealmente con la cantidad de metros perforados. "
         "Sin embargo, en escenarios reales, factores como el costo de movilización, "
         "el alquiler de equipos y las economías de escala pueden influir en la optimización.")

st.write("**Recomendación:** Explora diferentes escenarios y rangos de metros perforados para "
         "identificar posibles puntos de optimización en tu proyecto específico.")
