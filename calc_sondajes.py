import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

st.title("Calculadora y Visualizador de Costos de Sondajes Mineros")

# Datos iniciales
methods = {
    "DDH Diamantina": {"min_rate": 30, "max_rate": 60, "cost_per_meter": 100},
    "Aire Reverso RC": {"min_rate": 60, "max_rate": 120, "cost_per_meter": 80},
}

# Selección del método de perforación
st.sidebar.title("Parámetros de Sondaje")
method = st.sidebar.selectbox("Seleccionar Método de Perforación", list(methods.keys()))
days = st.sidebar.number_input("Número de Días de Operación", min_value=1, max_value=365, value=30)

# Datos de perforación
min_rate = methods[method]["min_rate"]
max_rate = methods[method]["max_rate"]
cost_per_meter = methods[method]["cost_per_meter"]

# Cálculo de los costos y metros perforados
total_min_meters = min_rate * days
total_max_meters = max_rate * days
total_min_cost = total_min_meters * cost_per_meter
total_max_cost = total_max_meters * cost_per_meter

# Resultados
st.header("Resultados del Sondaje")
st.write(f"Método de Perforación: {method}")
st.write(f"Rango de Perforación: {min_rate} - {max_rate} metros diarios")
st.write(f"Costo por Metro: ${cost_per_meter} USD")
st.write(f"Total de Metros Perforados: {total_min_meters} - {total_max_meters} metros")
st.write(f"Costo Total: ${total_min_cost} - ${total_max_cost} USD")

# Generación de datos para visualización
days_array = np.arange(1, days + 1)
min_meters_array = min_rate * days_array
max_meters_array = max_rate * days_array
min_cost_array = min_meters_array * cost_per_meter
max_cost_array = max_meters_array * cost_per_meter

# Generación de datos para ambos métodos
ddh_min_cost = 30 * days_array * 100
ddh_max_cost = 60 * days_array * 100
rc_min_cost = 60 * days_array * 80
rc_max_cost = 120 * days_array * 80

# Gráficas
st.header("Visualización de Datos")

# Gráfica de Metros Perforados
st.subheader("Metros Perforados a lo Largo del Tiempo")
plt.figure(figsize=(10, 5))
plt.plot(days_array, min_meters_array, label=f'Mínimo Metros ({min_rate} m/día)')
plt.plot(days_array, max_meters_array, label=f'Máximo Metros ({max_rate} m/día)')
plt.xlabel('Días')
plt.ylabel('Metros Perforados')
plt.title('Metros Perforados a lo Largo del Tiempo')
plt.legend()
st.pyplot(plt)

# Gráfica de Costos
st.subheader("Costos a lo Largo del Tiempo")
plt.figure(figsize=(10, 5))
plt.plot(days_array, min_cost_array, label=f'Costo Mínimo (${min_rate * cost_per_meter} USD/día)')
plt.plot(days_array, max_cost_array, label=f'Costo Máximo (${max_rate * cost_per_meter} USD/día)')
plt.xlabel('Días')
plt.ylabel('Costo en USD')
plt.title('Costos a lo Largo del Tiempo')
plt.legend()
st.pyplot(plt)

# Gráfica de Metros Perforados vs. Costos
st.subheader("Metros Perforados vs. Costos")
plt.figure(figsize=(10, 5))
sns.scatterplot(x=min_meters_array, y=min_cost_array, label=f'Costo Mínimo')
sns.scatterplot(x=max_meters_array, y=max_cost_array, label=f'Costo Máximo')
plt.xlabel('Metros Perforados')
plt.ylabel('Costo en USD')
plt.title('Metros Perforados vs. Costos')
plt.legend()
st.pyplot(plt)

# Mapa de Calor de Costos vs. Metros Perforados en Función del Tiempo
st.subheader("Mapa de Calor de Costos vs. Metros Perforados en Función del Tiempo")
heatmap_data = np.vstack([min_cost_array, max_cost_array])
sns.heatmap(heatmap_data, cmap="YlGnBu", xticklabels=days_array, yticklabels=["Costo Mínimo", "Costo Máximo"])
plt.xlabel('Días')
plt.ylabel('Costos')
plt.title('Mapa de Calor de Costos vs. Metros Perforados en Función del Tiempo')
st.pyplot(plt)

# Gráfica de Superficie 3D de ambos métodos
st.subheader("Gráfica de Superficie 3D de Costos de ambos Métodos")
fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111, projection='3d')

X, Y = np.meshgrid(days_array, ["DDH Diamantina", "Aire Reverso RC"])
Z1 = np.array([ddh_min_cost, ddh_max_cost])
Z2 = np.array([rc_min_cost, rc_max_cost])

ax.plot_surface(days_array, ddh_min_cost, ddh_max_cost, color='blue', alpha=0.5, label="DDH Diamantina")
ax.plot_surface(days_array, rc_min_cost, rc_max_cost, color='green', alpha=0.5, label="Aire Reverso RC")

ax.set_xlabel('Días')
ax.set_ylabel('Método de Perforación')
ax.set_zlabel('Costos')
ax.set_title('Gráfica de Superficie 3D de Costos de ambos Métodos')
plt.legend()
st.pyplot(fig)

# Gráfica de Interpolación
st.subheader("Gráfica de Interpolación de Costos a lo Largo del Tiempo")
plt.figure(figsize=(10, 5))
interp_min_cost = np.interp(days_array, days_array, min_cost_array)
interp_max_cost = np.interp(days_array, days_array, max_cost_array)
plt.plot(days_array, interp_min_cost, label=f'Costo Mínimo Interpolado')
plt.plot(days_array, interp_max_cost, label=f'Costo Máximo Interpolado')
plt.xlabel('Días')
plt.ylabel('Costo en USD')
plt.title('Interpolación de Costos a lo Largo del Tiempo')
plt.legend()
st.pyplot(plt)

st.sidebar.header("Optimización de Trabajo")
optimization_method = st.sidebar.selectbox("Método de Optimización", ["Minimizar Costo", "Maximizar Metros Perforados"])

if optimization_method == "Minimizar Costo":
    optimal_rate = min_rate
    optimal_cost = min_cost_array
else:
    optimal_rate = max_rate
    optimal_cost = max_cost_array

st.sidebar.write(f"Para {optimization_method}:")
st.sidebar.write(f"Rate: {optimal_rate} metros/día")
st.sidebar.write(f"Costo Total: ${optimal_cost[-1]} USD")

