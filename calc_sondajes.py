import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
