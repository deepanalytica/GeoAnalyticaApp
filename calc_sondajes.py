import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

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
fig = go.Figure()
fig.add_trace(go.Scatter(x=days_array, y=min_meters_array, mode='lines', name=f'Mínimo Metros ({min_rate} m/día)'))
fig.add_trace(go.Scatter(x=days_array, y=max_meters_array, mode='lines', name=f'Máximo Metros ({max_rate} m/día)'))
fig.update_layout(title='Metros Perforados a lo Largo del Tiempo', xaxis_title='Días', yaxis_title='Metros Perforados')
st.plotly_chart(fig)

# Gráfica de Costos
st.subheader("Costos a lo Largo del Tiempo")
fig = go.Figure()
fig.add_trace(go.Scatter(x=days_array, y=min_cost_array, mode='lines', name=f'Costo Mínimo (${min_rate * cost_per_meter} USD/día)'))
fig.add_trace(go.Scatter(x=days_array, y=max_cost_array, mode='lines', name=f'Costo Máximo (${max_rate * cost_per_meter} USD/día)'))
fig.update_layout(title='Costos a lo Largo del Tiempo', xaxis_title='Días', yaxis_title='Costo en USD')
st.plotly_chart(fig)

# Gráfica de Metros Perforados vs. Costos
st.subheader("Metros Perforados vs. Costos")
fig = go.Figure()
fig.add_trace(go.Scatter(x=min_meters_array, y=min_cost_array, mode='markers', name=f'Costo Mínimo'))
fig.add_trace(go.Scatter(x=max_meters_array, y=max_cost_array, mode='markers', name=f'Costo Máximo'))
fig.update_layout(title='Metros Perforados vs. Costos', xaxis_title='Metros Perforados', yaxis_title='Costo en USD')
st.plotly_chart(fig)

# Mapa de Calor de Costos vs. Metros Perforados en Función del Tiempo
st.subheader("Mapa de Calor de Costos vs. Metros Perforados en Función del Tiempo")
heatmap_data = np.vstack([min_cost_array, max_cost_array])
fig = px.imshow(heatmap_data, labels=dict(x="Días", y="Método de Perforación", color="Costo en USD"),
                x=days_array, y=["Costo Mínimo", "Costo Máximo"])
fig.update_layout(title='Mapa de Calor de Costos vs. Metros Perforados en Función del Tiempo')
st.plotly_chart(fig)

# Gráfica de Superficie 3D de ambos métodos
st.subheader("Gráfica de Superficie 3D de Costos de ambos Métodos")
X, Y = np.meshgrid(days_array, ["DDH Diamantina", "Aire Reverso RC"])
ddh_costs = np.array([ddh_min_cost, ddh_max_cost])
rc_costs = np.array([rc_min_cost, rc_max_cost])
fig = go.Figure(data=[
    go.Surface(z=ddh_costs, x=days_array, y=["DDH Diamantina"]*len(days_array), colorscale='Blues', opacity=0.5),
    go.Surface(z=rc_costs, x=days_array, y=["Aire Reverso RC"]*len(days_array), colorscale='Greens', opacity=0.5)
])
fig.update_layout(scene=dict(
    xaxis_title='Días',
    yaxis_title='Método de Perforación',
    zaxis_title='Costos'
), title='Gráfica de Superficie 3D de Costos de ambos Métodos')
st.plotly_chart(fig)

# Gráfica de Interpolación
st.subheader("Gráfica de Interpolación de Costos a lo Largo del Tiempo")
interp_min_cost = np.interp(days_array, days_array, min_cost_array)
interp_max_cost = np.interp(days_array, days_array, max_cost_array)
fig = go.Figure()
fig.add_trace(go.Scatter(x=days_array, y=interp_min_cost, mode='lines', name=f'Costo Mínimo Interpolado'))
fig.add_trace(go.Scatter(x=days_array, y=interp_max_cost, mode='lines', name=f'Costo Máximo Interpolado'))
fig.update_layout(title='Interpolación de Costos a lo Largo del Tiempo', xaxis_title='Días', yaxis_title='Costo en USD')
st.plotly_chart(fig)

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
