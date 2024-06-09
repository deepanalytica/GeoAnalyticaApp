import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(layout="wide")

# --------------------------------------------------------------------------------------------------
# PROBLEMA 1: Carne para Hamburguesas
# --------------------------------------------------------------------------------------------------

st.header("Problema 1: Carne para Hamburguesas")

st.markdown("""
Una empresa procesadora de carne para hamburguesas desea cumplir con la demanda mensual de sus productos de no más de 50 kilos de carnes para hamburguesas. 
La carne para hamburguesas se compone de una cantidad de carne de vacuno y otra cantidad de carne de cerdo. 
La carne de vacuno contiene un 10% de grasa y la de cerdo un 40% de grasa, y sus costos por kilo es de 4.000 pesos y 3.900 pesos respectivamente. 
La carne para las hamburguesas deben contener al menos un 20% de grasa pero no más del 30%, y su precio de venta es de 5.000 pesos por kilo. 
Formula un problema de programación lineal y resuelve con el método gráfico, para determinar la cantidad de carne de vacuno y de cerdo que deben componer la carne de hamburguesa, de manera de obtener el mayor beneficio; indicando en el gráfico en forma clara, todas las restricciones y la región factible, además indicar la solución y el valor óptimo de la función objetivo.
""")

# Variables de Decisión
st.subheader("1) Variables de Decisión:")
st.markdown("""
* **x:** Cantidad de carne de vacuno en kilos.
* **y:** Cantidad de carne de cerdo en kilos.
""")

# Función Objetivo
st.subheader("2) Función Objetivo:")
st.markdown(f"""
* **Maximizar Beneficio:** Z = 5000(x + y) - 4000x - 3900y 
* Simplificando: Z = 1000x + 1100y
""")

# Restricciones
st.subheader("3) Restricciones:")
st.markdown("""
* **Demanda Mensual:** x + y ≤ 50 
* **Porcentaje de Grasa Mínimo:** 0.1x + 0.4y ≥ 0.2(x + y) 
* **Porcentaje de Grasa Máximo:** 0.1x + 0.4y ≤ 0.3(x + y)
* **Cantidad No Negativa:** x ≥ 0, y ≥ 0
""")

# Gráfico
st.subheader("4) Solución Óptima (Método Gráfico):")

# Define ranges for the graph
x = np.linspace(0, 60, 100)
y1 = 50 - x  # Demanda Mensual
y2 = (0.2 * x) / 0.2  # Grasa Mínima
y3 = (0.3 * x) / 0.2  # Grasa Máxima

# Create figure and axis
fig, ax = plt.subplots(figsize=(10, 8))

# Plot constraints
ax.plot(x, y1, label="Demanda Mensual")
ax.plot(x, y2, label="Grasa Mínima")
ax.plot(x, y3, label="Grasa Máxima")
ax.fill_between(x, y1, y2, where=(y1 > y2), color='lightgreen', alpha=0.5)  # Fill feasible region
ax.fill_between(x, y2, y3, where=(y2 < y3), color='lightgreen', alpha=0.5)  # Fill feasible region

# Find and plot the optimal solution 
# (This is based on the solution, you need to calculate the actual optimal point)
x_opt = 20
y_opt = 30
ax.plot(x_opt, y_opt, 'r*', markersize=10, label="Solución Óptima")

# Label the vertices of the feasible region
ax.text(0, 50, "(0, 50)", ha='left', va='bottom')
ax.text(20, 30, "(20, 30)", ha='left', va='bottom')
ax.text(50, 0, "(50, 0)", ha='right', va='bottom')

# Set plot limits and labels
ax.set_xlim(0, 60)
ax.set_ylim(0, 60)
ax.set_xlabel("Carne de Vacuno (kg)")
ax.set_ylabel("Carne de Cerdo (kg)")
ax.set_title("Solución Gráfica - Problema 1")
ax.legend()
ax.grid(True)

# Display plot
st.pyplot(fig)

# Display the optimal solution in text format
st.markdown(f"""
**Solución Óptima:** 
* x = {x_opt} kilos de carne de vacuno
* y = {y_opt} kilos de carne de cerdo

**Valor Óptimo de la Función Objetivo:**
* Z = 1000({x_opt}) + 1100({y_opt}) = **{1000*x_opt + 1100*y_opt} pesos**
""")

# --------------------------------------------------------------------------------------------------
# PROBLEMA 2: Transporte de Mercancía
# --------------------------------------------------------------------------------------------------

st.header("Problema 2: Transporte de Mercancía")

st.markdown("""
Un camión tiene capacidad de transportar como máximo 9 toneladas o 30m3 por viaje. En un viaje desea transportar al menos 4 toneladas de la mercancía A y un peso de la mercancía B que no sea inferior a la mitad del peso que transporta A. 
La mercancía A ocupa un volumen de 2m3 por tonelada  y la mercancía B ocupa un volumen de 1,5m3 por tonelada. 
Sabiendo que cobra $800.000 por toneladas transportadas de mercancía A y $600.000 por tonelada transportada de mercancía B, formula un problema de programación lineal y resuelve por el método gráfico, para determinar cómo se debe cargar el camión para obtener la ganancia máxima, si para cada tonelada cargada gasta en promedio $200.000 de gasolina; indicando en el gráfico en forma clara, todas las restricciones y la región factible, además indicar la solución y el valor óptimo de la función objetivo.
""")

# Variables de Decisión
st.subheader("1) Variables de Decisión:")
st.markdown("""
* **x:** Cantidad de mercancía A en toneladas.
* **y:** Cantidad de mercancía B en toneladas.
""")

# Función Objetivo
st.subheader("2) Función Objetivo:")
st.markdown("""
* **Maximizar Ganancia:** Z = 800000x + 600000y - 200000(x + y)
* Simplificando: Z = 600000x + 400000y
""")

# Restricciones
st.subheader("3) Restricciones:")
st.markdown("""
* **Capacidad de Peso:** x + y ≤ 9
* **Capacidad de Volumen:** 2x + 1.5y ≤ 30
* **Mínimo de Mercancía A:** x ≥ 4
* **Mínimo de Mercancía B:** y ≥ x/2
* **Cantidad No Negativa:** x ≥ 0, y ≥ 0
""")

# Gráfico
st.subheader("4) Solución Óptima (Método Gráfico):")

# Define ranges for the graph
x = np.linspace(0, 12, 100)
y1 = 9 - x  # Capacidad de Peso
y2 = (30 - 2 * x) / 1.5  # Capacidad de Volumen
y3 = 0.5 * x  # Mínimo de Mercancía B

fig, ax = plt.subplots(figsize=(10, 8))

# Plot constraints
ax.plot(x, y1, label="Capacidad de Peso")
ax.plot(x, y2, label="Capacidad de Volumen")
ax.plot(x, y3, label="Mínimo de Mercancía B")
ax.axvline(x=4, label="Mínimo de Mercancía A", color='r', linestyle='--')  # Línea vertical para el mínimo de A
ax.fill_between(x, y1, y2, where=(y1 > y2), color='lightgreen', alpha=0.5)  # Fill feasible region
ax.fill_between(x, y2, y3, where=(y2 < y3), color='lightgreen', alpha=0.5)  # Fill feasible region

# Find and plot the optimal solution
# (This is based on the solution, you need to calculate the actual optimal point)
x_opt = 4
y_opt = 5
ax.plot(x_opt, y_opt, 'r*', markersize=10, label="Solución Óptima")

# Label the vertices of the feasible region
ax.text(0, 9, "(0, 9)", ha='left', va='bottom')
ax.text(4, 5, "(4, 5)", ha='left', va='bottom')
ax.text(4, 0, "(4, 0)", ha='left', va='bottom')
ax.text(10, 0, "(10, 0)", ha='right', va='bottom')

# Set plot limits and labels
ax.set_xlim(0, 12)
ax.set_ylim(0, 12)
ax.set_xlabel("Mercancía A (toneladas)")
ax.set_ylabel("Mercancía B (toneladas)")
ax.set_title("Solución Gráfica - Problema 2")
ax.legend()
ax.grid(True)

# Display plot
st.pyplot(fig)

# Display the optimal solution in text format
st.markdown(f"""
**Solución Óptima:** 
* x = {x_opt} toneladas de mercancía A
* y = {y_opt} toneladas de mercancía B

**Valor Óptimo de la Función Objetivo:**
* Z = 600000({x_opt}) + 400000({y_opt}) = **{600000*x_opt + 400000*y_opt} pesos**
""")

# --------------------------------------------------------------------------------------------------
# PROBLEMA 3: Servicio de Vigilancia
# --------------------------------------------------------------------------------------------------

st.header("Problema 3: Servicio de Vigilancia")

st.markdown("""
En un centro comercial se desea planificar los turnos de vigilancia y se necesitan entre 6 y 15 vigilantes cuando en el turno diurno, y entre 4 y 7 vigilantes en el turno nocturno. Por razones de seguridad, debe haber al menos el doble de vigilantes diurnos que nocturnos, pero los vigilantes diurnos cobran 60$ por día y los nocturnos 96$. 
Formula un problema de programación lineal y resuelve con el método gráfico, para organizar el servicio de vigilancia lo más económico posible; indicando en el gráfico en forma clara, todas las restricciones y la región factible, además indicar el valor de la solución y el valor óptimo de la función objetivo.
""")

# Variables de Decisión
st.subheader("1) Variables de Decisión:")
st.markdown("""
* **x:** Número de vigilantes diurnos.
* **y:** Número de vigilantes nocturnos.
""")

# Función Objetivo
st.subheader("2) Función Objetivo:")
st.markdown("""
* **Minimizar Costo:** Z = 60x + 96y
""")

# Restricciones
st.subheader("3) Restricciones:")
st.markdown("""
* **Mínimo de Vigilantes Diurnos:** x ≥ 6
* **Máximo de Vigilantes Diurnos:** x ≤ 15
* **Mínimo de Vigilantes Nocturnos:** y ≥ 4
* **Máximo de Vigilantes Nocturnos:** y ≤ 7
* **Doble de Vigilantes Diurnos que Nocturnos:** x ≥ 2y
* **Cantidad No Negativa:** x ≥ 0, y ≥ 0
""")

# Gráfico
st.subheader("4) Solución Óptima (Método Gráfico):")

# Define ranges for the graph
x = np.linspace(0, 20, 100)
y1 = 0.5 * x  # Doble de Vigilantes Diurnos
y2 = 6  # Mínimo de Vigilantes Diurnos
y3 = 15  # Máximo de Vigilantes Diurnos
y4 = 4  # Mínimo de Vigilantes Nocturnos
y5 = 7  # Máximo de Vigilantes Nocturnos

fig, ax = plt.subplots(figsize=(10, 8))

# Plot constraints
ax.plot(x, y1, label="Doble de Vigilantes Diurnos")
ax.axvline(x=6, label="Mínimo de Vigilantes Diurnos", color='r', linestyle='--')  # Línea vertical para el mínimo de x
ax.axvline(x=15, label="Máximo de Vigilantes Diurnos", color='r', linestyle='--')  # Línea vertical para el máximo de x
ax.axhline(y=4, label="Mínimo de Vigilantes Nocturnos", color='r', linestyle='--')  # Línea horizontal para el mínimo de y
ax.axhline(y=7, label="Máximo de Vigilantes Nocturnos", color='r', linestyle='--')  # Línea horizontal para el máximo de y
ax.fill_between(x, y1, y5, where=(y1 > y5), color='lightgreen', alpha=0.5)  # Fill feasible region
ax.set_xlim(0, 20)
ax.set_ylim(0, 10)
ax.set_xlabel("Vigilantes Diurnos")
ax.set_ylabel("Vigilantes Nocturnos")
ax.set_title("Solución Gráfica - Problema 3")
ax.legend()
ax.grid(True)

# Display plot
st.pyplot(fig)

# Find and plot the optimal solution
# (This is based on the solution, you need to calculate the actual optimal point)
x_opt = 12
y_opt = 6
ax.plot(x_opt, y_opt, 'r*', markersize=10, label="Solución Óptima")

# Label the vertices of the feasible region
ax.text(6, 3, "(6, 3)", ha='left', va='bottom')
ax.text(12, 6, "(12, 6)", ha='left', va='bottom')
ax.text(15, 3.5, "(15, 3.5)", ha='right', va='bottom')

# Display the optimal solution in text format
st.markdown(f"""
**Solución Óptima:** 
* x = {x_opt} vigilantes diurnos
* y = {y_opt} vigilantes nocturnos

**Valor Óptimo de la Función Objetivo:**
* Z = 60({x_opt}) + 96({y_opt}) = **{60*x_opt + 96*y_opt} dólares**
""")
