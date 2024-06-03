# geoquimica_minera/utils/data_utils.py

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

# Función para corregir tipos de datos
def corregir_tipos(datos):
    datos_corregidos = datos.copy()
    for columna in datos_corregidos.columns:
        try:
            datos_corregidos[columna] = pd.to_numeric(datos_corregidos[columna], errors='coerce')
        except ValueError:
            continue
    return datos_corregidos

# Función para extraer la unidad de una columna
def obtener_unidad(nombre_columna):
    partes = nombre_columna.split('_')
    if len(partes) > 1:
        return partes[-1]
    else:
        return ""

# Función para imputar valores faltantes
def imputar_valores_faltantes(datos, estrategia='mean'):
    imputer = SimpleImputer(strategy=estrategia)
    datos_imputados = imputer.fit_transform(datos)
    return pd.DataFrame(datos_imputados, columns=datos.columns)
