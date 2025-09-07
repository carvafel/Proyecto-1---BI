import pandas as pd
from ydata_profiling import ProfileReport

# 1. Cargar dataset de entrenamiento
df = pd.read_excel("Datos_proyecto.xlsx")

# 2. Generar reporte
profile = ProfileReport(df, title="EDA proyecto – Entrenamiento", explorative=True)

profile.to_file("eda_proyecto.html")
