import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Datos
data = {
    'Facilidad_de_Uso': [1, 2, 3, 4, 5]*9,
    'Utilidad': [1, 2, 3, 4, 5]*9,
    'Claridad': [2, 3, 4, 5, 5]*9,
    'Satisfacción': [2, 3, 4, 5, 5]*9,
    'Intención_Futura': [1, 2, 3, 4, 5]*9
}

# Crear DataFrame
df = pd.DataFrame(data)

# Matriz de correlación
correlation_matrix = df.corr()

# Mostrar matriz como tabla
print(correlation_matrix)

# Opcional: visualizar con mapa de calor
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Matriz de Correlación')
plt.show()