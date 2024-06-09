import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import re
import seaborn as sns
import matplotlib.pyplot as plt

import plotly.express as px

# Cargar datos
df = pd.read_csv("df/dataequipos.csv")
df_pre = pd.read_csv("df/dataequipos.csv")

imagen = "logos/logoTec.png"
st.set_page_config(page_title="DATA TEC",
					page_icon=imagen,
					layout="wide",
					initial_sidebar_state="auto"
					)


# Funciones de preprocesamiento
def processor_to_int(processor):
    match = re.search(r'i(\d)\s*(\d+)?\s*(\d+)?\w*?\s*(Gen\.)?', processor)
    if match:
        core_type = match.group(1)
        gen = match.group(3) if match.group(3) else match.group(2) if match.group(2) else '0'
        code = int(core_type + gen)
        return code if code >= 100 else int(core_type + '0' + gen)
    else:
        return 0

def resolution_to_int(resolution):
    match = re.match(r'(\d+)\s*x\s*(\d+)', resolution)
    if match:
        width = int(match.group(1))
        height = int(match.group(2))
        return width * height
    return 0

def preprocess_data(df):
    df = df.drop(columns=['Brand', 'Product_Description'])
    df['RAM'] = df['RAM'].replace('Up', '64')
    df['RAM'] = pd.to_numeric(df['RAM'], errors='coerce').fillna(0).astype(int)

    df['Processor_Int'] = df['Processor'].apply(processor_to_int)

    most_common_resolution = df['Resolution'].mode()[0]
    df['Resolution'] = df['Resolution'].fillna(most_common_resolution)

    df['Resolution_Int'] = df['Resolution'].apply(resolution_to_int)
    
    df = df.drop(columns=['Resolution', 'Processor'])

    df = pd.get_dummies(df, columns=['Condition'], prefix='Condition')

    label_encoder_gpu = LabelEncoder()
    label_encoder_gpu_type = LabelEncoder()

    df['GPU_Label'] = label_encoder_gpu.fit_transform(df['GPU'])
    df['GPU_Type_Label'] = label_encoder_gpu_type.fit_transform(df['GPU_Type'])

    df = df.drop(columns=['GPU', 'GPU_Type'])

    return df

# Preprocesar los datos
df = preprocess_data(df)

# Dividir datos en características y variable objetivo
X = df.drop(columns=['Price'])
Y = df['Price']

# Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

# Entrenar modelos
decision_tree_regressor = DecisionTreeRegressor(random_state=42)
decision_tree_regressor.fit(X_train, y_train)

random_forest = RandomForestRegressor(n_estimators=100, random_state=42)
random_forest.fit(X_train, y_train)

gradient_boosting = GradientBoostingRegressor(random_state=42)
gradient_boosting.fit(X_train, y_train)

# Streamlit app
st.title("Aplicación de Predicción de Precios de Laptops")
st.sidebar.image(imagen,  width=250)	
# Crear combos (selectboxes) para cada característica
st.sidebar.header("Selecciona Características de la Laptop")
brand = st.sidebar.selectbox("Marca", sorted(df_pre['Brand'].unique()))
product_description = st.sidebar.selectbox("Descripción del Producto", sorted(df_pre[df_pre['Brand'] == brand]['Product_Description'].unique()))
screen_size = st.sidebar.slider("Tamaño de Pantalla", float(df['Screen_Size'].min()), float(df['Screen_Size'].max()), float(df['Screen_Size'].mean()))

ram = st.sidebar.selectbox("RAM", sorted(df['RAM'].unique()))

# Procesador
df_pre['Processor_Int'] = df_pre['Processor'].apply(processor_to_int)
processor_map = {val: f"{val} - {df_pre[df_pre['Processor_Int'] == val]['Processor'].values[0]}" for val in df['Processor_Int'].unique()}
processor = st.sidebar.selectbox("Procesador", options=sorted(list(processor_map.keys())), format_func=lambda x: processor_map[x])

# GPU
label_encoder_gpu = LabelEncoder()
df_pre['GPU_Label'] = label_encoder_gpu.fit_transform(df_pre['GPU'])
gpu_map = {val: f"{val} - {df_pre[df_pre['GPU_Label'] == val]['GPU'].values[0]}" for val in df_pre['GPU_Label'].unique()}
gpu = st.sidebar.selectbox("GPU", options=sorted(list(gpu_map.keys())), format_func=lambda x: gpu_map[x])

label_encoder_gpu_type = LabelEncoder()
df_pre['GPU_Type_Label'] = label_encoder_gpu_type.fit_transform(df_pre['GPU_Type'])
# Tipo de GPU
gpu_type_map = {val: f"{val} - {df_pre[df_pre['GPU_Type_Label'] == val]['GPU_Type'].values[0]}" for val in df_pre['GPU_Type_Label'].unique()}
gpu_type = st.sidebar.selectbox("Tipo de GPU", options=sorted(list(gpu_type_map.keys())), format_func=lambda x: gpu_type_map[x])

# Resolución
most_common_resolution = df_pre['Resolution'].mode()[0]
df_pre['Resolution'] = df_pre['Resolution'].fillna(most_common_resolution)
df_pre['Resolution_Int'] = df_pre['Resolution'].apply(resolution_to_int)
resolution_map = {val: f"{val} - {df_pre[df_pre['Resolution_Int'] == val]['Resolution'].values[0]}" for val in df_pre['Resolution_Int'].unique()}
resolution = st.sidebar.selectbox("Resolución", options=sorted(list(resolution_map.keys())), format_func=lambda x: resolution_map[x])

condition = st.sidebar.selectbox("Condición", sorted(df.filter(like='Condition_').columns))

# Crear un DataFrame con la selección del usuario
input_data = pd.DataFrame({
    'Screen_Size': [screen_size],
    'RAM': [ram],
    'Processor_Int': [processor],
    'GPU_Label': [gpu],
    'GPU_Type_Label': [gpu_type],
    'Resolution_Int': [resolution],
})

# Añadir columnas de condición
for col in df.columns:
    if 'Condition_' in col:
        input_data[col] = 0

input_data[condition] = 1

# Asegurarse de que las columnas de input_data coincidan con las de X_train
input_data = input_data.reindex(columns=X_train.columns, fill_value=0)

# Predicciones
if st.sidebar.button("Predecir Precio"):
    prediction_tree = decision_tree_regressor.predict(input_data)[0]
    prediction_forest = random_forest.predict(input_data)[0]
    prediction_gb = gradient_boosting.predict(input_data)[0]

    st.write("### Precios Pronosticados")
    st.write(f"🌳 Árbol de Decisión: ${prediction_tree:.2f}")
    st.write(f"🌲 Bosque Aleatorio: ${prediction_forest:.2f}")
    st.write(f"📈 Boosting de Gradiente: ${prediction_gb:.2f}")

# Mostrar tabla filtrada
filtered_df = df_pre[
    (df_pre['Brand'] == brand) &
    (df_pre['Product_Description'] == product_description) &
    (df_pre['Screen_Size'] == screen_size) &
    (df_pre['RAM'] == ram) &
    (df_pre['Processor_Int'] == processor) &
    (df_pre['GPU_Label'] == gpu) &
    (df_pre['GPU_Type_Label'] == gpu_type) &
    (df_pre['Resolution_Int'] == resolution)
]

st.write("### Laptops Filtradas")
st.write(df)

# Gráficos adicionales
st.write("### Información Adicional")

# Distribución de Precios
price_distribution_fig = px.histogram(df, x='Price', nbins=20, title="Distribución de Precios de Laptops")
price_distribution_fig.update_layout(xaxis_title="Precio", yaxis_title="Frecuencia")
st.plotly_chart(price_distribution_fig)

# Relación entre características y precio
st.write("#### Relación entre Características y Precio")
feature = st.selectbox("Selecciona característica para graficar contra Precio", ['Screen_Size', 'RAM', 'Processor_Int', 'Resolution_Int'])
c1,c2=st.columns(2)
with c1:
    # Relación entre características y precio
    feature_price_fig = px.scatter(df, x=feature, y='Price', title=f"Precio vs {feature}", labels={feature: feature, 'Price': 'Precio'})
    feature_price_fig.update_layout(xaxis_title=feature, yaxis_title="Precio")
    st.plotly_chart(feature_price_fig)
with c2:
    # Importancia de características

    importances = random_forest.feature_importances_
    indices = np.argsort(importances)[::-1]


    # Importancia de características
    importance_df = pd.DataFrame({'Característica': X_train.columns[indices], 'Importancia': importances[indices]})
    feature_importance_fig = px.bar(importance_df, x='Importancia', y='Característica', orientation='h', title="Importancia de Características")
    feature_importance_fig.update_layout(xaxis_title="Importancia Relativa", yaxis_title="Característica")
    st.plotly_chart(feature_importance_fig)

# ---- HIDE STREAMLIT STYLE ----
hide_st_style = """
				<style>
				#MainMenu {visibility: hidden;}
				footer {visibility: hidden;}
				header {visibility: hidden;}
				</style>
				"""
st.markdown(hide_st_style, unsafe_allow_html=True)		
st.markdown("""
	  Realizado por Rudy Manzaneda - 2024
""")