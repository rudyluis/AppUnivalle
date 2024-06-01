import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import re

# Cargar datos
df = pd.read_csv("cleaned.csv")

# Funciones de preprocesamiento
def preprocess_data(df):
    df = df.drop(columns=['Brand','Product_Description'])
    df['RAM'] = df['RAM'].replace('Up', '64')
    df['RAM'] = pd.to_numeric(df['RAM'], errors='coerce').fillna(0).astype(int)

    def processor_to_int(processor):
        match = re.search(r'i(\d)\s*(\d+)?\s*(\d+)?\w*?\s*(Gen\.)?', processor)
        if match:
            core_type = match.group(1)
            gen = match.group(3) if match.group(3) else match.group(2) if match.group(2) else '0'
            code = int(core_type + gen)
            return code if code >= 100 else int(core_type + '0' + gen)
        else:
            return 0

    df['Processor_Int'] = df['Processor'].apply(processor_to_int)

    most_common_resolution = df['Resolution'].mode()[0]
    df['Resolution'] = df['Resolution'].fillna(most_common_resolution)

    def resolution_to_int(resolution):
        match = re.match(r'(\d+)\s*x\s*(\d+)', resolution)
        if match:
            width = int(match.group(1))
            height = int(match.group(2))
            return width * height
        return 0

    df['Resolution_Int'] = df['Resolution'].apply(resolution_to_int)
    
    df = df.drop(columns=['Resolution', 'Processor'])

    df = pd.get_dummies(df, columns=['Condition'], prefix=['Condition'])

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
st.title("Laptop Price Prediction App")

# Crear combos (selectboxes) para cada característica
st.sidebar.header("Select Laptop Features")

screen_size = st.sidebar.selectbox("Screen Size", sorted(df['Screen_Size'].unique()))
ram = st.sidebar.selectbox("RAM", sorted(df['RAM'].unique()))
processor = st.sidebar.selectbox("Processor", sorted(df['Processor_Int'].unique()))
gpu = st.sidebar.selectbox("GPU", sorted(df['GPU_Label'].unique()))
gpu_type = st.sidebar.selectbox("GPU Type", sorted(df['GPU_Type_Label'].unique()))
resolution = st.sidebar.selectbox("Resolution", sorted(df['Resolution_Int'].unique()))
condition = st.sidebar.selectbox("Condition", sorted(df.filter(like='Condition_').columns))

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

input_data[f'Condition_{condition}'] = 1

# Predicciones
if st.sidebar.button("Predict Price"):
    prediction_tree = decision_tree_regressor.predict(input_data)[0]
    prediction_forest = random_forest.predict(input_data)[0]
    prediction_gb = gradient_boosting.predict(input_data)[0]

    st.write("### Predicted Prices")
    st.write(f"Decision Tree Regressor: ${prediction_tree:.2f}")
    st.write(f"Random Forest Regressor: ${prediction_forest:.2f}")
    st.write(f"Gradient Boosting Regressor: ${prediction_gb:.2f}")
