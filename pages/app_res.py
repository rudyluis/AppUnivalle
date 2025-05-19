import streamlit as st
import pandas as pd
import plotly.express as px

# DAO: Función para cargar datos
def load_data():
    data = {
        'Edad': ['18-24']*10 + ['25-34']*20 + ['35-44']*10 + ['45-54']*5,
        'Género': ['Masculino']*25 + ['Femenino']*20,
        'Experiencia': ['Principiante']*15 + ['Intermedio']*20 + ['Avanzado']*10,
        'Facilidad_de_Uso': [1, 2, 3, 4, 5]*9,
        'Utilidad': [1, 2, 3, 4, 5]*9,
        'Claridad': [2, 3, 4, 5, 5]*9,
        'Satisfacción': [2, 3, 4, 5, 5]*9,
        'Intención_Futura': [1, 2, 3, 4, 5]*9
    }
    return pd.DataFrame(data)

# DAO: Función para agrupar datos por columna
def group_data(df, column):
    return df[column].value_counts().reset_index().rename(columns={'index': column, column: 'Frecuencia'})

# Cargar datos
df = load_data()

# Título
st.title('Análisis de Encuesta de Satisfacción - Streamlit y Plotly')

# Mostrar datos
st.subheader('Datos de la Encuesta')
st.dataframe(df)

# Gráfico y tabla de agrupación para "Edad"
st.subheader('Distribución de Edad')
fig1 = px.histogram(df, x='Edad', title='Distribución de Edad', color='Edad')
# Mover la leyenda abajo
fig1.update_layout(
    legend=dict(
        orientation="h",  # horizontal
        yanchor="bottom",
        y=-0.3,           # puedes ajustar este valor si se superpone
        xanchor="center",
        x=0.5
    )
)
st.plotly_chart(fig1)

st.subheader('Tabla de Distribución de Edad')
age_group = group_data(df, 'Edad')
st.table(age_group)

# Gráfico y tabla de agrupación para "Género"
st.subheader('Distribución de Género')
fig_pie = px.pie(df, names='Género', title='Distribución de Género de los Encuestados'
                )
fig_pie.update_layout(
    legend=dict(
        orientation="h",  # horizontal
        yanchor="bottom",
        y=-0.2,           # ajustar este valor según cuánto quieras que se aleje del gráfico
        xanchor="center",
        x=0.5
    )
)

st.plotly_chart(fig_pie)

st.subheader('Tabla de Distribución de Género')
gender_group = group_data(df, 'Género')
st.table(gender_group)

# Gráfico y tabla de agrupación para "Experiencia"
st.subheader('Nivel de Experiencia en Ciencia de Datos')
fig3 = px.histogram(df, x='Experiencia', title='Nivel de Experiencia', color='Experiencia')
fig3.update_layout(
    legend=dict(
        orientation="h",  # horizontal
        yanchor="bottom",
        y=-0.5,           # ajustar este valor según cuánto quieras que se aleje del gráfico
        xanchor="center",
        x=0.5
    )
)
st.plotly_chart(fig3)

st.subheader('Tabla de Nivel de Experiencia')
experience_group = group_data(df, 'Experiencia')
st.table(experience_group)

# Gráfico y tabla de agrupación para "Facilidad_de_Uso"
st.subheader('Facilidad de Uso de Streamlit')
fig4 = px.box(df, y='Facilidad_de_Uso', title='Facilidad de Uso de Streamlit',
              color_discrete_sequence=px.colors.qualitative.Set2)
st.plotly_chart(fig4)

st.subheader('Tabla de Facilidad de Uso')
facilidad_group = group_data(df, 'Facilidad_de_Uso')
st.table(facilidad_group)

# Gráfico y tabla de agrupación para "Utilidad"
st.subheader('Utilidad de las Funcionalidades de Streamlit')
fig5 = px.box(df, y='Utilidad', title='Utilidad de las Funcionalidades',
              color_discrete_sequence=px.colors.qualitative.Set2)
st.plotly_chart(fig5)

st.subheader('Tabla de Utilidad')
utilidad_group = group_data(df, 'Utilidad')
st.table(utilidad_group)

# Gráfico y tabla de agrupación para "Claridad"
st.subheader('Claridad de la Información Presentada')
fig6 = px.box(df, y='Claridad', title='Claridad de la Presentación',
              color_discrete_sequence=px.colors.qualitative.Set2)
st.plotly_chart(fig6)

st.subheader('Tabla de Claridad')
claridad_group = group_data(df, 'Claridad')
st.table(claridad_group)

# Gráfico y tabla de agrupación para "Satisfacción"
st.subheader('Satisfacción General con la Presentación')
fig7 = px.box(df, y='Satisfacción', title='Satisfacción General',
              color_discrete_sequence=px.colors.qualitative.Set2)
st.plotly_chart(fig7)

st.subheader('Tabla de Satisfacción General')
satisfaccion_group = group_data(df, 'Satisfacción')
st.table(satisfaccion_group)

# Gráfico y tabla de agrupación para "Intención_Futura"
st.subheader('Intención de Usar Streamlit en el Futuro')
fig8 = px.box(df, y='Intención_Futura', title='Intención de Usar Streamlit en el Futuro',
              color_discrete_sequence=px.colors.qualitative.Set2)
st.plotly_chart(fig8)

st.subheader('Tabla de Intención Futura')
intencion_group = group_data(df, 'Intención_Futura')
st.table(intencion_group)

# Gráfico de Cajas Combinado para todas las preguntas
st.subheader('Análisis Comparativo de las Preguntas')
df_melted = df.melt(value_vars=['Facilidad_de_Uso', 'Utilidad', 'Claridad', 'Satisfacción', 'Intención_Futura'],
                    var_name='Pregunta', value_name='Puntuación')
fig_combined = px.box(df_melted, x='Pregunta', y='Puntuación', title='Distribución de Respuestas por Pregunta',
                      color='Pregunta')
fig_combined.update_layout(
    legend=dict(
        orientation="h",  # horizontal
        yanchor="bottom",
        y=-0.5,           # ajustar este valor según cuánto quieras que se aleje del gráfico
        xanchor="center",
        x=0.5
    )
)
st.plotly_chart(fig_combined)

st.subheader('Tabla de Análisis Comparativo')
comparative_group = df_melted.groupby(['Pregunta', 'Puntuación']).size().reset_index(name='Frecuencia')
st.table(comparative_group.T)
