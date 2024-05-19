import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score

# Cargar los datos
df = pd.read_csv("LibroEncuestaLimpia.csv", encoding='latin-1', delimiter=';')

# Codificar todas las variables categóricas
encoder = LabelEncoder()
for column in df.select_dtypes(include=['object']).columns:
    df[column] = encoder.fit_transform(df[column])

# Manejar valores faltantes (rellenar con la moda)
df.fillna(df.mode().iloc[0], inplace=True)

# Definir las características (variables independientes) y la variable objetivo
X = df.drop(['plan_presupuesto_familiar', 'ID'], axis=1)
y = df['plan_presupuesto_familiar']

# Dividir los datos en conjuntos de entrenamiento y prueba (80% entrenamiento, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inicializar los clasificadores base
clf1 = RandomForestClassifier(random_state=42)
clf2 = RandomForestClassifier(random_state=42)  # Se pueden usar diferentes clasificadores

# Inicializar el Voting Classifier con los clasificadores base
voting_clf = VotingClassifier(estimators=[('rf1', clf1), ('rf2', clf2)], voting='hard')

# Entrenar el Voting Classifier
voting_clf.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred = voting_clf.predict(X_test)

# Calcular la precisión del modelo
accuracy = accuracy_score(y_test, y_pred)
print("Precisión del modelo:", accuracy)
