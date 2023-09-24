import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Create containers
header = st.container()
dataset = st.container()
features = st.container()
model_training = st.container()

with header:
    st.title("Linea de Acción incorporación de Tecnología")
    st.text("Análisis de los emprendedores visitados desde el 2016 al 2020")

with dataset:
    st.header("Dataset de emprendedores")
    st.markdown(
        """Este dataset surge de los resultados de un formulario de google utilizado por el equipo técnico para realizar las entrevistas a los emprendedores potenciales de ser fortalecidos por el programa."""
    )

    # Leer el dataset
    st.text("Tabla de emprendedores (5 primeras filas)")
    emprendedores = pd.read_csv("data/emprendedores_it.csv")
    st.write(emprendedores.head())

    # Creo una copia del dataset para trabajar
    emprendedores_copia = emprendedores.copy()

    # Eliminar filas con valor nulo en la columna "sexo"
    emprendedores_copia.dropna(
        subset=["sexo", "antiguedad_emprendimiento_meses"], inplace=True
    )

    columnas_a_eliminar = [
        "observaciones_iva",
        "fecha_entrevista",
        "fecha_nac",
        "capacitacion_desc",
        "cant_pers_emprendimiento",
        "monto_maximo",
        "Sub-Rubro:",
        "Sub-Rubro:.1",
        "Actividad:",
        "Actividad:.1",
        "Actividad:.2",
        "Actividad:.3",
        "Actividad:.4",
        "Actividad:.5",
        "Actividad:.6",
        "Actividad:.7",
        "Actividad:.8",
        "Actividad:.9",
        "Actividad:.10",
        "Actividad:.11",
        "Actividad:.12",
        "Actividad:.13",
        "Actividad:.14",
        "Actividad:.15",
        "situacion_familiar",
        "cant_clientes",
        "inversion",
        "otras_fuentes",
        "equipamiento_actual",
        "equipamiento_solicitado",
        "herramienta_aprobada",
    ]

    # Eliminar las columnas
    emprendedores_copia.drop(columns=columnas_a_eliminar, inplace=True)

    st.subheader("Dataset para entrenamiento del modelo:")

    st.write(emprendedores_copia.head())

    st.subheader("Exploración y Visualización:")

    # Show dataset
    st.text("Distribucion de emprendedores por localidad:")
    distribucion_emprendedores = pd.DataFrame(
        emprendedores_copia["localidad"].value_counts()
    ).head(50)
    st.bar_chart(distribucion_emprendedores)

    st.subheader("Identificar la Variable Objetivo y su Relación con localidad:")
    st.markdown(
        """Ahora, para avanzar en la preparación de datos y construcción del modelo, necesitamos entender mejor el dataset y cuál es la variable objetivo que queremos predecir ('devolucion_equipo_tecnico') y cómo se relaciona con otras variables, especialmente 'localidad'. Vamos a explorar eso."""
    )

    st.write(emprendedores_copia["devolucion_equipo_tecnico"].value_counts())

    st.markdown(
        """Ahora entendemos que la variable 'devolucion_equipo_tecnico' tiene tres categorías: Aprobado, No aprobado, y Aprobado - El emprendedor rechazó el fortalecimiento. Estamos interesados en predecir si está Aprobado o no. Vamos a simplificar esta tarea combinando las dos primeras categorías en una sola, ya que nos interesa saber si fue aprobado o no.
        
Aquí, creamos una nueva columna llamada 'target' que será nuestra variable objetivo. Asignamos 1 si 'devolucion_equipo_tecnico' es Aprobado y 0 en caso contrario.
        """
    )

    emprendedores_copia["target"] = emprendedores_copia[
        "devolucion_equipo_tecnico"
    ].apply(lambda x: 1 if x == "Aprobado" else 0)

    X = emprendedores_copia["localidad"]  # Feature: 'localidad'
    y = emprendedores_copia["target"]  # Target: 'Aprobado' (1) o no (0)

    st.write(emprendedores_copia[["devolucion_equipo_tecnico", "target"]].head())

    emprendedores_encoded = pd.get_dummies(emprendedores_copia, columns=["localidad"])

    # Preparar los datos para el modelo:
    # Seleccionar características (X) y la variable objetivo (y)
    X = emprendedores_encoded.drop(["target", "devolucion_equipo_tecnico"], axis=1)
    y = emprendedores_encoded["target"]

    # Verificación de los datos preparados
    st.subheader("Verificación de Datos")
    st.write("Primeras filas de los datos:")
    st.write(emprendedores_encoded.head())

    st.write("Estadísticas descriptivas:")
    st.write(emprendedores_encoded.describe())

    st.write("Información del conjunto de datos:")
    st.write(emprendedores_encoded.info())

    st.write("Características (X):")
    st.write(X.head())

    st.write("Variable objetivo (y):")
    st.write(y.head())


with model_training:
    st.markdown(
        """
        Vamos a utilizar las siguientes características para nuestro modelo:

                departamento: El departamento donde se encuentra el emprendedor.

                situacion_iva: La situación frente al IVA del emprendedor.

                antiguedad_emprendimiento_anio: La antigüedad del emprendimiento en años.

                ganancia_final_mensual: La ganancia final mensual del emprendedor.

                tiempo_dedicado_sem: El tiempo dedicado semanalmente al emprendimiento."""
    )

    # Seleccionar las características y la variable objetivo
    caracteristicas = [
        "departamento",
        "situacion_iva",
        "antiguedad_emprendimiento_anio",
        "ganancia_final_mensual",
        "tiempo_dedicado_sem",
        "target",
    ]
    datos_modelo = emprendedores_copia[caracteristicas].copy()

    # Separar las características y la variable objetivo
    X = datos_modelo.drop("target", axis=1)  # Características
    y = datos_modelo["target"]  # Variable objetivo

    # Codificación one-hot para las variables categóricas
    X = pd.get_dummies(X, columns=["departamento", "situacion_iva"])

    # Dividir los datos en conjuntos de entrenamiento y prueba

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Vamos a utilizar un clasificador de bosque aleatorio para predecir si un emprendedor será aprobado o no:

    # Crear y entrenar el modelo
    modelo = RandomForestClassifier(random_state=42)
    modelo.fit(X_train, y_train)

    # Predecir en el conjunto de prueba
    predicciones = modelo.predict(X_test)

    # Calcular la precisión
    precision = accuracy_score(y_test, predicciones)
    print("Precisión del modelo:", precision)

    st.text("La precisión del modelo es: {:.2f}%".format(precision * 100))

    # Ahora, vamos a utilizar el modelo para predecir si un emprendedor será aprobado o no.
    # Para ello, necesitamos crear un formulario para que el usuario ingrese los valores de las características.
    # Luego, usaremos el modelo para predecir si el emprendedor será aprobado o no.
    # Finalmente, mostraremos el resultado de la predicción al usuario.

with model_training:
    st.header("Predicción para un Nuevo Emprendedor")

    departamento = st.selectbox(
        "Departamento", emprendedores_copia["departamento"].unique()
    )
    situacion_iva = st.selectbox(
        "Situación IVA", emprendedores_copia["situacion_iva"].unique()
    )
    antiguedad_emprendimiento_anio = st.slider(
        "Antigüedad del emprendimiento (años)", 0, 30, 5
    )
    ganancia_final_mensual = st.slider("Ganancia final mensual", 0, 50000, 10000)
    tiempo_dedicado_sem = st.slider("Tiempo dedicado por semana", 0, 100, 20)

    # Crear un DataFrame con los datos del nuevo emprendedor
    nuevo_emprendedor = pd.DataFrame(
        {
            "departamento": [departamento],
            "situacion_iva": [situacion_iva],
            "antiguedad_emprendimiento_anio": [antiguedad_emprendimiento_anio],
            "ganancia_final_mensual": [ganancia_final_mensual],
            "tiempo_dedicado_sem": [tiempo_dedicado_sem],
        }
    )

    # Asegurar que el DataFrame del nuevo emprendedor tenga las mismas columnas que el conjunto de entrenamiento
    # Esto es importante para que el modelo pueda hacer predicciones correctamente
    nuevo_emprendedor_encoded = pd.get_dummies(
        nuevo_emprendedor, columns=["departamento", "situacion_iva"]
    )

    # Añadir columnas faltantes que estaban en el conjunto de entrenamiento pero no están en el nuevo DataFrame
    for col in X.columns:
        if col not in nuevo_emprendedor_encoded.columns:
            nuevo_emprendedor_encoded[col] = 0

    # Reordenar las columnas para que estén en el mismo orden que en el conjunto de entrenamiento
    nuevo_emprendedor_encoded = nuevo_emprendedor_encoded[X.columns]

    # Codificar variables categóricas
    nuevo_emprendedor = pd.get_dummies(
        nuevo_emprendedor, columns=["departamento", "situacion_iva"]
    )

    # Realizar la predicción
    prediccion_nuevo_emprendedor = modelo.predict(nuevo_emprendedor_encoded)

    st.text("Predicción para el nuevo emprendedor:")
    if prediccion_nuevo_emprendedor[0] == 1:
        st.text(
            "El emprendedor tiene alta probabilidad de ser aprobado por el equipo técnico."
        )
    else:
        st.text(
            "El emprendedor tiene baja probabilidad de ser aprobado por el equipo técnico."
        )
