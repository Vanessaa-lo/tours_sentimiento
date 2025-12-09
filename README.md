# tours_sentimiento
Clasificador de Sentimientos para Comentarios de Tours– Empresa Turística (Tours, Yates, Pesca)
Proyecto final de la materia de "Aprendizaje Maquina" de la carrera de Ing. En Computación

## Descripción del Proyecto

Este proyecto analiza comentarios de clientes para clasificarlos como positivos, neutrales o negativos.
Incluye una API en FastAPI, un modelo de Machine Learning, y un dashboard web interactivo para visualizar estadísticas.

## Este proyecto proporciona un sistema completo para analizar el sentimiento de comentarios de usuarios, aplicable a reseñas de tours, experiencias y servicios turísticos.

## Incluye:

- API REST construida con FastAPI
- Análisis de sentimientos usando:
- Regresión Logística (modelo clásico)
- MLPClassifier con 2 capas ocultas (128 y 64 neuronas, activación tanh)
- Vectorización con TF-IDF
- Almacenamiento automático de reseñas analizadas en CSV
- Dashboard web con:
- Estadísticas generales
- Gráfica de dona
- Tendencia de sentimientos por día
- Historial de reseñas
- Descarga de datos en CSV

## Instalación

1. ## Clonar el repositorio: ## 
```bash
git clone https://github.com/Vanessaa-lo/tours_sentimiento.git
cd tours_sentimiento
```

2. ## Crear entorno virtual: ##
```bash
python -m venv env_sentimiento
source env_sentimiento/Scripts/activate   # Windows
```
3. ## Instalar dependencias##
```bash
pip install -r requirements.txt
```
3. ## Entrenar el modelo##

Genera el modelo y vectorizador en la carpeta /models:

Regresión Logística
```bash
python -m src.entrenar_modelo
```
MLP – Red neuronal
```bash
python -m src.entrenar_modelo_mlp
```

4. ## Ejecutar la API##
```bash
uvicorn src.api_sentimiento:app --reload
```

5. ## ¿Cómo funciona el análisis?

- El texto se limpia y vectoriza con TF-IDF
- El modelo predice sentimiento: positivo / neutral / negativo
- Se calcula la probabilidad
- Se guarda la reseña en data/resenas_analizadas.csv
- Se actualizan estadísticas en el dashboard

## Modelos usados
# 🔹 Regresión Logística

- Modelo lineal para clasificación.
- Rápido, estable y eficiente para texto.
- Funciona excelente con TF-IDF.
- Ideal para producción.

# 🔹 MLPClassifier

Red neuronal con:
- Capa oculta 1 → 128 neuronas (tanh)
- Capa oculta 2 → 64 neuronas (tanh)
- Optimizador Adam
- Captura relaciones no lineales en el texto.

## 📊 Dashboard incluido

- El dashboard muestra:
- Total de reseñas
- Porcentaje por categoría
- Gráfica de dona
- Tendencia diaria
- Tabla con historial
- Botón para exportar CSV
- Boton para exportar a excel
Acceso:

👉 http://127.0.0.1:8000/ui

http://127.0.0.1:8000/dashboard

Permite ingresar comentarios y visualizar el sentimiento predicho por el modelo.

