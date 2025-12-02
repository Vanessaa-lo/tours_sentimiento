# tours_sentimiento
Clasificador de Sentimientos para Comentarios de Tours– Empresa Turística (Tours, Yates, Pesca)
Proyecto final de la materia de "Aprendizaje Maquina" de la carrera de Ing. En Computación

## Descripción del Proyecto

Este proyecto implementa un clasificador de sentimientos basado en Machine Learning y
Procesamiento de Lenguaje Natural (PLN), diseñado para integrarse con un identificador de comentarios de 
atención al cliente de una empresa turística que maneja reservaciones de yates, tours
marítimos, y pesca deportiva.

## Este proyecto proporciona un sistema completo para analizar el sentimiento de comentarios de usuarios, aplicable a reseñas de tours, experiencias y servicios turísticos.
Incluye:

- API REST desarrollada con FastAPI

- Modelo de Machine Learning (TF-IDF + Logistic Regression)

- Interfaz web ligera para pruebas en tiempo real

- Pipeline de entrenamiento completamente automatizado

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
```bash
python -m src.entrenar_modelo
```
4. ## Ejecutar la API##
```bash
uvicorn src.api_sentimiento:app --reload
```


#Interfaz Web

El proyecto incluye una UI simple y funcional.

Acceso:

👉 http://127.0.0.1:8000/ui

Permite ingresar comentarios y visualizar el sentimiento predicho por el modelo.

tours_sentimiento/
│
├── src/
│   ├── api_sentimiento.py          # API principal en FastAPI
│   └── entrenar_modelo.py          # Script para entrenar el modelo ML
│
├── models/
│   ├── modelo_sentimiento.joblib   # Modelo entrenado
│   └── vectorizador_tfidf.joblib   # Vectorizador TF-IDF
│
├── ui/
│   ├── index.html                  # Interfaz web
│   ├── app.js                      # Lógica en JavaScript
│   └── styles.css                  # Estilos de la interfaz
│
├── env_sentimiento/                # Entorno virtual (opcional)
├── requirements.txt                # Dependencias del proyecto
└── README.md                       # Documentación
