# tours_sentimiento
Chatbot de Atención al Cliente – Empresa Turística (Tours, Yates, Pesca)
Proyecto final de la materia de "Aprendizaje Maquina" de la carrera de Ing. En Computación

## Descripción del Proyecto

Este proyecto implementa un clasificador de sentimientos basado en Machine Learning y
Procesamiento de Lenguaje Natural (PLN), diseñado para integrarse con un chatbot de 
atención al cliente de una empresa turística que maneja reservaciones de yates, tours
marítimos, y pesca deportiva.

## El sistema analiza mensajes de los usuarios y determina si el sentimiento expresado es positivo o negativo, lo cual permite:

- Detectar clientes molestos o frustrados
- Priorizar quejas reales
- Automatizar respuestas empáticas
- Sugerir promociones a clientes satisfechos
- Mejorar la calidad del servicio

## El modelo utiliza una red neuronal (MLP) con:

- ReLU en la capa oculta
- Sigmoide en la capa de salida
- Vectorización de texto mediante TF-IDF

## Instalación

1. ## Clonar el repositorio: ## 
```bash
git clone https://github.com/Vanessaa-lo/tours_sentimiento.git
cd tours_sentimiento
```

2. ## Crear entorno virtual: ##
```bash
python -m venv venv
venv\Scripts\activate  # Windows
```
3. ## Instalar dependencias##
```bash
pip install -r requirements.txt
```



