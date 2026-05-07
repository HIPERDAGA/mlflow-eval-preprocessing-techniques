# MLflow Evaluation of Classical Image Preprocessing Techniques

Repositorio experimental para la evaluación sistemática de técnicas clásicas de preprocesamiento de imágenes bajo condiciones ambientales adversas, utilizando métricas de calidad sin referencia y seguimiento reproducible con MLflow.

---

# Descripción del proyecto

Este proyecto hace parte del trabajo de investigación:

**“Sistema adaptativo multiagente de procesamiento de imágenes para reconocimiento multiobjetivo en diversas condiciones ambientales mediante técnicas de visión por computadora”**.

El objetivo principal del repositorio es:

- evaluar técnicas clásicas de preprocesamiento,
- analizar su impacto visual,
- identificar configuraciones óptimas,
- construir una base de conocimiento experimental,
- integrar posteriormente los resultados en un sistema multiagente adaptativo.

---

# Objetivo del repositorio

Desarrollar una plataforma reproducible basada en MLflow para:

- ejecutar experimentos de preprocesamiento de imágenes,
- registrar configuraciones y métricas,
- comparar resultados automáticamente,
- seleccionar las mejores configuraciones por condición ambiental.

---

# Condiciones ambientales evaluadas

El sistema evalúa imágenes bajo 12 condiciones ambientales:

```text
fog_day
fog_night
fog_twilight

rain_day
rain_night
rain_twilight

sand_day
sand_night
sand_twilight

snow_day
snow_night
snow_twilight
```

Cada condición se evalúa de manera independiente para evitar sesgos relacionados con iluminación o clima.

---

# Técnicas clásicas de preprocesamiento evaluadas

## Corrección cromática

- Gray World
- Simple White Balance

## Mejora de contraste

- Global Histogram Equalization
- CLAHE

## Corrección de iluminación

- Gamma Correction
- Brightness Normalization

## Retinex

- SSR
- MSR
- MSRCR

## Reducción de ruido

- Gaussian Filter
- Median Filter
- Bilateral Filter

## Restauración atmosférica

- Dark Channel Prior (DCP)

---

# Métricas utilizadas

El proyecto utiliza métricas sin referencia para evaluar calidad perceptual:

- NIQE
- BRISQUE
- PIQE

## Criterio de evaluación

En este proyecto:

```text
Menor valor = mejor calidad perceptual
```

El objetivo experimental es minimizar simultáneamente las tres métricas.

---

# Arquitectura del repositorio

```text
mlflow-eval-preprocessing-techniques/
│
├── datasets/
├── preprocessing/
├── metrics/
├── experiments/
├── outputs/
├── notebooks/
├── utils/
├── docs/
└── mlruns/
```

---

# Flujo experimental

```text
Imagen original
        ↓
Preprocesamiento
        ↓
Imagen procesada
        ↓
NIQE / BRISQUE / PIQE
        ↓
Registro en MLflow
        ↓
Comparación de configuraciones
        ↓
Selección óptima
```

---

# Instalación

## 1. Clonar repositorio

```bash
git clone https://github.com/HIPERDAGA/mlflow-eval-preprocessing-techniques.git
cd mlflow-eval-preprocessing-techniques
```

---

## 2. Crear entorno virtual

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / Mac

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Instalar dependencias

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

# Ejecución de MLflow

## Iniciar interfaz MLflow

```bash
mlflow ui --backend-store-uri ./mlruns --port 5000
```

Abrir en navegador:

```text
http://127.0.0.1:5000
```

---

# Ejecución de experimentos

## Ejemplo Gray World

```bash
python experiments/gray_world_experiment.py
```

## Ejemplo CLAHE

```bash
python experiments/clahe_experiment.py
```

---

# Diseño experimental

Cada técnica se evalúa mediante combinaciones sistemáticas de parámetros.

Ejemplo:

## Gray World

```text
preserve_luminance:
    True
    False

channel_gain_limit:
    1.2
    1.5
    2.0
```

Total:

```text
6 configuraciones por condición
72 corridas totales
```

---

# Resultados registrados en MLflow

Cada run almacena:

## Parámetros

- técnica
- condición ambiental
- configuración
- dataset
- número de imágenes

## Métricas

- mean_niqe
- mean_brisque
- mean_piqe

## Artefactos

- imágenes procesadas
- CSV por imagen
- resumen estadístico
- visualizaciones

---

# Salidas del sistema

Los resultados se exportan en:

```text
outputs/
```

Incluyendo:

- imágenes procesadas,
- tablas comparativas,
- reportes,
- gráficas,
- resúmenes CSV.

---

# Relación con el sistema multiagente

Este repositorio constituye la fase experimental offline del sistema adaptativo.

Los resultados permiten construir:

```text
adaptive_preprocessing_policy.csv
```

El cual será utilizado posteriormente por el sistema multiagente para seleccionar dinámicamente la mejor técnica según la condición ambiental detectada.

---

# Tecnologías utilizadas

- Python
- OpenCV
- NumPy
- MLflow
- Scikit-image
- Pandas
- Matplotlib

---

# Reproducibilidad

Todos los experimentos son:

- trazables,
- versionados,
- reproducibles,
- comparables.

Gracias al uso de MLflow y metodología MLOps.

---

# Futuras extensiones

- integración con YOLO
- evaluación del impacto en detección multiobjeto
- selección automática de pipelines
- integración en tiempo real
- interfaz Streamlit
- sistema multiagente distribuido

---

# Autor

Diego Alberto Guevara  
Proyecto de tesis de maestría  
Visión por computadora y sistemas multiagente

---

# Licencia

Proyecto académico y de investigación.
