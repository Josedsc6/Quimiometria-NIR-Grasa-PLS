# Quimiometría aplicada a espectroscopía NIR: predicción de grasa mediante PLS

**Proyecto de portfolio · Química + Data Science · Python · Machine Learning**

## Objetivo
Construir y validar un modelo que estime el porcentaje de grasa de muestras a partir de sus espectros de absorbancia NIR. El caso utiliza el conjunto de datos público **Tecator**, un benchmark clásico de quimiometría: 215 muestras de carne picada, 100 canales espectrales entre 850 y 1050 nm y resultados de referencia de grasa, agua y proteína.

El objetivo del proyecto es demostrar un flujo de trabajo reproducible de análisis químico de datos: exploración espectral, PCA, preprocesamiento, calibración PLS, comparación con modelos alternativos y evaluación sobre un conjunto de prueba no utilizado durante el ajuste.

> **Alcance y honestidad:** los datos son de carne, no de aceite ni de una planta farmacéutica. El proyecto demuestra métodos transferibles a control de calidad y análisis químico, pero no acredita experiencia práctica operando un espectrómetro NIR ni valida un método industrial listo para uso regulado.

## Pregunta de trabajo
¿Hasta qué punto puede predecirse el contenido de grasa a partir del espectro NIR y cómo se compara la calibración quimiométrica PLS con modelos de referencia?

## Dataset y procedencia
- Conjunto Tecator distribuido en el paquete `modeldata` y en el repositorio Rdatasets.
- CSV: <https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/modeldata/meats.csv>
- Descripción técnica y permiso de redistribución: <https://modeldata.tidymodels.org/reference/meats.html>
- Los espectros se registraron con un analizador Tecator Infratec mediante transmisión NIR (NIT); los porcentajes de grasa, agua y proteína se determinaron por métodos de química analítica.
- El dataset contiene duplicados exactos. El análisis los detecta y elimina antes de dividir los datos para reducir el riesgo de que una misma muestra aparezca en entrenamiento y prueba.

**Nota de atribución solicitada por la fuente:** los datos originales se atribuyen al instrumento y empresa Tecator. La fuente pide que se mencione Tecator Infratec Food and Feed Analyzer y que se le envíe una copia previa de cualquier publicación. Para este portfolio se cita la procedencia del conjunto de datos y no se presenta como una publicación científica.

## Métodos
1. Carga y revisión de estructura, valores ausentes y duplicados.
2. Visualización de espectros y distribución del porcentaje de grasa.
3. PCA para explorar la estructura multivariante (exploración, no predictor final).
4. División entrenamiento/prueba después de eliminar duplicados.
5. Validación cruzada de 5 particiones dentro del conjunto de entrenamiento.
6. Comparación de PLS sobre espectros originales, PLS con primera derivada Savitzky–Golay, Ridge y Random Forest.
7. Evaluación final en prueba con RMSE, MAE y R²; gráficas observado frente a predicho y residuos.

## Cómo ejecutarlo
Se recomienda Python 3.10–3.12.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

Abre `notebooks/Analisis_Quimiometria_Nir.ipynb` y ejecuta las celdas en orden. La primera ejecución necesita conexión a Internet para descargar el CSV. El dataset se guarda en `data/` y los resultados en `results/`.

También se puede ejecutar el análisis de principio a fin desde la terminal:

```bash
python src/analisis_nir.py
```

## Resultados generados
- `results/metricas_modelos.csv`: métricas de validación final.
- `results/predicciones_test.csv`: valores reales y predichos del conjunto de prueba.
- `results/figures/`: espectros, PCA, observado vs. predicho y residuos.
- `results/resumen_ejecucion.txt`: tamaños, duplicados eliminados y mejor modelo según RMSE de prueba.

### Resultados de la ejecución de referencia (`random_state=42`)

| Modelo | RMSE prueba | R² prueba | RMSE validación cruzada |
|---|---|---|---|
| PLS, espectro original | 1.82 | 0.980 | 2.73 |
| Ridge | 2.36 | 0.967 | 3.04 |
| PLS, primera derivada | 2.42 | 0.965 | 2.94 |
| Random Forest | 7.09 | 0.699 | 8.60 |

RMSE en puntos porcentuales de grasa, con 154 muestras de entrenamiento y 39 de prueba. Con una única partición y tan pocas muestras de prueba, las diferencias entre PLS, Ridge y PLS con derivada no son concluyentes; la estimación de error más realista es la de validación cruzada.

## Estructura
```text
proyecto_quimiometria_nir/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── README.md
├── notebooks/
│   └── Analisis_Quimiometria_Nir.ipynb
├── src/
│   └── analisis_nir.py
├── docs/
│   ├── ficha_proyecto.md
│   └── atribucion_dataset.txt
└── results/
    └── figures/
```

## Competencias demostradas
- Química analítica y lectura de datos espectrales.
- Preprocesamiento de señales y análisis exploratorio multivariante.
- PCA, calibración PLS y comparación de algoritmos.
- Validación cruzada, separación entrenamiento/prueba y prevención de fuga de datos.
- Python, pandas, NumPy, scikit-learn, SciPy y Matplotlib.
- Documentación y reproducibilidad.

## Próximas mejoras posibles
- Investigar estrategias de selección de variables y número de componentes latentes.
- Añadir métricas de calibración como bias y RPD, interpretadas con cautela por el tamaño de muestra.
- Replicar el flujo con un dataset NIR de aceites o materias primas más cercano a la industria objetivo.
- Estudiar validación externa por lote/instrumento si se dispone de datos adecuados.
- Ampliar las rejillas de hiperparámetros (el mejor `alpha` de Ridge coincide con el límite inferior de la rejilla) y repetir la partición entrenamiento/prueba varias veces para estimar la variabilidad del RMSE.
