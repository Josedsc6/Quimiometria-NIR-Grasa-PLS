# Ficha de proyecto para portfolio

## Predicción de grasa mediante espectroscopía NIR y quimiometría

**Objetivo:** estimar el porcentaje de grasa de muestras a partir de sus espectros de absorbancia NIR.

**Métodos:** análisis exploratorio, PCA, preprocesamiento Savitzky–Golay, calibración PLS, comparación con Ridge y Random Forest, validación cruzada y evaluación final mediante RMSE, MAE y R².

**Tecnologías:** Python, pandas, NumPy, SciPy, scikit-learn, Matplotlib y Jupyter.

**Aplicación profesional:** demostración de competencias transferibles a calibración analítica, control de calidad y análisis multivariante en entornos de laboratorio e industria.

**Dataset:** Tecator, datos públicos de 215 muestras con 100 canales espectrales y valores de referencia de grasa, agua y proteína. Los duplicados exactos se eliminan antes de la partición para reducir la fuga de datos.

**Importante:** los resultados numéricos se completan tras ejecutar el notebook. El dataset es de carne; no afirmar que representa directamente aceites, medicamentos ni una validación industrial.
