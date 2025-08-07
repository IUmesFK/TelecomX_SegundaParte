# 🧠 Análisis de Cancelación de Clientes (Churn) - Empresa de Telecomunicaciones

Este proyecto tiene como objetivo analizar los factores que influyen en la cancelación de clientes (churn) en una empresa de telecomunicaciones, utilizando modelos de clasificación y técnicas de selección de variables. El objetivo final es detectar a tiempo a los clientes propensos a cancelar el servicio para implementar estrategias de retención más efectivas.

---

## 📚 Librerías utilizadas

El análisis y modelado fue realizado en Python utilizando las siguientes librerías:

- `pandas` → Manipulación de datos estructurados
- `matplotlib` & `seaborn` → Visualización de datos
- `scikit-learn` → Modelado, evaluación y selección de características
- `imbalanced-learn` → Técnicas de balanceo como SMOTE

---

## 📊 Objetivos del proyecto

- Explorar los datos y entender patrones de cancelación.
- Evaluar distintos modelos de clasificación:
  - `RandomForestClassifier`
  - `KNeighborsClassifier`
  - `DecisionTreeClassifier`
- Determinar las variables más relevantes para la predicción.
- Optimizar los modelos priorizando la métrica de **recall**, clave para no pasar por alto posibles cancelaciones.
- Proponer estrategias de retención fundamentadas en los hallazgos del modelo.

---

## 🧪 Modelos evaluados

### 🌲 Random Forest Classifier
- **Mejor desempeño general y en recall.**
- Principales variables:
  - Tipo de contrato (mes a mes, 1 año, 2 años)
  - Método de pago (cheque electrónico)
  - Antigüedad
  - Servicio de internet (fibra óptica)
- **Modelo recomendado** para predicción de churn.

### 🔍 KNeighbors Classifier
- Importancia de variables determinada con `permutation_importance`.
- Factores relevantes: antigüedad, género, factura digital y número de líneas.
- Desempeño más bajo en recall.

### 🌳 Decision Tree Classifier
- Alta interpretabilidad.
- Coincide con Random Forest en variables clave.
- Importancia fuerte de contrato mes a mes y pagos con cheque electrónico.

---

## 🧠 Estrategias sugeridas

- Migrar clientes con contrato "mes a mes" a contratos anuales con beneficios exclusivos.
- Incentivar métodos de pago electrónicos más estables.
- Segmentar y cuidar especialmente a clientes con poca antigüedad.
- Mejorar servicios asociados a fibra óptica o brindar alternativas.

---

## 📁 Estructura del repositorio

📂 telecom-churn-project/

│

├── 📄 Telecom_X_Segunda_parte.ipynb # Notebook principal con análisis y modelado

├── 📄 telecomx.csv # Dataset original

├── 📄 requirements.txt # Dependencias del entorno

├── 📄 .gitignore # Exclusión de archivos temporales

└── 📄 README.md # Descripción del proyecto (este archivo)
