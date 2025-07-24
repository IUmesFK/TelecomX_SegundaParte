import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from imblearn.pipeline import Pipeline

datos = pd.read_csv('telecomx.csv')
print(datos)
print(datos.columns)

# Eliminando idCliente, puesto que no va a aportar valor al analisis ni al modelo
datos = datos.drop(columns='idCliente', axis=1)

print(datos)
datos.info()

# variables explicativas
X = datos.drop(columns='Churn')
# variable respuesta
y = datos['Churn']

columnas = X.columns


# Encoding
# Transforma las variables categóricas a formato numérico para hacerlas compatibles con los algoritmos de machine learning. Utiliza un método de codificación adecuado, como one-hot encoding.
preprocesador = make_column_transformer((OneHotEncoder(drop='if_binary'), 
                                        ['genero', 'es_jubilado', 'tiene_pareja', 'tiene_dependientes', 'servicio_telefonico', 'multiples_lineas', 'servicio_internet', 'servicio_seguridad',
                                        'servicio_backup', 'servicio_proteccion', 'servicio_soporte_tecnico', 'streaming_tv', 'streaming_peliculas', 'tipo_contrato', 'factura_digital', 'metodo_pago']), 
                                        remainder='passthrough',
                                        sparse_threshold=0)

X = preprocesador.fit_transform(X)

df_codificado = pd.DataFrame(X, columns=preprocesador.get_feature_names_out(columnas))

label_encoder = LabelEncoder()

# codificando la variable respuesta
y = label_encoder.fit_transform(y)

print(df_codificado)
print(df_codificado.info())

# Verificación de la Proporción de Cancelación (Churn)

print(datos['Churn'].value_counts())
print(datos['Churn'].value_counts(normalize=True)*100) 
# Tasa de churn:
# False -> 73.42%
# True -> 26.57%


# Balanceo de clases

oversample = SMOTE()

X_resampled, y_resampled = oversample.fit_resample(X, y)

# Verificando el balanceo de clases realizado con SMOTE()
df_resample = pd.DataFrame(X_resampled, columns=preprocesador.get_feature_names_out(columnas))
df_resample['Churn'] = y_resampled

print(df_resample['Churn'].value_counts()) # output: 
# Churn
# 0    5163
# 1    5163


# Normalización o estandarización

normalizacion = MinMaxScaler()

X_normalize = normalizacion.fit_transform(X_resampled)

# Analisis de correlación

print('\n', df_resample.corr())

