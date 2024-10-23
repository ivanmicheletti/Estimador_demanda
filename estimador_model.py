#importar las librerias necesarias
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
import xgboost as xgb


from sklearn.metrics import mean_squared_error, r2_score

#Lectura del archivo Excel
df=pd.read_csv('https://raw.githubusercontent.com/ivanmicheletti/Estimador_demanda/refs/heads/main/BD.csv')
df

df.isna().sum()

df.columns

df2=df[['Día y Hora', 'Temp amb','TP2 - MVA','TP1 - MVA','33kV - P3 - MVA','33kV - Ent TP2 - MVA','33kV - P1 - MVA','33kV - Ent TP1 - MVA',
    '33kV - P2 - MVA','13,2kV - VGGZI - MVA','13,2kV - Ent TP2 - MVA','13,2kV - P. Nuevo - MVA','13,2kV - Arsat - MVA','MVA totales']]

df2.isna().sum()

df2.isna().sum().sum()

df2.info()

df3 = df2.dropna()

df3.isna().sum().sum()

df3.columns

#Cambio de nombre de las columnas
df3.columns=['tiempo', 'temp_amb', 'TP2_MVA', 'TP1_MVA', 'P3_MVA',
       '33kV_Ent_TP2_MVA', 'P1_MVA', '33kV_Ent_TP1_MVA',
       'P2_MVA', 'VGGZI_MVA', '13,2kV_Ent_TP2_MVA',
       'P_Nuevo_MVA', 'Arsat_MVA', 'MVA_totales']

df3.head()

# Convertir la columna 'fecha' a tipo datetime
df3['tiempo'] = pd.to_datetime(df3['tiempo'])

# Extraer características de la columna 'tiempo'
df3['año'] = df3['tiempo'].dt.year
df3['mes'] = df3['tiempo'].dt.month
df3['día'] = df3['tiempo'].dt.day
df3['hora'] = df3['tiempo'].dt.hour
df3['día_semana'] = df3['tiempo'].dt.dayofweek  # Lunes=0, Domingo=6

# Eliminar la columna 'tiempo' original (si ya no es útil)
df3 = df3.drop('tiempo', axis=1)

df3.head()

df3.dtypes

# Convert problematic columns to numeric
for column in df3.select_dtypes(include=['object']).columns:
    df3[column] = df3[column].str.replace(',', '.').astype(float)

df3.dtypes

X=df3.drop(['MVA_totales'],axis=1)

y=df3.MVA_totales

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

print('Dimensiones en train \n-X:{}'.format(X_train.shape, y_train.shape))

print('Dimensiones en test \n-X:{}'.format(X_test.shape, y_test.shape))

# Crear y entrenar el modelo
lin_regr = LinearRegression()
lin_regr.fit(X_train, y_train)

xgboots=XGBRegressor()
xgboots.fit(X_train, y_train)

# Evaluar el modelo
y_pred = lin_regr.predict(X_test)
y_pred2 = xgboots.predict(X_test)

#Calculo la métrica para validar el modelo:

r2_lineal = r2_score(y_test, y_pred)
r2_xgb = r2_score(y_test, y_pred2)

print(f"R2 Score: {r2_lineal}")
print(f"R2 Score: {r2_xgb}")

with open('lin_reg.pkl', 'wb') as li:
    pickle.dump(lin_regr, li)


with open('xgb.pkl', 'wb') as xg:
    pickle.dump(xgboots, xg)

#pip freeze > requirements.txt