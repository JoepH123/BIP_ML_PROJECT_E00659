import pandas as pd
from joblib import dump
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


# LOAD DATA
file_path = '../data/luiss_data_anonym.xlsx'
df = pd.read_excel(file_path)

categorical_features = ["Tdoc", "Iva"]
numerical_features = ['Ateco', 'Importo', 'Conto', 'ContoStd', 'CoDitta', 'TIva', 'Caus']

df = df[categorical_features + numerical_features + ["IvaM"]]

df = df[~df.IvaM.isna()]

X = df.drop('IvaM', axis=1)
y = df[['IvaM']]


# CREATE ALL ENCODERS AND SCALERS
encoder = OneHotEncoder()
encoder.fit(y[['IvaM']])
dump(encoder, "../assets/ui_demo2/target_variable_encoder.joblib")

encoder = OneHotEncoder(drop="first")  
categorical_features = ["Tdoc", "Iva"]
encoder.fit(X[categorical_features])
dump(encoder, "../assets/ui_demo2/categorical_encoder.joblib")

scaler = StandardScaler()
numeric_features = ['Ateco', 'Importo', 'Conto', 'ContoStd', 'CoDitta', 'TIva', 'Caus']
scaler.fit(X[numeric_features])
dump(encoder, "../assets/ui_demo2/numerical_scaler.joblib")


# CREATE DECISION TREE
encoder = OneHotEncoder()
encoded_data = encoder.fit_transform(y[['IvaM']]).toarray()
encoded_df_y = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(['IvaM']))

encoder = OneHotEncoder(drop="first")  
categorical_features = ["Tdoc", "Iva"]
encoded_data = encoder.fit_transform(X[categorical_features]).toarray()
encoded_df_cat = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_features))

scaler = StandardScaler()
numeric_features = ['Ateco', 'Importo', 'Conto', 'ContoStd', 'CoDitta', 'TIva', 'Caus']
scaled_X = scaler.fit_transform(X[numeric_features])
encoded_df_num = pd.DataFrame(scaled_X, columns=scaler.get_feature_names_out(numeric_features))

total_encoded_df_X = pd.concat([encoded_df_cat, encoded_df_num], axis=1)
X_train, X_test, y_train, y_test = train_test_split(total_encoded_df_X, encoded_df_y, test_size=0.2, random_state=42)

model_dt = DecisionTreeClassifier(random_state=42)
model_dt.fit(X_train, y_train)
dump(model_dt, "../assets/ui_demo2/final_decision_tree.joblib")


# CREATE SMALL DEMO DATASET
random_cases_from_test_set = X_test.index.tolist()[:50]
file_path = '../data/luiss_data_anonym.xlsx'
df = pd.read_excel(file_path)
df_test = df[df.index.isin(random_cases_from_test_set)]
df_test.to_csv("../assets/ui_demo2/UI_data_50_cases_from_test_set.csv")