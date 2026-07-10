# -*- coding: utf-8 -*-
"""
train_model.py
Ejecutar este archivo en VS Code para generar 'model_pipeline.pkl'
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from imblearn.over_sampling import SMOTE

# Transformador para limitar valores atípicos (Capping de outliers)
class OutlierCapper(BaseEstimator, TransformerMixin):
    def __init__(self, age_cap=74.0, hpw_lower=32.5, hpw_upper=45.0, fnl_lower=12285.0, fnl_upper=415000.0):
        self.age_cap = age_cap
        self.hpw_lower = hpw_lower
        self.hpw_upper = hpw_upper
        self.fnl_lower = fnl_lower
        self.fnl_upper = fnl_upper

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_out = X.copy()
        if 'age' in X_out.columns:
            X_out['age'] = X_out['age'].clip(upper=self.age_cap)
        if 'hours-per-week' in X_out.columns:
            X_out['hours-per-week'] = X_out['hours-per-week'].clip(lower=self.hpw_lower, upper=self.hpw_upper)
        if 'fnlwgt' in X_out.columns:
            X_out['fnlwgt'] = X_out['fnlwgt'].clip(lower=self.fnl_lower, upper=self.fnl_upper)
        return X_out

def main():
    # Carga local del dataset (debe estar en la misma carpeta que el script)
    try:
        df = pd.read_csv('adult.csv', sep=';')

    except FileNotFoundError:
        print("Error: No se encontró el archivo 'adult.csv' en la ruta local.")
        return

    # Eliminación de duplicados
    df_clean = df.drop_duplicates().reset_index(drop=True)

    # Separación en X (características) e y (variable objetivo)
    y = (df_clean['income'] == '>50K').astype(int)
    X = df_clean.drop(columns=['income'])

    # Definición de variables
    numeric_cols = ['age', 'fnlwgt', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    categorical_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country']

    # Tratamiento manual de nulos específicos para mantener coherencia con el análisis
    X['occupation'] = X['occupation'].fillna('Unknown')

    # Configuración de pipelines de preprocesamiento
    num_pipeline = Pipeline([
        ('capper', OutlierCapper()),
        ('scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', num_pipeline, numeric_cols),
        ('cat', cat_pipeline, categorical_cols)
    ])

    # División en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Preprocesamiento de los conjuntos
    X_train_proc = preprocessor.fit_transform(X_train)
    
    # Manejo de desbalance con SMOTE
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_proc, y_train)

    # Inicialización de modelos optimizados (Grid Search)
    best_rf = RandomForestClassifier(
        n_estimators=200, 
        max_depth=15, 
        min_samples_split=5, 
        class_weight='balanced', 
        random_state=42, 
        n_jobs=-1
    )

    best_gb = GradientBoostingClassifier(
        n_estimators=150, 
        learning_rate=0.05, 
        max_depth=5, 
        random_state=42
    )

    # Creación del ensamble VotingClassifier
    ensemble_model = VotingClassifier(
        estimators=[
            ('rf_opt', best_rf),
            ('gb_opt', best_gb)
        ],
        voting='soft'
    )

    print("Entrenando el ensamble de modelos...")
    ensemble_model.fit(X_train_res, y_train_res)

    # Empaquetado de los componentes para Streamlit
    export_data = {
        'preprocessor': preprocessor,
        'model': ensemble_model,
        'features': {
            'numeric': numeric_cols,
            'categorical': categorical_cols
        }
    }

    # Guardado local del archivo
    with open('model_pipeline.pkl', 'wb') as file:
        pickle.dump(export_data, file)

    print("Proceso finalizado. Se ha generado 'model_pipeline.pkl' de forma local.")

if __name__ == '__main__':
    main()