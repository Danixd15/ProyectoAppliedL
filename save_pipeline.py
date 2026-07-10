# -*- coding: utf-8 -*-
"""
save_pipeline.py
Ejecutar este archivo en VS Code para generar 'model_pipeline.pkl' con todos los modelos entrenados.
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
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from imblearn.over_sampling import SMOTE

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
    try:
        df = pd.read_csv('adult.csv', sep=';')
    except FileNotFoundError:
        print("Error: No se encontró el archivo 'adult.csv' en la ruta local.")
        return

    df_clean = df.drop_duplicates().reset_index(drop=True)
    y = (df_clean['income'] == '>50K').astype(int)
    X = df_clean.drop(columns=['income'])

    numeric_cols = ['age', 'fnlwgt', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    categorical_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country']

    X['occupation'] = X['occupation'].fillna('Unknown')

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

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train_proc = preprocessor.fit_transform(X_train)
    
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_proc, y_train)

    # Definir todos los modelos para poder compararlos en caliente en la app
    models = {
        'Regresión Logística': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
        'Árbol de Decisión': DecisionTreeClassifier(max_depth=8, class_weight='balanced', random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_split=5, class_weight='balanced', random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, max_depth=5, random_state=42)
    }

    # Crear y entrenar el Ensamble Híbrido con los modelos optimizados
    ensemble = VotingClassifier(
        estimators=[
            ('rf_opt', models['Random Forest']),
            ('gb_opt', models['Gradient Boosting'])
        ],
        voting='soft'
    )
    
    # Entrenar cada modelo individualmente
    trained_models = {}
    for name, model in models.items():
        print(f"Entrenando {name}...")
        model.fit(X_train_res, y_train_res)
        trained_models[name] = model

    print("Entrenando Ensamble Híbrido...")
    ensemble.fit(X_train_res, y_train_res)
    trained_models['Ensamble Híbrido'] = ensemble

    # Exportar preprocesador y diccionario de modelos
    export_data = {
        'preprocessor': preprocessor,
        'models': trained_models,
        'features': {
            'numeric': numeric_cols,
            'categorical': categorical_cols
        }
    }

    with open('model_pipeline.pkl', 'wb') as file:
        pickle.dump(export_data, file)

    print("Proceso finalizado. Se ha generado 'model_pipeline.pkl' de forma local.")

if __name__ == '__main__':
    main()
