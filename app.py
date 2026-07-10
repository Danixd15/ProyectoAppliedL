# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
import seaborn as sns
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

# 1. Transformador personalizado para mitigar el efecto de valores atípicos
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

# Registro para compatibilidad
sys.modules['save_pipeline'] = sys.modules[__name__]

st.set_page_config(
    page_title="CrediNova - Evaluación de Capacidad Económica",
    page_icon="💳",
    layout="wide"
)

# Estilos CSS avanzados
st.markdown("""
    <style>
    .main-title {
        font-size: 2.3rem;
        color: #1E3A8A;
        font-weight: bold;
        text-align: center;
        margin-bottom: 5px;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #4B5563;
        text-align: center;
        margin-bottom: 25px;
    }
    .section-header {
        color: #1E3A8A;
        border-bottom: 2px solid #E5E7EB;
        padding-bottom: 8px;
        margin-top: 25px;
        margin-bottom: 15px;
        font-weight: bold;
    }
    .result-card-high {
        background-color: #ECFDF5;
        border-left: 6px solid #10B981;
        padding: 20px;
        border-radius: 8px;
        margin-bottom: 20px;
    }
    .result-card-low {
        background-color: #FFFBEB;
        border-left: 6px solid #F59E0B;
        padding: 20px;
        border-radius: 8px;
        margin-bottom: 20px;
    }
    .kpi-container {
        display: flex;
        justify-content: space-between;
        gap: 15px;
        margin-bottom: 20px;
    }
    .kpi-card {
        background-color: #F9FAFB;
        border: 1px solid #E5E7EB;
        padding: 15px;
        border-radius: 8px;
        flex: 1;
        text-align: center;
    }
    .kpi-value {
        font-size: 1.4rem;
        font-weight: bold;
        color: #1E3A8A;
    }
    .kpi-label {
        font-size: 0.85rem;
        color: #6B7280;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>CrediNova Financial Services S.A.C.</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Sistema de Evaluación Predictiva de Ingresos para Segmentación Financiera y Riesgos</div>", unsafe_allow_html=True)


# =========================================================================
# ENTRENAMIENTO AUTOMÁTICO EN LA NUBE (CON MEMORIA EN CACHÉ)
# =========================================================================
@st.cache_resource
def train_pipeline_in_cloud():
    # Carga del dataset local (subido a GitHub)
    try:
        df = pd.read_csv('adult.csv', sep=';')
    except FileNotFoundError:
        return None

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

    # Modelos configurados para ejecutarse de manera rápida en la nube
    models = {
        'Regresión Logística': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
        'Árbol de Decisión': DecisionTreeClassifier(max_depth=8, class_weight='balanced', random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=30, max_depth=8, min_samples_split=15, class_weight='balanced', random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42)
    }

    ensemble = VotingClassifier(
        estimators=[
            ('rf_opt', models['Random Forest']),
            ('gb_opt', models['Gradient Boosting'])
        ],
        voting='soft'
    )
    
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train_res, y_train_res)
        trained_models[name] = model

    ensemble.fit(X_train_res, y_train_res)
    trained_models['Ensamble Híbrido'] = ensemble

    return {
        'preprocessor': preprocessor,
        'models': trained_models
    }


# Spinner para avisar al usuario mientras se entrena la primera vez
with st.spinner("Iniciando el servidor de CrediNova y entrenando los modelos predictivos en la nube..."):
    pipeline_data = train_pipeline_in_cloud()

if pipeline_data is None:
    st.error("No se pudo iniciar la aplicación porque falta el archivo 'adult.csv' en su repositorio de GitHub.")
else:
    preprocessor = pipeline_data['preprocessor']
    models_dict = pipeline_data['models']

    tab1, tab2 = st.tabs(["📊 Contexto & Comparativa General", "🔍 Evaluación Crediticia e Ingreso de Datos"])

    with tab1:
        st.markdown("<h3 class='section-header'>1. Contexto y Problemática</h3>", unsafe_allow_html=True)
        col_ctx1, col_ctx2 = st.columns(2)
        with col_ctx1:
            st.write("""
            **CrediNova Financial Services S.A.C.** requiere identificar con precisión la capacidad socioeconómica 
            de los solicitantes para mitigar el riesgo crediticio y optimizar la colocación de productos financieros. 
            El uso de este sistema analítico reemplaza los criterios tradicionales y reduce el riesgo de sobreendeudamiento.
            """)
        with col_ctx2:
            st.write("""
            El dataset empleado es el **Adult Census Income (UCI)**, el cual sirve como proxy de capacidad económica. 
            La meta del proyecto es clasificar si el solicitante supera el umbral anual de **$50,000 USD** utilizando 
            predictores demográficos y laborales.
            """)

        st.markdown("<h3 class='section-header'>2. Comparativa de Desempeño Histórico (Métricas de Entrenamiento)</h3>", unsafe_allow_html=True)
        
        datos_comparativa = {
            "Corrida / Estrategia": [
                "1. Modelo Base", "1. Modelo Base", "2. Validación Cruzada", 
                "3. Con Grid Search", "3. Con Grid Search", "4. Mezcla de Modelos"
            ],
            "Modelo": [
                "Random Forest Base", "Gradient Boosting Base", "Random Forest (Muestreo CV)", 
                "Random Forest Optimizado", "Gradient Boosting Optimizado", "Ensamble Híbrido (Mezcla)"
            ],
            "Accuracy": [0.8208, 0.8614, 0.8195, 0.8272, 0.8707, 0.8684],
            "Precision": [0.5863, 0.7834, 0.6540, 0.5960, 0.7712, 0.7451],
            "Recall": [0.8536, 0.5822, 0.6810, 0.8490, 0.6133, 0.6845],
            "F1-Score": [0.6951, 0.6680, 0.6672, 0.7003, 0.6835, 0.7135],
            "ROC AUC": [0.9152, 0.9159, 0.8950, 0.9180, 0.9260, 0.9221]
        }
        df_comparativa = pd.DataFrame(datos_comparativa)
        st.dataframe(df_comparativa.style.highlight_max(axis=0, subset=["F1-Score", "ROC AUC"], color="#D1FAE5"))

    with tab2:
        st.markdown("<h3 class='section-header'>1. Formulario de Captura de Datos</h3>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Edad (años)", min_value=17, max_value=90, value=38, step=1)
            education = st.selectbox("Nivel de Instrucción", [
                'Bachelors', 'Some-college', '11th', 'HS-grad', 'Masters', '9th', 
                'Assoc-acdm', 'Assoc-voc', '7th-8th', 'Doctorate', 'Prof-school', 
                '5th-6th', '10th', '1st-4th', 'Preschool', '12th'
            ])
            educational_num = st.slider("Años de Educación Equivalente", 1, 16, 10)
            gender = st.selectbox("Género", ['Male', 'Female'])

        with col2:
            workclass = st.selectbox("Tipo de Empleador", [
                'Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 
                'Local-gov', 'State-gov', 'Without-pay', 'Never-worked'
            ])
            occupation = st.selectbox("Ocupación Principal", [
                'Tech-support', 'Craft-repair', 'Other-service', 'Sales', 
                'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 
                'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 
                'Transport-moving', 'Priv-house-serv', 'Protective-serv', 
                'Armed-Forces', 'Unknown'
            ])
            hours_per_week = st.number_input("Horas de Trabajo Semanal", min_value=1, max_value=99, value=40, step=1)
            marital_status = st.selectbox("Estado Civil", [
                'Married-civ-spouse', 'Never-married', 'Divorced', 'Separated', 
                'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'
            ])

        with col3:
            relationship = st.selectbox("Rol Familiar", [
                'Wife', 'Husband', 'Own-child', 'Other-relative', 'Not-in-family', 'Unmarried'
            ])
            race = st.selectbox("Grupo Racial", ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'])
            capital_gain = st.number_input("Ganancia de Capital Anual (USD)", min_value=0, max_value=100000, value=0, step=1000)
            capital_loss = st.number_input("Pérdida de Capital Anual (USD)", min_value=0, max_value=5000, value=0, step=100)
            native_country = st.selectbox("País de Origen", [
                'United-States', 'Mexico', 'Philippines', 'Germany', 'Canada', 'Puerto-Rico', 'El-Salvador', 'India'
            ])

        input_df = pd.DataFrame({
            'age': [age],
            'workclass': [workclass],
            'fnlwgt': [189664],
            'education': [education],
            'educational-num': [educational_num],
            'marital-status': [marital_status],
            'occupation': [occupation],
            'relationship': [relationship],
            'race': [race],
            'gender': [gender],
            'capital-gain': [capital_gain],
            'capital-loss': [capital_loss],
            'hours-per-week': [hours_per_week],
            'native-country': [native_country]
        })

        st.markdown("---")
        if st.button("Ejecutar Evaluación Crediticia", type="primary"):
            try:
                processed_input = preprocessor.transform(input_df)
                
                resultados_usuario = []
                probabilidades_grafico = []
                nombres_grafico = []

                for name, model in models_dict.items():
                    pred = model.predict(processed_input)[0]
                    prob = model.predict_proba(processed_input)[0]
                    
                    etiqueta_pred = ">50K" if pred == 1 else "<=50K"
                    prob_alta = prob[1] * 100
                    
                    resultados_usuario.append({
                        "Modelo": name,
                        "Predicción del Nivel de Ingreso": etiqueta_pred,
                        "Probabilidad de Ingreso Alto (>50K)": f"{prob_alta:.2f}%"
                    })
                    probabilidades_grafico.append(prob_alta)
                    nombres_grafico.append(name)

                df_res_usuario = pd.DataFrame(resultados_usuario)

                pred_final = models_dict['Ensamble Híbrido'].predict(processed_input)[0]
                prob_final = models_dict['Ensamble Híbrido'].predict_proba(processed_input)[0]

                st.markdown("<h3 class='section-header'>2. Diagnóstico de Capacidad Económica</h3>", unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class='kpi-container'>
                    <div class='kpi-card'>
                        <div class='kpi-value'>{age} años</div>
                        <div class='kpi-label'>Edad del Solicitante</div>
                    </div>
                    <div class='kpi-card'>
                        <div class='kpi-value'>{education}</div>
                        <div class='kpi-label'>Grado de Instrucción</div>
                    </div>
                    <div class='kpi-card'>
                        <div class='kpi-value'>{hours_per_week} hrs</div>
                        <div class='kpi-label'>Horas de Trabajo Semanal</div>
                    </div>
                    <div class='kpi-card'>
                        <div class='kpi-value'>${capital_gain:,.0f} USD</div>
                        <div class='kpi-label'>Ganancias de Capital</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                col_res1, col_res2 = st.columns(2)

                with col_res1:
                    if pred_final == 1:
                        st.markdown(f"""
                        <div class='result-card-high'>
                            <h4 style='color: #047857; margin-top:0;'>🟢 CAPACIDAD DE INGRESO ALTO (>50K USD Anuales)</h4>
                            <p style='color: #065F46; font-size: 0.95rem; line-height: 1.5;'>
                                El algoritmo de <b>Ensamble Híbrido</b> ha clasificado al solicitante dentro del segmento de ingresos superiores con un <b>{prob_final[1]*100:.2f}% de probabilidad</b>.<br>
                                Este perfil se asocia con estabilidad laboral e ingresos compatibles con productos financieros de gama alta.
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class='result-card-low'>
                            <h4 style='color: #B45309; margin-top:0;'>🟡 CAPACIDAD DE INGRESO ESTÁNDAR (<=50K USD Anuales)</h4>
                            <p style='color: #92400E; font-size: 0.95rem; line-height: 1.5;'>
                                El algoritmo de <b>Ensamble Híbrido</b> ha clasificado al solicitante dentro del segmento de ingresos moderados con un <b>{prob_final[0]*100:.2f}% de certeza</b>.<br>
                                Se sugiere precaución al asignar líneas de financiamiento elevadas sin aval adicional.
                            </p>
                        </div>
                        """, unsafe_allow_html=True)

                    st.write("**Detalle de estimaciones por modelo:**")
                    st.dataframe(df_res_usuario, hide_index=True, use_container_width=True)

                with col_res2:
                    st.write("**Comparativa de Probabilidad del Cliente (>50K):**")
                    
                    fig, ax = plt.subplots(figsize=(7, 4.2))
                    sns.set_style("whitegrid")
                    
                    colores_grafico = ['#B0C4DE', '#B0C4DE', '#B0C4DE', '#B0C4DE', '#1E3A8A']
                    
                    bars = ax.barh(nombres_grafico, probabilidades_grafico, color=colores_grafico, edgecolor='black', height=0.55)
                    ax.set_xlim(0, 100)
                    ax.set_xlabel("Probabilidad de Ingreso Alto (%)")
                    ax.set_title("Predicciones estimadas por modelo para este cliente")
                    
                    for bar in bars:
                        width = bar.get_width()
                        ax.text(width + 2, bar.get_y() + bar.get_height()/2, f'{width:.1f}%', 
                                va='center', ha='left', fontsize=9, fontweight='bold' if width > 50 else 'normal')
                                
                    plt.tight_layout()
                    st.pyplot(fig)

                st.markdown("<h3 class='section-header'>3. Recomendaciones de Negocio para CrediNova</h3>", unsafe_allow_html=True)
                
                if pred_final == 1:
                    st.markdown("""
                    - **Estrategia de Ofertas Premium:** Habilitar de manera directa ofertas para tarjetas de crédito de categoría superior (Signature/Black) con límites de crédito adaptados a su capacidad.
                    - **Campañas Cruzadas de Inversión:** Dirigir al cliente a flujos de captación para productos financieros de mediano plazo (como depósitos a plazo u opciones de inversión).
                    - **Flujo de Aprobación Expedito:** Agilizar el tiempo de procesamiento y reducir el número de verificaciones manuales para garantizar una excelente experiencia de usuario.
                    """)
                else:
                    st.markdown("""
                    - **Estrategia de Mitigación de Riesgos:** Limitar la exposición inicial ofreciendo productos de consumo tradicionales (tarjetas básicas) con cupos iniciales moderados.
                    - **Estructura de Garantías:** Solicitar avales o garantías reales si el cliente requiere un financiamiento vehicular o de montos significativos.
                    - **Monitoreo de Comportamiento:** Programar evaluaciones periódicas del uso de la línea a los 6 meses de uso antes de proponer incrementos.
                    """)

            except Exception as e:
                st.error(f"Error en la ejecución del análisis: {str(e)}")
