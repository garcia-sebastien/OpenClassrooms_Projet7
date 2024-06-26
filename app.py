import streamlit as st
import pandas as pd
import mlflow.sklearn
import json
import requests
import matplotlib.pyplot as plt
import numpy as np

# Titre de l'application
st.title("OpenClassrooms Projet 7 - Probabilité de faillite d'un client")
st.markdown("""---""")

# Initialisation des variables
uploaded_file = None
id_string = '0'
submit = None
prediction = 'nan'
probabilites = 'nan'

# ----------- Sidebar
st.sidebar.write("Créé par Sébastien Garcia")

# ----------- File upload
uploaded_file = st.file_uploader("Téléchargez le fichier de données d'entrée.", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.markdown("""---""")
    id_string = st.text_input("Spécifier l'ID du client (SK_ID_CURR).")
  
st.markdown("""---""")
if len(id_string) >= 6:
    submit = st.button("Obtenir les prédictions pour le client " + id_string)

# ----------- Submission
if submit:
    with st.spinner(text="Récupération de la prédiction du modèle..."):
        # Sélection du client
        X = df.copy()
        X = X.loc[df['SK_ID_CURR'] == int(id_string)]
    
        # Suppression de la colonne SK_ID_CURR
        X = X.drop(columns=['SK_ID_CURR'])
        # Suppression de la colonne TARGET si elle est présente
        if 'TARGET' in X.columns:
            X = X.drop(columns=['TARGET'])

        # Définition du nom des colonnes numériques et catégorielles
        num_cols = X.select_dtypes(include=['number']).columns
        cat_cols = X.select_dtypes(include=['object']).columns

        # URL de l'API MLflow
        MLFLOW_URI = "http://127.0.0.1:1234/invocations"

        # Convertir le DataFrame en JSON
        data_json = X.to_dict(orient="records")

        # Préparer la charge utile JSON pour MLflow 2.0
        payload = {
            "dataframe_records": data_json
        }

        # Définir les en-têtes de la requête
        headers = {
                "Content-Type": "application/json"
        }

        # Faire la requête POST pour obtenir les prédictions
        response = requests.post(MLFLOW_URI, headers=headers, data=json.dumps(payload))  
        
        # Prédiction
        prediction = response.json()
        prediction = prediction['predictions'][0]
    
    # ----------- Ouputs
    if prediction == 0:
        st.success("Crédit accordé")
    elif prediction == 1:
        st.error("Crédit refusé")

    # ----------- Graphique probabilité
    st.markdown("Probabilité de faillite du client " + id_string + " :")

    # Chargement de la pipeline
    mlflow.set_tracking_uri("http://localhost:5000")
    model_name = "finalModel"
    stage = "Production"
    pipeline = mlflow.sklearn.load_model(
        model_uri=f"models:/{model_name}/{stage}"
    )
    
    # Probabilités des classes
    probabilites = pipeline.predict_proba(X)
    
    # Labels pour les sections du graphique
    labels = ['Non-Faillite', 'Faillite']

    # Couleurs pour les sections du graphique
    colors = ['#008bfb','#ff0051']

    # Créer la figure
    fig, ax = plt.subplots()
    ax.pie(probabilites[0], labels=labels, colors=colors, startangle=90, counterclock=False, wedgeprops=dict(width=0.3))

    # Ajouter un cercle blanc au centre pour faire un effet de demi-camembert
    centre_circle = plt.Circle((0,0),0.70,fc='white')
    ax.text(-0.15, -0.01, str(round(probabilites[0][1]*100)) + '%', size='xx-large')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)

    # Egalité des axes pour faire un cercle parfait
    ax.axis('equal')

    # Afficher le graphique
    st.pyplot(fig)
    
    # ----------- Graphique feature importance locale
    st.markdown("""---""")
    with st.spinner(text="Récupération du détail des variables les plus importantes..."):
        from sklearn.model_selection import train_test_split
        import shap
    
        data = df.copy()
        data = data.loc[df['SK_ID_CURR'] != int(id_string)]
    
        # Suppression de la colonne SK_ID_CURR
        data = data.drop(columns=['SK_ID_CURR'])
    
        # Réduction du dataset pour diminuer les temps d'entraînement de SHAP Explainer
        data = data.sample(frac=0.1, random_state=42)
    
        # Séparation en ensemble d'entraînement et de test
        X_train, X_test = train_test_split(data, test_size=0.05, random_state=42)
    
        # Ajouter les données du client aux données de test pour SHAP
        X_test = pd.concat([X, X_test], ignore_index=True)

        def process_shap_data(df):
            """
            Applique les étapes de feature engineering et de preprocessing à un DataFrame et renvoie
            un DataFrame transformé avec les colonnes appropriées.

            Parameters
            ----------
            df : pandas.DataFrame
                Le DataFrame d'entrée contenant les données brutes.

            Returns
            -------
            pandas.DataFrame
                Le DataFrame transformé avec les colonnes après feature engineering et preprocessing.
            """
            # Appliquer l'étape de feature_engineering
            df_transformed = pipeline.named_steps['feature_engineering'].transform(df)

            # Appliquer l'étape de preprocessing
            df_transformed = pipeline.named_steps['preprocessing'].transform(df_transformed)
        
            # Obtenir les noms des colonnes après preprocessing
            num_cols_transformed = pipeline.named_steps['preprocessing'].transformers_[0][1] \
                                    .named_steps['simpleimputer'].get_feature_names_out(num_cols)
            cat_cols_transformed = pipeline.named_steps['preprocessing'].transformers_[1][1] \
                                    .named_steps['onehotencoder'].get_feature_names_out(cat_cols)
    
            # Concaténer les noms des colonnes numériques et catégorielles
            transformed_columns = np.concatenate((num_cols_transformed, cat_cols_transformed))
    
            # Créer un DataFrame avec les colonnes transformées
            df_transformed = pd.DataFrame(df_transformed, columns=transformed_columns)
    
            return df_transformed

        X_train = process_shap_data(X_train)
        X_test = process_shap_data(X_test)

        # Récupération du modèle
        model = pipeline.named_steps['classifier']
        
        # Initialisation de Explainer de SHAP avec le modèle entraîné
        explainer = shap.Explainer(model, X_train)

        # Calculer les valeurs SHAP pour l'ensemble de test
        shap_values = explainer(X_test)
            
    st.write("Variables les plus importantes pour la prédiction du modèle :")
        
    # Créer la figure
    fig_2, ax_2 = plt.subplots()
    shap.plots.waterfall(shap_values[0])
        
    # Afficher le graphique
    st.pyplot(fig_2)