# OpenClassrooms Projet 7 - Implémentez un modèle de scoring

## Objectif du Projet

La mission principale est de développer un modèle de scoring qui prédit la probabilité de faillite d'un client d'une société financière qui propose des crédits à la consommation. Le but est de classer automatiquement les demandes de crédit en "accordé" ou "refusé" en utilisant des données variées telles que des données comportementales et celles provenant d'autres institutions financières.

## Missions

### Mission 1: Modélisation de Scoring

1. **Construction du Modèle de Scoring**: 
   - Utiliser des sources de données variées pour entraîner le modèle.
   - Prédire la probabilité de faillite d'un client.

2. **Analyse des Features**:
   - Identifier les features les plus importantes pour le modèle globalement (feature importance globale).
   - Analyser la contribution des features au niveau individuel (feature importance locale).

3. **Mise en Production**:
   - Développer une API pour le modèle de scoring.
   - Créer une interface de test pour cette API.
   - Mettre en œuvre une approche MLOps complète, incluant le suivi des expérimentations et l'analyse en production du data drift.

### Mission 2: Suivi de la Performance en Production

1. **Détection du Data Drift**:
   - Utiliser la librairie Evidently pour détecter le data drift en production.
   - Comparer le dataset "application_train" (données de modélisation) avec le dataset "application_test" (données de nouveaux clients).
   - Générer un tableau HTML d'analyse du data drift avec Evidently.

## Données

Les données utilisées pour ce projet sont disponibles sur [Kaggle](https://www.kaggle.com/c/home-credit-default-risk/data). Elles comprennent des informations détaillées sur les demandes de crédit et permettent de construire et d'évaluer le modèle de scoring.

## Structure des Dossiers

Le projet est organisé comme suit :

``` 
root
│
├── notebooks/
│ └── Garcia_Sébastien_2_notebook_préparation_des_données_062024.ipynb
│ └── Garcia_Sébastien_3_notebook_modélisation_062024.ipynb
│
├── requirements/
│ └── liste_des_packages.txt
```

- **notebooks/** : Contient les notebooks et scripts relatifs à la préparation des données et à la modélisation
  - `Garcia_Sébastien_2_notebook_préparation_des_données_062024.ipynb` : Notebook de préparation des données.
  - `Garcia_Sébastien_3_notebook_modélisation_062024.ipynb` : Notebook de modélisation du scoring.

- **requirements/** : Contient la liste des packages nécessaires pour exécuter le projet.
  - `liste_des_packages.txt` : Liste des packages requis.

## Instructions pour Démarrer

1. **Installation des Dépendances** :
   - Assurez-vous d'avoir installé tous les packages listés dans `requirements/liste_des_packages.txt`.

2. **Exécution des Notebooks** :
   - Commencez par exécuter le notebook de préparation des données (`notebooks/Garcia_Sébastien_2_notebook_préparation_des_données_062024.ipynb`).
   - Ensuite, passez au notebook de modélisation (`notebooks/Garcia_Sébastien_3_notebook_modélisation_062024.ipynb`).
