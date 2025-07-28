"""
Analyse du dataset Iris
======================
Ce script charge le dataset Iris et utilise diverses fonctions pour comprendre sa structure.
"""

import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
import matplotlib.pyplot as plt
import seaborn as sns

def load_iris_dataset():
    """
    Charge le dataset Iris depuis UCI ML Repository
    
    Returns:
        tuple: (X, y, metadata, variables) où X contient les caractéristiques,
               y les targets, metadata les métadonnées et variables les informations sur les variables
    """
    print("Chargement du dataset Iris...")
    
    # Fetch dataset from UCI ML Repository
    iris = fetch_ucirepo(id=53)
    
    # Data (as pandas dataframes)
    X = iris.data.features
    y = iris.data.targets
    
    print("Dataset chargé avec succès!\n")
    
    return X, y, iris.metadata, iris.variables

def display_dataset_info(X, y):
    """
    Affiche les informations générales sur le dataset
    
    Args:
        X (DataFrame): Caractéristiques du dataset
        y (DataFrame): Variables cibles du dataset
    """
    print("="*60)
    print("INFORMATIONS GÉNÉRALES SUR LE DATASET")
    print("="*60)
    
    # Combiner X et y pour avoir le dataset complet
    dataset = pd.concat([X, y], axis=1)
    
    print(f"Nombre total d'observations: {len(dataset)}")
    print(f"Nombre de caractéristiques (features): {X.shape[1]}")
    print(f"Nombre de variables cibles: {y.shape[1]}")
    print(f"Dimensions complètes du dataset: {dataset.shape}")
    print()

def display_first_rows(X, y, n_rows=10):
    """
    Affiche les premières lignes du dataset
    
    Args:
        X (DataFrame): Caractéristiques du dataset
        y (DataFrame): Variables cibles du dataset
        n_rows (int): Nombre de lignes à afficher
    """
    print("="*60)
    print(f"PREMIÈRES {n_rows} LIGNES DU DATASET")
    print("="*60)
    
    # Combiner X et y pour affichage
    dataset = pd.concat([X, y], axis=1)
    print(dataset.head(n_rows))
    print()

def analyze_data_types(X, y):
    """
    Analyse et affiche les types de données de chaque caractéristique
    
    Args:
        X (DataFrame): Caractéristiques du dataset
        y (DataFrame): Variables cibles du dataset
    """
    print("="*60)
    print("TYPES DE DONNÉES")
    print("="*60)
    
    # Combiner X et y pour analyse complète
    dataset = pd.concat([X, y], axis=1)
    
    print("Types de données par colonne:")
    for column in dataset.columns:
        dtype = dataset[column].dtype
        print(f"  {column}: {dtype}")
    
    print("\nRésumé des types:")
    print(dataset.dtypes.value_counts())
    print()

def display_statistical_summary(X, y):
    """
    Affiche un résumé statistique du dataset
    
    Args:
        X (DataFrame): Caractéristiques du dataset
        y (DataFrame): Variables cibles du dataset
    """
    print("="*60)
    print("RÉSUMÉ STATISTIQUE")
    print("="*60)
    
    # Statistiques descriptives pour les caractéristiques numériques
    print("Statistiques descriptives des caractéristiques:")
    print(X.describe())
    print()
    
    # Information sur les valeurs manquantes
    print("Valeurs manquantes:")
    dataset = pd.concat([X, y], axis=1)
    missing_values = dataset.isnull().sum()
    if missing_values.sum() == 0:
        print("Aucune valeur manquante détectée!")
    else:
        print(missing_values[missing_values > 0])
    print()

def analyze_target_distribution(y):
    """
    Analyse la distribution des variables cibles
    
    Args:
        y (DataFrame): Variables cibles du dataset
    """
    print("="*60)
    print("DISTRIBUTION DES CLASSES CIBLES")
    print("="*60)
    
    for column in y.columns:
        print(f"Distribution de {column}:")
        print(y[column].value_counts().sort_index())
        print(f"Pourcentages:")
        print(y[column].value_counts(normalize=True).sort_index() * 100)
        print()

def display_metadata_and_variables(metadata, variables):
    """
    Affiche les métadonnées et informations sur les variables
    
    Args:
        metadata: Métadonnées du dataset
        variables: Informations sur les variables
    """
    print("="*60)
    print("MÉTADONNÉES DU DATASET")
    print("="*60)
    print(metadata)
    print()
    
    print("="*60)
    print("INFORMATIONS SUR LES VARIABLES")
    print("="*60)
    print(variables)
    print()

def create_basic_visualizations(X, y):
    """
    Crée des visualisations basiques du dataset
    
    Args:
        X (DataFrame): Caractéristiques du dataset
        y (DataFrame): Variables cibles du dataset
    """
    print("="*60)
    print("CRÉATION DE VISUALISATIONS BASIQUES")
    print("="*60)
    
    # Créer une figure avec plusieurs sous-graphiques
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Analyse visuelle du dataset Iris', fontsize=16)
    
    # Combiner X et y
    dataset = pd.concat([X, y], axis=1)
    
    # 1. Histogrammes des caractéristiques
    dataset.iloc[:, :-1].hist(ax=axes[0, 0], bins=20)
    axes[0, 0].set_title('Distribution des caractéristiques')
    
    # 2. Boxplot des caractéristiques par classe
    melted_data = pd.melt(dataset, id_vars=[dataset.columns[-1]], 
                         value_vars=dataset.columns[:-1])
    sns.boxplot(data=melted_data, x='variable', y='value', 
                hue=dataset.columns[-1], ax=axes[0, 1])
    axes[0, 1].set_title('Boxplot par classe')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 3. Matrice de corrélation
    corr_matrix = X.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=axes[1, 0])
    axes[1, 0].set_title('Matrice de corrélation')
    
    # 4. Scatter plot des deux premières caractéristiques
    scatter = axes[1, 1].scatter(X.iloc[:, 0], X.iloc[:, 1], 
                                c=pd.Categorical(y.iloc[:, 0]).codes, 
                                cmap='viridis', alpha=0.7)
    axes[1, 1].set_xlabel(X.columns[0])
    axes[1, 1].set_ylabel(X.columns[1])
    axes[1, 1].set_title('Scatter plot des 2 premières caractéristiques')
    plt.colorbar(scatter, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.savefig('iris_analysis_plots.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Visualisations créées et sauvegardées dans 'iris_analysis_plots.png'")

def main():
    """
    Fonction principale qui exécute toute l'analyse
    """
    print("ANALYSE COMPLÈTE DU DATASET IRIS")
    print("=" * 80)
    print()
    
    try:
        # 1. Charger le dataset
        X, y, metadata, variables = load_iris_dataset()
        
        # 2. Afficher les informations générales
        display_dataset_info(X, y)
        
        # 3. Afficher les premières lignes
        display_first_rows(X, y, n_rows=10)
        
        # 4. Analyser les types de données
        analyze_data_types(X, y)
        
        # 5. Résumé statistique
        display_statistical_summary(X, y)
        
        # 6. Distribution des classes cibles
        analyze_target_distribution(y)
        
        # 7. Métadonnées et variables
        display_metadata_and_variables(metadata, variables)
        
        # 8. Créer des visualisations
        create_basic_visualizations(X, y)
        
        print("\n" + "="*80)
        print("ANALYSE TERMINÉE AVEC SUCCÈS!")
        print("="*80)
        
    except Exception as e:
        print(f"Erreur lors de l'analyse: {e}")
        print("Assurez-vous que toutes les librairies nécessaires sont installées.")

if __name__ == "__main__":
    main()
