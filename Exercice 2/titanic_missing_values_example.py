"""
Exemple pratique : Traitement des valeurs manquantes sur le dataset Titanic
===========================================================================

Ce script utilise le dataset Titanic pour démontrer l'impact des différentes
stratégies de traitement des valeurs manquantes sur un cas réel.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

def load_titanic_dataset():
    """
    Charge le dataset Titanic depuis seaborn
    """
    try:
        # Charger le dataset Titanic
        titanic = sns.load_dataset('titanic')
        print("Dataset Titanic chargé avec succès!")
        print(f"Forme du dataset: {titanic.shape}")
        return titanic
    except Exception as e:
        print(f"Erreur lors du chargement: {e}")
        # Créer un dataset Titanic simulé
        return create_simulated_titanic()

def create_simulated_titanic():
    """
    Crée un dataset Titanic simulé si le vrai n'est pas disponible
    """
    print("Création d'un dataset Titanic simulé...")
    np.random.seed(42)
    n = 891
    
    data = {
        'survived': np.random.choice([0, 1], n, p=[0.62, 0.38]),
        'pclass': np.random.choice([1, 2, 3], n, p=[0.24, 0.21, 0.55]),
        'sex': np.random.choice(['male', 'female'], n, p=[0.65, 0.35]),
        'age': np.random.normal(29.7, 14.5, n),
        'sibsp': np.random.poisson(0.5, n),
        'parch': np.random.poisson(0.4, n),
        'fare': np.random.lognormal(3.0, 1.0, n),
        'embarked': np.random.choice(['S', 'C', 'Q'], n, p=[0.72, 0.19, 0.09]),
        'class': np.random.choice(['First', 'Second', 'Third'], n, p=[0.24, 0.21, 0.55]),
        'deck': np.random.choice(['A', 'B', 'C', 'D', 'E', 'F', 'G'], n)
    }
    
    df = pd.DataFrame(data)
    
    # Introduire des valeurs manquantes réalistes
    # Age manquant (20% des cas)
    age_missing = np.random.choice(df.index, int(0.2 * len(df)), replace=False)
    df.loc[age_missing, 'age'] = np.nan
    
    # Deck manquant (77% des cas comme dans le vrai dataset)
    deck_missing = np.random.choice(df.index, int(0.77 * len(df)), replace=False)
    df.loc[deck_missing, 'deck'] = np.nan
    
    # Embarked manquant (quelques cas)
    embarked_missing = np.random.choice(df.index, 2, replace=False)
    df.loc[embarked_missing, 'embarked'] = np.nan
    
    return df

def analyze_titanic_missing_values(df):
    """
    Analyse spécifique des valeurs manquantes du Titanic
    """
    print("="*60)
    print("ANALYSE DES VALEURS MANQUANTES - DATASET TITANIC")
    print("="*60)
    
    # Informations générales
    print(f"Dimensions: {df.shape}")
    print(f"Total de valeurs manquantes: {df.isnull().sum().sum()}")
    
    # Détail par colonne
    missing_info = pd.DataFrame({
        'Colonne': df.columns,
        'Valeurs_Manquantes': df.isnull().sum(),
        'Pourcentage': (df.isnull().sum() / len(df)) * 100
    }).sort_values('Valeurs_Manquantes', ascending=False)
    
    print("\nValeurs manquantes par colonne:")
    print(missing_info[missing_info['Valeurs_Manquantes'] > 0])
    
    # Visualisation
    plt.figure(figsize=(15, 10))
    
    # Heatmap des valeurs manquantes
    plt.subplot(2, 2, 1)
    sns.heatmap(df.isnull(), yticklabels=False, cbar=True, cmap='viridis')
    plt.title('Heatmap des Valeurs Manquantes')
    
    # Barplot des valeurs manquantes
    plt.subplot(2, 2, 2)
    missing_counts = df.isnull().sum()
    missing_counts = missing_counts[missing_counts > 0].sort_values()
    missing_counts.plot(kind='barh', color='coral')
    plt.title('Nombre de Valeurs Manquantes par Colonne')
    
    # Distribution de l'âge avec et sans valeurs manquantes
    plt.subplot(2, 2, 3)
    df['age'].hist(bins=30, alpha=0.7, label='Âges disponibles')
    plt.xlabel('Âge')
    plt.ylabel('Fréquence')
    plt.title('Distribution de l\'Âge')
    plt.legend()
    
    # Survie par classe (pour comprendre l'importance des variables)
    plt.subplot(2, 2, 4)
    if 'survived' in df.columns and 'pclass' in df.columns:
        survival_by_class = df.groupby('pclass')['survived'].mean()
        survival_by_class.plot(kind='bar', color='lightgreen')
        plt.title('Taux de Survie par Classe')
        plt.ylabel('Taux de Survie')
        plt.xticks(rotation=0)
    
    plt.tight_layout()
    plt.show()
    
    return missing_info

def compare_titanic_strategies(df):
    """
    Compare différentes stratégies sur le dataset Titanic
    """
    print("\n" + "="*70)
    print("COMPARAISON DES STRATÉGIES SUR LE DATASET TITANIC")
    print("="*70)
    
    strategies_results = {}
    target_col = 'survived'
    
    if target_col not in df.columns:
        print(f"Colonne cible '{target_col}' non trouvée. Utilisation d'une cible simulée.")
        df[target_col] = np.random.choice([0, 1], len(df))
    
    # Stratégie 1: Suppression des lignes
    print("\n1. Suppression des lignes avec valeurs manquantes:")
    df_drop_rows = df.dropna()
    strategies_results['Suppression_Lignes'] = df_drop_rows
    print(f"   Données conservées: {len(df_drop_rows)}/{len(df)} ({(len(df_drop_rows)/len(df))*100:.1f}%)")
    
    # Stratégie 2: Suppression des colonnes avec >50% de valeurs manquantes
    print("\n2. Suppression des colonnes avec >50% de valeurs manquantes:")
    missing_threshold = 0.5
    missing_pct = df.isnull().sum() / len(df)
    cols_to_drop = missing_pct[missing_pct > missing_threshold].index
    df_drop_cols = df.drop(columns=cols_to_drop)
    strategies_results['Suppression_Colonnes'] = df_drop_cols
    print(f"   Colonnes supprimées: {list(cols_to_drop)}")
    print(f"   Colonnes conservées: {df_drop_cols.shape[1]}/{df.shape[1]}")
    
    # Stratégie 3: Imputation simple
    print("\n3. Imputation simple (moyenne/mode):")
    df_impute_simple = df.copy()
    
    # Imputer l'âge par la moyenne
    if 'age' in df_impute_simple.columns:
        age_mean = df_impute_simple['age'].mean()
        df_impute_simple['age'].fillna(age_mean, inplace=True)
        print(f"   Âge imputé par la moyenne: {age_mean:.1f}")
    
    # Imputer les variables catégorielles par le mode
    categorical_cols = df_impute_simple.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df_impute_simple[col].isnull().any():
            mode_value = df_impute_simple[col].mode()
            if not mode_value.empty:
                df_impute_simple[col].fillna(mode_value[0], inplace=True)
                print(f"   {col} imputé par le mode: {mode_value[0]}")
    
    strategies_results['Imputation_Simple'] = df_impute_simple
    
    # Stratégie 4: Imputation par groupe (âge par classe et sexe)
    print("\n4. Imputation par groupe (âge par classe et sexe):")
    df_impute_group = df.copy()
    
    if 'age' in df_impute_group.columns and 'pclass' in df_impute_group.columns and 'sex' in df_impute_group.columns:
        # Calculer la moyenne de l'âge par groupe
        age_by_group = df_impute_group.groupby(['pclass', 'sex'])['age'].mean()
        
        # Imputer les valeurs manquantes
        mask = df_impute_group['age'].isnull()
        for idx in df_impute_group[mask].index:
            pclass = df_impute_group.loc[idx, 'pclass']
            sex = df_impute_group.loc[idx, 'sex']
            if (pclass, sex) in age_by_group.index:
                df_impute_group.loc[idx, 'age'] = age_by_group[(pclass, sex)]
        
        print(f"   Âge imputé par groupe (classe/sexe)")
        print(f"   Valeurs manquantes restantes pour age: {df_impute_group['age'].isnull().sum()}")
    
    # Imputer les autres colonnes par le mode
    for col in categorical_cols:
        if df_impute_group[col].isnull().any():
            mode_value = df_impute_group[col].mode()
            if not mode_value.empty:
                df_impute_group[col].fillna(mode_value[0], inplace=True)
    
    strategies_results['Imputation_Groupe'] = df_impute_group
    
    # Évaluation avec un modèle de classification
    print("\n" + "="*50)
    print("ÉVALUATION AVEC UN MODÈLE DE CLASSIFICATION")
    print("="*50)
    
    model_results = []
    
    for strategy_name, data in strategies_results.items():
        try:
            if data is None or len(data) == 0:
                continue
            
            # Préparer les features et target
            if target_col in data.columns:
                X = data.drop(columns=[target_col])
                y = data[target_col]
            else:
                continue
            
            # Supprimer les colonnes encore avec des valeurs manquantes
            cols_with_missing = X.columns[X.isnull().any()].tolist()
            if cols_with_missing:
                print(f"   {strategy_name}: Suppression des colonnes avec valeurs manquantes: {cols_with_missing}")
                X = X.drop(columns=cols_with_missing)
            
            # Encoder les variables catégorielles
            X_encoded = X.copy()
            for col in X_encoded.select_dtypes(include=['object']).columns:
                le = LabelEncoder()
                X_encoded[col] = le.fit_transform(X_encoded[col])
            
            # Vérifier qu'il y a assez de données
            if len(X_encoded) < 50:
                print(f"   {strategy_name}: Pas assez de données ({len(X_encoded)} observations)")
                continue
            
            # Split train/test
            X_train, X_test, y_train, y_test = train_test_split(
                X_encoded, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Entraîner le modèle
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Évaluation
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            model_results.append({
                'Stratégie': strategy_name,
                'Accuracy': accuracy,
                'Nb_Observations': len(data),
                'Nb_Features': len(X.columns),
                'Taille_Train': len(X_train)
            })
            
            print(f"   ✅ {strategy_name}: Accuracy = {accuracy:.4f}")
            
        except Exception as e:
            print(f"   ❌ {strategy_name}: Erreur - {e}")
    
    # Afficher les résultats comparatifs
    if model_results:
        results_df = pd.DataFrame(model_results)
        print(f"\nRésultats comparatifs:")
        print(results_df.round(4))
        
        # Visualisation des résultats
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        results_df.plot(x='Stratégie', y='Accuracy', kind='bar', ax=plt.gca(), color='lightblue')
        plt.title('Accuracy par Stratégie')
        plt.xticks(rotation=45)
        plt.ylabel('Accuracy')
        
        plt.subplot(1, 3, 2)
        results_df.plot(x='Stratégie', y='Nb_Observations', kind='bar', ax=plt.gca(), color='lightgreen')
        plt.title('Nombre d\'Observations')
        plt.xticks(rotation=45)
        plt.ylabel('Nombre d\'observations')
        
        plt.subplot(1, 3, 3)
        plt.scatter(results_df['Nb_Observations'], results_df['Accuracy'], s=100, alpha=0.7)
        for i, row in results_df.iterrows():
            plt.annotate(row['Stratégie'], (row['Nb_Observations'], row['Accuracy']), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        plt.xlabel('Nombre d\'Observations')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs Taille des Données')
        
        plt.tight_layout()
        plt.show()
        
        return results_df
    
    return None

def titanic_specific_insights(df):
    """
    Insights spécifiques au dataset Titanic
    """
    print("\n" + "="*70)
    print("INSIGHTS SPÉCIFIQUES AU DATASET TITANIC")
    print("="*70)
    
    print("💡 OBSERVATIONS SUR LES VALEURS MANQUANTES:")
    print("-" * 45)
    
    if 'age' in df.columns:
        age_missing_pct = (df['age'].isnull().sum() / len(df)) * 100
        print(f"• Âge manquant: {age_missing_pct:.1f}% des passagers")
        print("  → Impact: Variable importante pour la survie")
        print("  → Stratégie recommandée: Imputation par groupe (classe/sexe)")
    
    if 'deck' in df.columns:
        deck_missing_pct = (df['deck'].isnull().sum() / len(df)) * 100
        print(f"• Pont (deck) manquant: {deck_missing_pct:.1f}% des passagers")
        print("  → Impact: Très forte corrélation avec la classe")
        print("  → Stratégie recommandée: Suppression ou création d'une catégorie 'Unknown'")
    
    if 'embarked' in df.columns:
        embarked_missing_pct = (df['embarked'].isnull().sum() / len(df)) * 100
        print(f"• Port d'embarquement manquant: {embarked_missing_pct:.1f}% des passagers")
        print("  → Impact: Faible impact sur la survie")
        print("  → Stratégie recommandée: Imputation par le mode")
    
    print("\n🎯 RECOMMANDATIONS POUR LE DATASET TITANIC:")
    print("-" * 50)
    print("1. Ne pas supprimer les lignes (perte de 20% des données)")
    print("2. Imputer l'âge par groupe (classe + sexe) pour préserver les patterns")
    print("3. Créer une variable 'Age_Missing' comme indicateur")
    print("4. Supprimer la colonne 'deck' si >70% de valeurs manquantes")
    print("5. Utiliser la stratégie d'imputation par groupe comme référence")

def main():
    """
    Fonction principale pour l'analyse du Titanic
    """
    print("ANALYSE DES VALEURS MANQUANTES - CAS PRATIQUE: TITANIC")
    print("="*70)
    
    # Charger les données
    df = load_titanic_dataset()
    
    # Analyser les valeurs manquantes
    missing_info = analyze_titanic_missing_values(df)
    
    # Comparer les stratégies
    results = compare_titanic_strategies(df)
    
    # Insights spécifiques
    titanic_specific_insights(df)
    
    print("\n" + "="*70)
    print("✅ ANALYSE TERMINÉE!")
    print("📊 Vous avez maintenant une compréhension pratique du traitement")
    print("   des valeurs manquantes sur un cas réel d'analyse de données.")
    print("="*70)

if __name__ == "__main__":
    main()
