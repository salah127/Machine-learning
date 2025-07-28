"""
Exercice 2 : Identification et Traitement des Valeurs Manquantes
===============================================================

Ce script explore différentes stratégies pour identifier et traiter les valeurs manquantes
dans les jeux de données, en utilisant plusieurs datasets pour illustrer les différentes approches.

Stratégies couvertes :
1. Identification des valeurs manquantes
2. Visualisation des patterns de valeurs manquantes
3. Suppression de lignes/colonnes
4. Imputation par moyenne, médiane, mode
5. Imputation avancée (KNN, itérative)
6. Analyse de l'impact des différentes stratégies
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class MissingValueAnalyzer:
    """
    Classe pour analyser et traiter les valeurs manquantes dans les datasets
    """
    
    def __init__(self, data, target_col=None):
        """
        Initialise l'analyseur avec un dataset
        
        Args:
            data (DataFrame): Le dataset à analyser
            target_col (str): Nom de la colonne cible (optionnel)
        """
        self.original_data = data.copy()
        self.data = data.copy()
        self.target_col = target_col
        self.missing_stats = {}
        
    def analyze_missing_values(self):
        """
        Analyse complète des valeurs manquantes
        """
        print("="*70)
        print("ANALYSE DES VALEURS MANQUANTES")
        print("="*70)
        
        # Statistiques générales
        total_cells = np.product(self.data.shape)
        total_missing = self.data.isnull().sum().sum()
        missing_percentage = (total_missing / total_cells) * 100
        
        print(f"Dimensions du dataset: {self.data.shape}")
        print(f"Total de cellules: {total_cells:,}")
        print(f"Total de valeurs manquantes: {total_missing:,}")
        print(f"Pourcentage global de valeurs manquantes: {missing_percentage:.2f}%")
        print()
        
        # Statistiques par colonne
        missing_by_column = self.data.isnull().sum()
        missing_percentage_by_column = (missing_by_column / len(self.data)) * 100
        
        missing_df = pd.DataFrame({
            'Colonne': missing_by_column.index,
            'Valeurs_Manquantes': missing_by_column.values,
            'Pourcentage': missing_percentage_by_column.values
        }).sort_values('Valeurs_Manquantes', ascending=False)
        
        print("Valeurs manquantes par colonne:")
        print("-" * 50)
        print(missing_df[missing_df['Valeurs_Manquantes'] > 0])
        
        # Sauvegarder les statistiques
        self.missing_stats = {
            'total_missing': total_missing,
            'missing_percentage': missing_percentage,
            'missing_by_column': missing_df
        }
        
        return missing_df
    
    def visualize_missing_patterns(self):
        """
        Visualise les patterns de valeurs manquantes
        """
        print("\n" + "="*70)
        print("VISUALISATION DES PATTERNS DE VALEURS MANQUANTES")
        print("="*70)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Analyse des Valeurs Manquantes', fontsize=16)
        
        # 1. Heatmap des valeurs manquantes
        missing_data = self.data.isnull()
        sns.heatmap(missing_data, yticklabels=False, cbar=True, cmap='viridis', ax=axes[0, 0])
        axes[0, 0].set_title('Heatmap des Valeurs Manquantes')
        
        # 2. Barplot des valeurs manquantes par colonne
        missing_counts = self.data.isnull().sum()
        missing_counts = missing_counts[missing_counts > 0].sort_values(ascending=True)
        if len(missing_counts) > 0:
            missing_counts.plot(kind='barh', ax=axes[0, 1])
            axes[0, 1].set_title('Nombre de Valeurs Manquantes par Colonne')
            axes[0, 1].set_xlabel('Nombre de valeurs manquantes')
        
        # 3. Matrice de corrélation des patterns manquants
        if missing_data.sum().sum() > 0:
            missing_corr = missing_data.corr()
            sns.heatmap(missing_corr, annot=True, cmap='coolwarm', center=0, ax=axes[1, 0])
            axes[1, 0].set_title('Corrélation des Patterns Manquants')
        
        # 4. Distribution des lignes avec valeurs manquantes
        missing_per_row = missing_data.sum(axis=1)
        missing_per_row.hist(bins=20, ax=axes[1, 1])
        axes[1, 1].set_title('Distribution des Valeurs Manquantes par Ligne')
        axes[1, 1].set_xlabel('Nombre de valeurs manquantes par ligne')
        axes[1, 1].set_ylabel('Fréquence')
        
        plt.tight_layout()
        plt.savefig('missing_values_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Visualisations sauvegardées dans 'missing_values_analysis.png'")
    
    def strategy_drop_rows(self, threshold=None):
        """
        Stratégie 1: Suppression des lignes avec valeurs manquantes
        
        Args:
            threshold (float): Seuil de valeurs non-manquantes requises (0-1)
        """
        print("\n" + "="*70)
        print("STRATÉGIE 1: SUPPRESSION DES LIGNES")
        print("="*70)
        
        original_shape = self.data.shape
        
        if threshold is None:
            # Supprimer toutes les lignes avec au moins une valeur manquante
            cleaned_data = self.data.dropna()
            print(f"Suppression de toutes les lignes avec valeurs manquantes")
        else:
            # Supprimer les lignes avec moins de 'threshold' % de valeurs non-manquantes
            min_values = int(threshold * self.data.shape[1])
            cleaned_data = self.data.dropna(thresh=min_values)
            print(f"Suppression des lignes avec moins de {threshold*100:.0f}% de valeurs non-manquantes")
        
        print(f"Forme originale: {original_shape}")
        print(f"Forme après suppression: {cleaned_data.shape}")
        print(f"Lignes supprimées: {original_shape[0] - cleaned_data.shape[0]}")
        print(f"Pourcentage de données conservées: {(cleaned_data.shape[0]/original_shape[0])*100:.2f}%")
        
        return cleaned_data
    
    def strategy_drop_columns(self, threshold=0.5):
        """
        Stratégie 2: Suppression des colonnes avec trop de valeurs manquantes
        
        Args:
            threshold (float): Seuil de valeurs manquantes au-dessus duquel supprimer la colonne
        """
        print("\n" + "="*70)
        print("STRATÉGIE 2: SUPPRESSION DES COLONNES")
        print("="*70)
        
        original_shape = self.data.shape
        
        # Calculer le pourcentage de valeurs manquantes par colonne
        missing_percentage = self.data.isnull().sum() / len(self.data)
        columns_to_drop = missing_percentage[missing_percentage > threshold].index
        
        cleaned_data = self.data.drop(columns=columns_to_drop)
        
        print(f"Seuil de suppression: {threshold*100:.0f}% de valeurs manquantes")
        print(f"Colonnes supprimées: {list(columns_to_drop)}")
        print(f"Forme originale: {original_shape}")
        print(f"Forme après suppression: {cleaned_data.shape}")
        print(f"Colonnes conservées: {cleaned_data.shape[1]}/{original_shape[1]}")
        
        return cleaned_data
    
    def strategy_simple_imputation(self, strategy='mean'):
        """
        Stratégie 3: Imputation simple (moyenne, médiane, mode)
        
        Args:
            strategy (str): 'mean', 'median', 'most_frequent', 'constant'
        """
        print("\n" + "="*70)
        print(f"STRATÉGIE 3: IMPUTATION SIMPLE ({strategy.upper()})")
        print("="*70)
        
        cleaned_data = self.data.copy()
        
        # Séparer les colonnes numériques et catégorielles
        numeric_cols = cleaned_data.select_dtypes(include=[np.number]).columns
        categorical_cols = cleaned_data.select_dtypes(include=['object']).columns
        
        # Imputation pour les colonnes numériques
        if len(numeric_cols) > 0 and strategy in ['mean', 'median']:
            imputer_numeric = SimpleImputer(strategy=strategy)
            cleaned_data[numeric_cols] = imputer_numeric.fit_transform(cleaned_data[numeric_cols])
            print(f"Imputation {strategy} appliquée aux colonnes numériques: {list(numeric_cols)}")
        
        # Imputation pour les colonnes catégorielles
        if len(categorical_cols) > 0:
            imputer_categorical = SimpleImputer(strategy='most_frequent')
            cleaned_data[categorical_cols] = imputer_categorical.fit_transform(cleaned_data[categorical_cols])
            print(f"Imputation par mode appliquée aux colonnes catégorielles: {list(categorical_cols)}")
        
        # Vérification
        remaining_missing = cleaned_data.isnull().sum().sum()
        print(f"Valeurs manquantes restantes: {remaining_missing}")
        
        return cleaned_data
    
    def strategy_knn_imputation(self, n_neighbors=5):
        """
        Stratégie 4: Imputation par KNN
        
        Args:
            n_neighbors (int): Nombre de voisins pour l'imputation KNN
        """
        print("\n" + "="*70)
        print(f"STRATÉGIE 4: IMPUTATION KNN (k={n_neighbors})")
        print("="*70)
        
        # Encoder les variables catégorielles
        data_encoded = self.data.copy()
        label_encoders = {}
        
        categorical_cols = data_encoded.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            # Gérer les valeurs manquantes en les remplaçant temporairement
            mask = data_encoded[col].notna()
            if mask.sum() > 0:  # Si il y a des valeurs non-manquantes
                data_encoded.loc[mask, col] = le.fit_transform(data_encoded.loc[mask, col])
                label_encoders[col] = le
        
        # Convertir en numérique
        numeric_data = data_encoded.apply(pd.to_numeric, errors='coerce')
        
        # Appliquer l'imputation KNN
        imputer = KNNImputer(n_neighbors=n_neighbors)
        imputed_data = imputer.fit_transform(numeric_data)
        
        # Créer le DataFrame résultat
        cleaned_data = pd.DataFrame(imputed_data, columns=self.data.columns, index=self.data.index)
        
        # Décoder les variables catégorielles
        for col, le in label_encoders.items():
            # Arrondir et convertir en entier pour le décodage
            encoded_values = cleaned_data[col].round().astype(int)
            # S'assurer que les valeurs sont dans la plage valide
            encoded_values = np.clip(encoded_values, 0, len(le.classes_) - 1)
            cleaned_data[col] = le.inverse_transform(encoded_values)
        
        remaining_missing = cleaned_data.isnull().sum().sum()
        print(f"Valeurs manquantes restantes: {remaining_missing}")
        print(f"Colonnes traitées: {list(self.data.columns)}")
        
        return cleaned_data
    
    def strategy_iterative_imputation(self, max_iter=10):
        """
        Stratégie 5: Imputation itérative (MICE)
        
        Args:
            max_iter (int): Nombre maximum d'itérations
        """
        print("\n" + "="*70)
        print(f"STRATÉGIE 5: IMPUTATION ITÉRATIVE (MICE, max_iter={max_iter})")
        print("="*70)
        
        # Encoder les variables catégorielles
        data_encoded = self.data.copy()
        label_encoders = {}
        
        categorical_cols = data_encoded.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            mask = data_encoded[col].notna()
            if mask.sum() > 0:
                data_encoded.loc[mask, col] = le.fit_transform(data_encoded.loc[mask, col])
                label_encoders[col] = le
        
        # Convertir en numérique
        numeric_data = data_encoded.apply(pd.to_numeric, errors='coerce')
        
        # Appliquer l'imputation itérative
        imputer = IterativeImputer(max_iter=max_iter, random_state=42)
        imputed_data = imputer.fit_transform(numeric_data)
        
        # Créer le DataFrame résultat
        cleaned_data = pd.DataFrame(imputed_data, columns=self.data.columns, index=self.data.index)
        
        # Décoder les variables catégorielles
        for col, le in label_encoders.items():
            encoded_values = cleaned_data[col].round().astype(int)
            encoded_values = np.clip(encoded_values, 0, len(le.classes_) - 1)
            cleaned_data[col] = le.inverse_transform(encoded_values)
        
        remaining_missing = cleaned_data.isnull().sum().sum()
        print(f"Valeurs manquantes restantes: {remaining_missing}")
        print(f"Nombre d'itérations utilisées: {imputer.n_iter_}")
        
        return cleaned_data
    
    def compare_strategies_impact(self, strategies_results):
        """
        Compare l'impact des différentes stratégies sur les performances du modèle
        
        Args:
            strategies_results (dict): Dictionnaire avec les résultats de chaque stratégie
        """
        print("\n" + "="*70)
        print("COMPARAISON DE L'IMPACT DES STRATÉGIES")
        print("="*70)
        
        if self.target_col is None:
            print("Aucune colonne cible spécifiée. Comparaison basée sur les statistiques descriptives.")
            self._compare_descriptive_stats(strategies_results)
            return
        
        results_summary = []
        
        for strategy_name, cleaned_data in strategies_results.items():
            if cleaned_data is None or len(cleaned_data) == 0:
                continue
                
            try:
                # Préparer les données pour la modélisation
                X = cleaned_data.drop(columns=[self.target_col])
                y = cleaned_data[self.target_col]
                
                # Encoder les variables catégorielles si nécessaire
                X_encoded = X.copy()
                for col in X_encoded.select_dtypes(include=['object']).columns:
                    le = LabelEncoder()
                    X_encoded[col] = le.fit_transform(X_encoded[col])
                
                # Encoder la variable cible si catégorielle
                if y.dtype == 'object':
                    le_target = LabelEncoder()
                    y_encoded = le_target.fit_transform(y)
                else:
                    y_encoded = y
                
                # Diviser les données
                X_train, X_test, y_train, y_test = train_test_split(
                    X_encoded, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
                )
                
                # Entraîner un modèle simple
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                
                # Prédictions et évaluation
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                
                results_summary.append({
                    'Stratégie': strategy_name,
                    'Taille_Dataset': len(cleaned_data),
                    'Nb_Features': len(X.columns),
                    'Accuracy': accuracy,
                    'Données_Conservées_%': (len(cleaned_data) / len(self.original_data)) * 100
                })
                
            except Exception as e:
                print(f"Erreur lors de l'évaluation de la stratégie {strategy_name}: {e}")
                results_summary.append({
                    'Stratégie': strategy_name,
                    'Taille_Dataset': len(cleaned_data) if cleaned_data is not None else 0,
                    'Nb_Features': len(cleaned_data.columns) - 1 if cleaned_data is not None else 0,
                    'Accuracy': np.nan,
                    'Données_Conservées_%': (len(cleaned_data) / len(self.original_data)) * 100 if cleaned_data is not None else 0
                })
        
        # Afficher les résultats
        if results_summary:
            results_df = pd.DataFrame(results_summary)
            print(results_df.round(4))
            
            # Visualisation
            self._plot_strategy_comparison(results_df)
        
        return results_summary
    
    def _compare_descriptive_stats(self, strategies_results):
        """
        Compare les statistiques descriptives entre les stratégies
        """
        print("\nComparaison des statistiques descriptives:")
        print("-" * 50)
        
        for strategy_name, cleaned_data in strategies_results.items():
            if cleaned_data is not None:
                print(f"\n{strategy_name}:")
                print(f"  Forme: {cleaned_data.shape}")
                print(f"  Valeurs manquantes: {cleaned_data.isnull().sum().sum()}")
                print(f"  Données conservées: {(len(cleaned_data)/len(self.original_data))*100:.1f}%")
    
    def _plot_strategy_comparison(self, results_df):
        """
        Visualise la comparaison des stratégies
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Accuracy comparison
        if not results_df['Accuracy'].isna().all():
            results_df.plot(x='Stratégie', y='Accuracy', kind='bar', ax=axes[0])
            axes[0].set_title('Accuracy par Stratégie')
            axes[0].set_ylabel('Accuracy')
            axes[0].tick_params(axis='x', rotation=45)
        
        # Dataset size comparison
        results_df.plot(x='Stratégie', y='Taille_Dataset', kind='bar', ax=axes[1], color='orange')
        axes[1].set_title('Taille du Dataset par Stratégie')
        axes[1].set_ylabel('Nombre d\'observations')
        axes[1].tick_params(axis='x', rotation=45)
        
        # Data retention comparison
        results_df.plot(x='Stratégie', y='Données_Conservées_%', kind='bar', ax=axes[2], color='green')
        axes[2].set_title('Pourcentage de Données Conservées')
        axes[2].set_ylabel('Pourcentage (%)')
        axes[2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('strategies_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

def create_dataset_with_missing_values():
    """
    Crée un dataset avec des valeurs manquantes pour la démonstration
    """
    print("Création d'un dataset avec valeurs manquantes pour la démonstration...")
    
    # Créer un dataset de base
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'age': np.random.randint(18, 80, n_samples),
        'income': np.random.normal(50000, 15000, n_samples),
        'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples),
        'experience': np.random.randint(0, 40, n_samples),
        'city': np.random.choice(['Paris', 'Lyon', 'Marseille', 'Toulouse', 'Nice'], n_samples),
        'satisfaction': np.random.randint(1, 6, n_samples),
        'target': np.random.choice(['A', 'B', 'C'], n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Introduire des valeurs manquantes de manière réaliste
    # 1. Valeurs manquantes complètement aléatoires (MCAR)
    missing_indices = np.random.choice(df.index, size=int(0.05 * len(df)), replace=False)
    df.loc[missing_indices, 'age'] = np.nan
    
    # 2. Valeurs manquantes aléatoires (MAR) - revenus manquants pour les jeunes
    young_indices = df[df['age'] < 25].index
    missing_income = np.random.choice(young_indices, size=int(0.3 * len(young_indices)), replace=False)
    df.loc[missing_income, 'income'] = np.nan
    
    # 3. Valeurs manquantes non aléatoires (MNAR) - éducation manquante pour satisfaction faible
    low_satisfaction = df[df['satisfaction'] <= 2].index
    missing_education = np.random.choice(low_satisfaction, size=int(0.4 * len(low_satisfaction)), replace=False)
    df.loc[missing_education, 'education'] = np.nan
    
    # 4. Quelques valeurs manquantes supplémentaires
    df.loc[np.random.choice(df.index, 50, replace=False), 'experience'] = np.nan
    df.loc[np.random.choice(df.index, 30, replace=False), 'city'] = np.nan
    
    print(f"Dataset créé avec {len(df)} observations et {len(df.columns)} colonnes")
    print(f"Valeurs manquantes introduites: {df.isnull().sum().sum()}")
    
    return df

def demonstrate_missing_values_strategies():
    """
    Démontre toutes les stratégies de traitement des valeurs manquantes
    """
    print("DÉMONSTRATION DES STRATÉGIES DE TRAITEMENT DES VALEURS MANQUANTES")
    print("="*80)
    
    # Créer ou charger un dataset avec des valeurs manquantes
    df = create_dataset_with_missing_values()
    
    # Initialiser l'analyseur
    analyzer = MissingValueAnalyzer(df, target_col='target')
    
    # 1. Analyser les valeurs manquantes
    missing_analysis = analyzer.analyze_missing_values()
    
    # 2. Visualiser les patterns
    analyzer.visualize_missing_patterns()
    
    # 3. Tester différentes stratégies
    strategies_results = {}
    
    # Stratégie 1: Suppression des lignes
    try:
        strategies_results['Suppression_Lignes'] = analyzer.strategy_drop_rows()
    except Exception as e:
        print(f"Erreur avec suppression des lignes: {e}")
        strategies_results['Suppression_Lignes'] = None
    
    # Stratégie 2: Suppression des colonnes
    try:
        strategies_results['Suppression_Colonnes'] = analyzer.strategy_drop_columns(threshold=0.3)
    except Exception as e:
        print(f"Erreur avec suppression des colonnes: {e}")
        strategies_results['Suppression_Colonnes'] = None
    
    # Stratégie 3: Imputation simple (moyenne)
    try:
        strategies_results['Imputation_Moyenne'] = analyzer.strategy_simple_imputation('mean')
    except Exception as e:
        print(f"Erreur avec imputation moyenne: {e}")
        strategies_results['Imputation_Moyenne'] = None
    
    # Stratégie 4: Imputation simple (médiane)
    try:
        strategies_results['Imputation_Médiane'] = analyzer.strategy_simple_imputation('median')
    except Exception as e:
        print(f"Erreur avec imputation médiane: {e}")
        strategies_results['Imputation_Médiane'] = None
    
    # Stratégie 5: Imputation KNN
    try:
        strategies_results['Imputation_KNN'] = analyzer.strategy_knn_imputation(n_neighbors=5)
    except Exception as e:
        print(f"Erreur avec imputation KNN: {e}")
        strategies_results['Imputation_KNN'] = None
    
    # Stratégie 6: Imputation itérative
    try:
        strategies_results['Imputation_Itérative'] = analyzer.strategy_iterative_imputation(max_iter=10)
    except Exception as e:
        print(f"Erreur avec imputation itérative: {e}")
        strategies_results['Imputation_Itérative'] = None
    
    # 4. Comparer l'impact des stratégies
    comparison_results = analyzer.compare_strategies_impact(strategies_results)
    
    # 5. Recommandations
    print("\n" + "="*70)
    print("RECOMMANDATIONS")
    print("="*70)
    print("Choix de la stratégie en fonction du contexte:")
    print("• Suppression des lignes: Quand les données manquantes sont <5% et MCAR")
    print("• Suppression des colonnes: Quand >50% de valeurs manquantes dans une colonne")
    print("• Imputation moyenne/médiane: Simple et rapide pour données numériques")
    print("• Imputation KNN: Préserve les relations locales entre observations")
    print("• Imputation itérative: Plus sophistiquée, modélise les relations entre variables")
    print("\nConsidérations importantes:")
    print("• Type de mécanisme de manquance (MCAR, MAR, MNAR)")
    print("• Taille du dataset et proportion de valeurs manquantes")
    print("• Nature des variables (numériques vs catégorielles)")
    print("• Objectif de l'analyse (exploration vs prédiction)")
    
    return analyzer, strategies_results, comparison_results

if __name__ == "__main__":
    # Exécuter la démonstration complète
    analyzer, results, comparison = demonstrate_missing_values_strategies()
