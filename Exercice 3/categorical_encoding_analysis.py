"""
Exercice 3 : Encodage des Variables Catégorielles
================================================

Ce script explore différentes techniques d'encodage des variables catégorielles
et compare leur impact sur les performances des modèles de machine learning.

Techniques couvertes :
1. Label Encoding
2. One-Hot Encoding  
3. Ordinal Encoding
4. Target Encoding
5. Binary Encoding
6. Frequency Encoding

Comparaison sur plusieurs modèles :
- Random Forest
- Logistic Regression
- SVM
- Gradient Boosting
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import category_encoders as ce
import warnings
warnings.filterwarnings('ignore')

class CategoricalEncodingAnalyzer:
    """
    Classe pour analyser et comparer différentes techniques d'encodage des variables catégorielles
    """
    
    def __init__(self, data, target_column, categorical_columns=None):
        """
        Initialise l'analyseur
        
        Args:
            data (DataFrame): Dataset à analyser
            target_column (str): Nom de la colonne cible
            categorical_columns (list): Liste des colonnes catégorielles (auto-détection si None)
        """
        self.original_data = data.copy()
        self.data = data.copy()
        self.target_column = target_column
        
        if categorical_columns is None:
            self.categorical_columns = list(data.select_dtypes(include=['object']).columns)
            if target_column in self.categorical_columns:
                self.categorical_columns.remove(target_column)
        else:
            self.categorical_columns = categorical_columns
            
        self.numerical_columns = list(data.select_dtypes(include=[np.number]).columns)
        if target_column in self.numerical_columns:
            self.numerical_columns.remove(target_column)
            
        self.encoded_datasets = {}
        self.model_results = {}
        
        print(f"Dataset initialisé:")
        print(f"  - Forme: {data.shape}")
        print(f"  - Colonnes catégorielles: {self.categorical_columns}")
        print(f"  - Colonnes numériques: {self.numerical_columns}")
        print(f"  - Colonne cible: {target_column}")
    
    def analyze_categorical_variables(self):
        """
        Analyse les variables catégorielles
        """
        print("\n" + "="*70)
        print("ANALYSE DES VARIABLES CATÉGORIELLES")
        print("="*70)
        
        categorical_info = []
        
        for col in self.categorical_columns:
            unique_values = self.data[col].nunique()
            most_frequent = self.data[col].mode().iloc[0] if not self.data[col].mode().empty else 'N/A'
            most_frequent_count = self.data[col].value_counts().iloc[0] if not self.data[col].value_counts().empty else 0
            missing_count = self.data[col].isnull().sum()
            
            categorical_info.append({
                'Colonne': col,
                'Valeurs_Uniques': unique_values,
                'Valeur_Plus_Fréquente': most_frequent,
                'Fréquence_Max': most_frequent_count,
                'Valeurs_Manquantes': missing_count,
                'Cardinalité': 'Faible' if unique_values <= 5 else 'Moyenne' if unique_values <= 20 else 'Élevée'
            })
        
        categorical_df = pd.DataFrame(categorical_info)
        print(categorical_df)
        
        # Visualisation
        if len(self.categorical_columns) > 0:
            self._plot_categorical_analysis()
        
        return categorical_df
    
    def _plot_categorical_analysis(self):
        """
        Visualise l'analyse des variables catégorielles
        """
        n_cols = min(3, len(self.categorical_columns))
        n_rows = (len(self.categorical_columns) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes
        else:
            axes = axes.flatten()
        
        for i, col in enumerate(self.categorical_columns):
            if i < len(axes):
                value_counts = self.data[col].value_counts().head(10)
                value_counts.plot(kind='bar', ax=axes[i])
                axes[i].set_title(f'Distribution de {col}')
                axes[i].set_xlabel('Valeurs')
                axes[i].set_ylabel('Fréquence')
                axes[i].tick_params(axis='x', rotation=45)
        
        # Masquer les axes inutilisés
        for i in range(len(self.categorical_columns), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('categorical_variables_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def apply_label_encoding(self):
        """
        Applique le Label Encoding
        """
        print("\n" + "="*50)
        print("LABEL ENCODING")
        print("="*50)
        
        data_encoded = self.data.copy()
        label_encoders = {}
        
        for col in self.categorical_columns:
            le = LabelEncoder()
            # Gérer les valeurs manquantes
            mask = data_encoded[col].notna()
            if mask.sum() > 0:
                data_encoded.loc[mask, col] = le.fit_transform(data_encoded.loc[mask, col])
                label_encoders[col] = le
                
                print(f"✅ {col}: {len(le.classes_)} classes → [0, {len(le.classes_)-1}]")
                print(f"   Classes: {list(le.classes_)[:5]}{'...' if len(le.classes_) > 5 else ''}")
        
        self.encoded_datasets['Label_Encoding'] = data_encoded
        self.label_encoders = label_encoders
        
        print(f"\nForme du dataset après Label Encoding: {data_encoded.shape}")
        return data_encoded
    
    def apply_onehot_encoding(self, drop_first=True, max_categories=10):
        """
        Applique le One-Hot Encoding
        
        Args:
            drop_first (bool): Supprimer la première catégorie pour éviter la multicolinéarité
            max_categories (int): Nombre maximum de catégories à encoder (limitation pour éviter l'explosion dimensionnelle)
        """
        print("\n" + "="*50)
        print("ONE-HOT ENCODING")
        print("="*50)
        
        data_encoded = self.data.copy()
        
        # Filtrer les colonnes avec trop de catégories
        valid_columns = []
        for col in self.categorical_columns:
            n_categories = self.data[col].nunique()
            if n_categories <= max_categories:
                valid_columns.append(col)
                print(f"✅ {col}: {n_categories} catégories")
            else:
                print(f"⚠️  {col}: {n_categories} catégories (trop élevé, ignoré)")
        
        if valid_columns:
            # Appliquer One-Hot Encoding
            data_encoded = pd.get_dummies(data_encoded, 
                                        columns=valid_columns, 
                                        drop_first=drop_first,
                                        prefix=valid_columns)
            
            print(f"\nNombres de colonnes créées:")
            original_cols = len(self.data.columns)
            new_cols = len(data_encoded.columns)
            print(f"  Avant: {original_cols}")
            print(f"  Après: {new_cols}")
            print(f"  Différence: +{new_cols - original_cols}")
        
        self.encoded_datasets['OneHot_Encoding'] = data_encoded
        
        print(f"\nForme du dataset après One-Hot Encoding: {data_encoded.shape}")
        return data_encoded
    
    def apply_ordinal_encoding(self, ordinal_mappings=None):
        """
        Applique l'Ordinal Encoding
        
        Args:
            ordinal_mappings (dict): Mappings personnalisés pour l'ordre {colonne: [ordre]}
        """
        print("\n" + "="*50)
        print("ORDINAL ENCODING")
        print("="*50)
        
        data_encoded = self.data.copy()
        ordinal_encoders = {}
        
        for col in self.categorical_columns:
            if ordinal_mappings and col in ordinal_mappings:
                # Utiliser l'ordre personnalisé
                categories = [ordinal_mappings[col]]
                print(f"✅ {col}: Ordre personnalisé {ordinal_mappings[col]}")
            else:
                # Ordre automatique basé sur la fréquence
                categories = [self.data[col].value_counts().index.tolist()]
                print(f"✅ {col}: Ordre par fréquence {categories[0][:3]}{'...' if len(categories[0]) > 3 else ''}")
            
            oe = OrdinalEncoder(categories=categories, handle_unknown='use_encoded_value', unknown_value=-1)
            mask = data_encoded[col].notna()
            if mask.sum() > 0:
                data_encoded.loc[mask, col] = oe.fit_transform(data_encoded.loc[mask, col].values.reshape(-1, 1)).flatten()
                ordinal_encoders[col] = oe
        
        self.encoded_datasets['Ordinal_Encoding'] = data_encoded
        self.ordinal_encoders = ordinal_encoders
        
        print(f"\nForme du dataset après Ordinal Encoding: {data_encoded.shape}")
        return data_encoded
    
    def apply_target_encoding(self, cv_folds=5, smoothing=1.0):
        """
        Applique le Target Encoding
        
        Args:
            cv_folds (int): Nombre de folds pour la validation croisée
            smoothing (float): Paramètre de lissage pour éviter l'overfitting
        """
        print("\n" + "="*50)
        print("TARGET ENCODING")
        print("="*50)
        
        data_encoded = self.data.copy()
        
        try:
            # Utiliser category_encoders pour le Target Encoding
            te = ce.TargetEncoder(cols=self.categorical_columns, 
                                smoothing=smoothing,
                                cv=cv_folds)
            
            X = data_encoded.drop(columns=[self.target_column])
            y = data_encoded[self.target_column]
            
            # Encoder les variables catégorielles
            X_encoded = te.fit_transform(X, y)
            
            # Recombiner avec la target
            data_encoded = pd.concat([X_encoded, y], axis=1)
            
            print(f"✅ Target Encoding appliqué avec CV={cv_folds} et smoothing={smoothing}")
            for col in self.categorical_columns:
                if col in X_encoded.columns:
                    print(f"   {col}: Moyenne des encodages = {X_encoded[col].mean():.3f}")
            
            self.encoded_datasets['Target_Encoding'] = data_encoded
            self.target_encoder = te
            
        except Exception as e:
            print(f"❌ Erreur avec Target Encoding: {e}")
            print("Utilisation du Label Encoding comme fallback")
            data_encoded = self.apply_label_encoding()
            self.encoded_datasets['Target_Encoding'] = data_encoded
        
        print(f"\nForme du dataset après Target Encoding: {data_encoded.shape}")
        return data_encoded
    
    def apply_frequency_encoding(self):
        """
        Applique le Frequency Encoding
        """
        print("\n" + "="*50)
        print("FREQUENCY ENCODING")
        print("="*50)
        
        data_encoded = self.data.copy()
        
        for col in self.categorical_columns:
            # Calculer les fréquences
            freq_map = self.data[col].value_counts().to_dict()
            
            # Appliquer l'encodage
            data_encoded[col] = data_encoded[col].map(freq_map)
            
            print(f"✅ {col}: Encodé par fréquence")
            print(f"   Fréquence max: {max(freq_map.values())}")
            print(f"   Fréquence min: {min(freq_map.values())}")
        
        self.encoded_datasets['Frequency_Encoding'] = data_encoded
        
        print(f"\nForme du dataset après Frequency Encoding: {data_encoded.shape}")
        return data_encoded
    
    def compare_encoding_methods(self, test_size=0.2, random_state=42):
        """
        Compare toutes les méthodes d'encodage sur plusieurs modèles
        
        Args:
            test_size (float): Proportion du test set
            random_state (int): Seed pour la reproductibilité
        """
        print("\n" + "="*70)
        print("COMPARAISON DES MÉTHODES D'ENCODAGE")
        print("="*70)
        
        # Modèles à tester
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=random_state),
            'Logistic Regression': LogisticRegression(random_state=random_state, max_iter=1000),
            'SVM': SVC(random_state=random_state),
            'Gradient Boosting': GradientBoostingClassifier(random_state=random_state)
        }
        
        results = []
        
        for encoding_method, data_encoded in self.encoded_datasets.items():
            print(f"\n🔍 Test de {encoding_method}:")
            
            try:
                # Préparer les données
                X = data_encoded.drop(columns=[self.target_column])
                y = data_encoded[self.target_column]
                
                # Gérer les valeurs manquantes si nécessaire
                if X.isnull().sum().sum() > 0:
                    X = X.fillna(X.mean() if X.select_dtypes(include=[np.number]).shape[1] > 0 else 0)
                
                # Encoder la target si nécessaire
                if y.dtype == 'object':
                    le_target = LabelEncoder()
                    y = le_target.fit_transform(y)
                
                # Split train/test
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state, stratify=y
                )
                
                # Tester chaque modèle
                for model_name, model in models.items():
                    try:
                        # Entraînement
                        model.fit(X_train, y_train)
                        
                        # Prédiction
                        y_pred = model.predict(X_test)
                        accuracy = accuracy_score(y_test, y_pred)
                        
                        # Validation croisée
                        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
                        cv_mean = cv_scores.mean()
                        cv_std = cv_scores.std()
                        
                        results.append({
                            'Encoding': encoding_method,
                            'Model': model_name,
                            'Test_Accuracy': accuracy,
                            'CV_Mean': cv_mean,
                            'CV_Std': cv_std,
                            'Features_Count': X.shape[1],
                            'Train_Size': len(X_train)
                        })
                        
                        print(f"   {model_name}: {accuracy:.4f} (CV: {cv_mean:.4f}±{cv_std:.4f})")
                        
                    except Exception as e:
                        print(f"   ❌ {model_name}: Erreur - {e}")
                        
            except Exception as e:
                print(f"   ❌ Erreur générale: {e}")
        
        # Créer le DataFrame des résultats
        if results:
            results_df = pd.DataFrame(results)
            self.model_results = results_df
            
            # Afficher le résumé
            print(f"\n📊 RÉSUMÉ DES PERFORMANCES:")
            print("="*50)
            
            # Moyenne par encodage
            avg_by_encoding = results_df.groupby('Encoding')['Test_Accuracy'].agg(['mean', 'std']).round(4)
            print("\nPerformance moyenne par méthode d'encodage:")
            print(avg_by_encoding)
            
            # Meilleur résultat par modèle
            best_by_model = results_df.loc[results_df.groupby('Model')['Test_Accuracy'].idxmax()]
            print(f"\nMeilleurs résultats par modèle:")
            print(best_by_model[['Model', 'Encoding', 'Test_Accuracy']])
            
            # Visualisation
            self._plot_encoding_comparison(results_df)
            
            return results_df
        
        return None
    
    def _plot_encoding_comparison(self, results_df):
        """
        Visualise la comparaison des méthodes d'encodage
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Comparaison des Méthodes d\'Encodage', fontsize=16)
        
        # 1. Heatmap des performances par encodage et modèle
        pivot_table = results_df.pivot(index='Encoding', columns='Model', values='Test_Accuracy')
        sns.heatmap(pivot_table, annot=True, fmt='.3f', cmap='viridis', ax=axes[0, 0])
        axes[0, 0].set_title('Accuracy par Encodage et Modèle')
        
        # 2. Box plot des performances par encodage
        sns.boxplot(data=results_df, x='Encoding', y='Test_Accuracy', ax=axes[0, 1])
        axes[0, 1].set_title('Distribution des Performances par Encodage')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Nombre de features par encodage
        features_by_encoding = results_df.groupby('Encoding')['Features_Count'].first()
        features_by_encoding.plot(kind='bar', ax=axes[1, 0], color='orange')
        axes[1, 0].set_title('Nombre de Features par Encodage')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].set_ylabel('Nombre de Features')
        
        # 4. Performance vs Nombre de features
        avg_perf = results_df.groupby('Encoding').agg({
            'Test_Accuracy': 'mean',
            'Features_Count': 'first'
        }).reset_index()
        
        scatter = axes[1, 1].scatter(avg_perf['Features_Count'], avg_perf['Test_Accuracy'], 
                                   s=100, alpha=0.7, c=range(len(avg_perf)), cmap='viridis')
        for i, row in avg_perf.iterrows():
            axes[1, 1].annotate(row['Encoding'], 
                              (row['Features_Count'], row['Test_Accuracy']),
                              xytext=(5, 5), textcoords='offset points', fontsize=8)
        axes[1, 1].set_xlabel('Nombre de Features')
        axes[1, 1].set_ylabel('Accuracy Moyenne')
        axes[1, 1].set_title('Performance vs Complexité')
        
        plt.tight_layout()
        plt.savefig('encoding_methods_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_feature_importance(self, encoding_method='Random Forest'):
        """
        Analyse l'importance des features pour différents encodages
        """
        print("\n" + "="*70)
        print("ANALYSE DE L'IMPORTANCE DES FEATURES")
        print("="*70)
        
        feature_importance_results = {}
        
        for encoding_name, data_encoded in self.encoded_datasets.items():
            try:
                # Préparer les données
                X = data_encoded.drop(columns=[self.target_column])
                y = data_encoded[self.target_column]
                
                # Gérer les valeurs manquantes
                if X.isnull().sum().sum() > 0:
                    X = X.fillna(X.mean() if X.select_dtypes(include=[np.number]).shape[1] > 0 else 0)
                
                # Encoder la target si nécessaire
                if y.dtype == 'object':
                    le_target = LabelEncoder()
                    y = le_target.fit_transform(y)
                
                # Entraîner un Random Forest pour l'importance des features
                rf = RandomForestClassifier(n_estimators=100, random_state=42)
                rf.fit(X, y)
                
                # Récupérer l'importance
                feature_importance = pd.DataFrame({
                    'Feature': X.columns,
                    'Importance': rf.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                feature_importance_results[encoding_name] = feature_importance
                
                print(f"\n🔍 {encoding_name} - Top 5 features importantes:")
                print(feature_importance.head())
                
            except Exception as e:
                print(f"❌ Erreur avec {encoding_name}: {e}")
        
        # Visualisation de l'importance des features
        if feature_importance_results:
            self._plot_feature_importance(feature_importance_results)
        
        return feature_importance_results
    
    def _plot_feature_importance(self, feature_importance_results):
        """
        Visualise l'importance des features pour chaque encodage
        """
        n_encodings = len(feature_importance_results)
        n_cols = min(2, n_encodings)
        n_rows = (n_encodings + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes
        else:
            axes = axes.flatten()
        
        for i, (encoding_name, importance_df) in enumerate(feature_importance_results.items()):
            if i < len(axes):
                top_features = importance_df.head(10)
                top_features.plot(x='Feature', y='Importance', kind='barh', ax=axes[i])
                axes[i].set_title(f'Top Features - {encoding_name}')
                axes[i].set_xlabel('Importance')
        
        # Masquer les axes inutilisés
        for i in range(len(feature_importance_results), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('feature_importance_by_encoding.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_recommendations(self):
        """
        Génère des recommandations basées sur l'analyse
        """
        print("\n" + "="*70)
        print("RECOMMANDATIONS")
        print("="*70)
        
        if hasattr(self, 'model_results') and not self.model_results.empty:
            # Trouver la meilleure méthode d'encodage
            best_encoding = self.model_results.groupby('Encoding')['Test_Accuracy'].mean().idxmax()
            best_score = self.model_results.groupby('Encoding')['Test_Accuracy'].mean().max()
            
            print(f"🏆 MEILLEURE MÉTHODE D'ENCODAGE: {best_encoding}")
            print(f"📊 Score moyen: {best_score:.4f}")
            
            # Analyse par cardinalité
            print(f"\n📋 RECOMMANDATIONS PAR CONTEXTE:")
            print("-" * 40)
            
            for col in self.categorical_columns:
                n_categories = self.data[col].nunique()
                
                print(f"\n🔹 {col} ({n_categories} catégories):")
                
                if n_categories == 2:
                    print("   → Label Encoding (binaire)")
                elif n_categories <= 5:
                    print("   → One-Hot Encoding (faible cardinalité)")
                elif n_categories <= 20:
                    print("   → Target Encoding ou Ordinal Encoding")
                else:
                    print("   → Target Encoding ou Frequency Encoding (haute cardinalité)")
                
                # Vérifier s'il y a un ordre naturel
                unique_values = self.data[col].unique()
                if any(val in str(unique_values).lower() for val in ['low', 'medium', 'high', 'small', 'large']):
                    print("   ⚠️  Ordre naturel détecté → Considérer Ordinal Encoding")
        
        print(f"\n💡 RÈGLES GÉNÉRALES:")
        print("-" * 20)
        print("• Variables binaires → Label Encoding")
        print("• Variables ordinales → Ordinal Encoding")
        print("• Faible cardinalité (<10) → One-Hot Encoding")
        print("• Haute cardinalité (>20) → Target/Frequency Encoding")
        print("• Modèles linéaires → Éviter Label Encoding pour variables nominales")
        print("• Modèles basés sur les arbres → Plus flexibles avec Label Encoding")

def create_sample_dataset_with_categories():
    """
    Crée un dataset d'exemple avec différents types de variables catégorielles
    """
    print("Création d'un dataset d'exemple avec variables catégorielles...")
    
    np.random.seed(42)
    n_samples = 2000
    
    # Variables catégorielles de différents types
    data = {
        # Variable binaire
        'gender': np.random.choice(['Male', 'Female'], n_samples, p=[0.6, 0.4]),
        
        # Variable ordinale
        'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples, p=[0.4, 0.3, 0.2, 0.1]),
        
        # Variable nominale (faible cardinalité)
        'city': np.random.choice(['Paris', 'Lyon', 'Marseille', 'Toulouse', 'Nice'], n_samples),
        
        # Variable nominale (cardinalité moyenne)
        'department': np.random.choice([f'Dept_{i:02d}' for i in range(1, 16)], n_samples),
        
        # Variable avec ordre de qualité
        'satisfaction': np.random.choice(['Very Low', 'Low', 'Medium', 'High', 'Very High'], n_samples),
        
        # Variables numériques
        'age': np.random.randint(18, 65, n_samples),
        'salary': np.random.normal(50000, 15000, n_samples),
        'experience': np.random.randint(0, 30, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Créer une variable cible influencée par les autres variables
    target_proba = np.zeros(n_samples)
    
    # Influence du genre
    target_proba += np.where(df['gender'] == 'Female', 0.1, 0.0)
    
    # Influence de l'éducation (ordinale)
    education_map = {'High School': 0, 'Bachelor': 0.1, 'Master': 0.2, 'PhD': 0.3}
    target_proba += df['education'].map(education_map)
    
    # Influence de la satisfaction
    satisfaction_map = {'Very Low': -0.2, 'Low': -0.1, 'Medium': 0, 'High': 0.1, 'Very High': 0.2}
    target_proba += df['satisfaction'].map(satisfaction_map)
    
    # Influence de l'âge (normalisée)
    target_proba += (df['age'] - df['age'].min()) / (df['age'].max() - df['age'].min()) * 0.2
    
    # Influence du salaire (normalisée)
    target_proba += (df['salary'] - df['salary'].min()) / (df['salary'].max() - df['salary'].min()) * 0.3
    
    # Ajouter du bruit
    target_proba += np.random.normal(0, 0.1, n_samples)
    
    # Créer la variable cible binaire
    df['promotion'] = (target_proba > np.median(target_proba)).astype(int)
    
    print(f"Dataset créé:")
    print(f"  - {n_samples} observations")
    print(f"  - {len(df.columns)-1} features ({len(df.select_dtypes(include=['object']).columns)} catégorielles)")
    print(f"  - Target: 'promotion' ({df['promotion'].value_counts().to_dict()})")
    
    return df

def demonstrate_categorical_encoding():
    """
    Démonstration complète des techniques d'encodage catégoriel
    """
    print("DÉMONSTRATION DES TECHNIQUES D'ENCODAGE CATÉGORIEL")
    print("="*80)
    
    # Créer le dataset d'exemple
    df = create_sample_dataset_with_categories()
    
    # Initialiser l'analyseur
    analyzer = CategoricalEncodingAnalyzer(df, target_column='promotion')
    
    # 1. Analyser les variables catégorielles
    categorical_analysis = analyzer.analyze_categorical_variables()
    
    # 2. Appliquer différentes méthodes d'encodage
    print(f"\n🔧 APPLICATION DES MÉTHODES D'ENCODAGE:")
    print("="*50)
    
    # Label Encoding
    analyzer.apply_label_encoding()
    
    # One-Hot Encoding
    analyzer.apply_onehot_encoding(drop_first=True, max_categories=10)
    
    # Ordinal Encoding avec ordre personnalisé
    ordinal_mappings = {
        'education': ['High School', 'Bachelor', 'Master', 'PhD'],
        'satisfaction': ['Very Low', 'Low', 'Medium', 'High', 'Very High']
    }
    analyzer.apply_ordinal_encoding(ordinal_mappings)
    
    # Target Encoding
    analyzer.apply_target_encoding(cv_folds=5, smoothing=1.0)
    
    # Frequency Encoding
    analyzer.apply_frequency_encoding()
    
    # 3. Comparer les méthodes
    results = analyzer.compare_encoding_methods(test_size=0.2, random_state=42)
    
    # 4. Analyser l'importance des features
    feature_importance = analyzer.analyze_feature_importance()
    
    # 5. Générer des recommandations
    analyzer.generate_recommendations()
    
    print("\n" + "="*80)
    print("✅ DÉMONSTRATION TERMINÉE!")
    print("📊 Vous maîtrisez maintenant les principales techniques d'encodage catégoriel.")
    print("="*80)
    
    return analyzer, results, feature_importance

if __name__ == "__main__":
    # Exécuter la démonstration complète
    analyzer, results, importance = demonstrate_categorical_encoding()
