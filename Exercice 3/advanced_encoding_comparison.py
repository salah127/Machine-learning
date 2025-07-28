"""
Exemple pratique : Comparaison des encodages sur le dataset Adult (Income)
=========================================================================

Ce script utilise le dataset Adult pour démontrer l'impact des différentes
techniques d'encodage des variables catégorielles sur la prédiction de revenus.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

def create_adult_like_dataset():
    """
    Crée un dataset similaire au dataset Adult pour la démonstration
    """
    print("Création d'un dataset similaire au dataset Adult...")
    
    np.random.seed(42)
    n_samples = 5000
    
    # Variables catégorielles réalistes
    data = {
        # Variables ordinales
        'education': np.random.choice(['Some-college', 'HS-grad', 'Bachelors', 'Masters', 'Doctorate'], 
                                    n_samples, p=[0.3, 0.4, 0.2, 0.08, 0.02]),
        
        # Variables nominales
        'workclass': np.random.choice(['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov'], 
                                    n_samples, p=[0.7, 0.1, 0.05, 0.05, 0.05, 0.05]),
        
        'occupation': np.random.choice(['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 
                                      'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical'], 
                                     n_samples),
        
        'relationship': np.random.choice(['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'], 
                                       n_samples, p=[0.2, 0.15, 0.4, 0.15, 0.05, 0.05]),
        
        'race': np.random.choice(['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black'], 
                               n_samples, p=[0.8, 0.05, 0.02, 0.03, 0.1]),
        
        'sex': np.random.choice(['Male', 'Female'], n_samples, p=[0.6, 0.4]),
        
        'native_country': np.random.choice(['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany', 'India', 'Other'], 
                                         n_samples, p=[0.9, 0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.03]),
        
        # Variables numériques
        'age': np.random.randint(17, 80, n_samples),
        'hours_per_week': np.random.randint(1, 99, n_samples),
        'capital_gain': np.random.exponential(100, n_samples),
        'capital_loss': np.random.exponential(50, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Créer une variable cible réaliste basée sur les autres variables
    income_proba = np.zeros(n_samples)
    
    # Influence de l'éducation
    education_impact = {'Some-college': 0.1, 'HS-grad': 0.0, 'Bachelors': 0.3, 'Masters': 0.5, 'Doctorate': 0.7}
    income_proba += df['education'].map(education_impact)
    
    # Influence du sexe
    income_proba += np.where(df['sex'] == 'Male', 0.2, 0.0)
    
    # Influence de l'âge
    income_proba += (df['age'] - 17) / (80 - 17) * 0.3
    
    # Influence des heures de travail
    income_proba += (df['hours_per_week'] - 1) / (99 - 1) * 0.2
    
    # Influence du capital gain
    income_proba += np.log1p(df['capital_gain']) / np.log1p(df['capital_gain'].max()) * 0.4
    
    # Ajouter du bruit
    income_proba += np.random.normal(0, 0.2, n_samples)
    
    # Créer la variable cible binaire
    df['income'] = (income_proba > np.percentile(income_proba, 75)).astype(int)
    
    print(f"Dataset créé avec {n_samples} observations")
    print(f"Distribution de la variable cible: {df['income'].value_counts().to_dict()}")
    
    return df

def comprehensive_encoding_comparison(df, target_col='income'):
    """
    Compare exhaustivement toutes les méthodes d'encodage
    """
    print("" + "="*80)
    print("COMPARAISON EXHAUSTIVE DES MÉTHODES D'ENCODAGE")
    print("="*80)
    
    # Identifier les variables catégorielles
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    print(f"Variables catégorielles à encoder: {categorical_cols}")
    
    # Préparer les différents encodages
    encoded_datasets = {}
    
    # 1. Label Encoding
    print("🔧 Application du Label Encoding...")
    df_label = df.copy()
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df_label[col] = le.fit_transform(df_label[col])
        label_encoders[col] = le
    encoded_datasets['Label_Encoding'] = df_label
    
    # 2. One-Hot Encoding
    print("🔧 Application du One-Hot Encoding...")
    df_onehot = pd.get_dummies(df, columns=categorical_cols, drop_first=True, prefix=categorical_cols)
    encoded_datasets['OneHot_Encoding'] = df_onehot
    
    # 3. Ordinal Encoding (avec ordre personnalisé pour education)
    print("🔧 Application de l'Ordinal Encoding...")
    df_ordinal = df.copy()
    
    # Ordre personnalisé pour l'éducation
    education_order = ['HS-grad', 'Some-college', 'Bachelors', 'Masters', 'Doctorate']
    
    for col in categorical_cols:
        if col == 'education':
            # Ordre personnalisé pour l'éducation
            cat_order = [education_order]
        else:
            # Ordre par fréquence pour les autres
            cat_order = [df[col].value_counts().index.tolist()]
        
        oe = OrdinalEncoder(categories=cat_order, handle_unknown='use_encoded_value', unknown_value=-1)
        df_ordinal[col] = oe.fit_transform(df_ordinal[[col]]).flatten()
    
    encoded_datasets['Ordinal_Encoding'] = df_ordinal
    
    # 4. Frequency Encoding
    print("🔧 Application du Frequency Encoding...")
    df_freq = df.copy()
    for col in categorical_cols:
        freq_map = df[col].value_counts().to_dict()
        df_freq[col] = df_freq[col].map(freq_map)
    encoded_datasets['Frequency_Encoding'] = df_freq
    
    # 5. Binary Encoding (manuel pour les variables à haute cardinalité)
    print("🔧 Application du Binary Encoding (simplifié)...")
    df_binary = df.copy()
    
    for col in categorical_cols:
        if df[col].nunique() > 5:  # Appliquer seulement aux variables à haute cardinalité
            # Créer un mapping binaire simple
            unique_values = df[col].unique()
            n_bits = int(np.ceil(np.log2(len(unique_values))))
            
            for i in range(n_bits):
                df_binary[f'{col}_bit_{i}'] = 0
            
            for idx, value in enumerate(unique_values):
                binary_rep = format(idx, f'0{n_bits}b')
                mask = df_binary[col] == value
                for i, bit in enumerate(binary_rep):
                    df_binary.loc[mask, f'{col}_bit_{i}'] = int(bit)
            
            df_binary = df_binary.drop(columns=[col])
        else:
            # Utiliser Label Encoding pour les variables à faible cardinalité
            le = LabelEncoder()
            df_binary[col] = le.fit_transform(df_binary[col])
    
    encoded_datasets['Binary_Encoding'] = df_binary
    
    return encoded_datasets

def evaluate_encodings_performance(encoded_datasets, target_col='income'):
    """
    Évalue les performances de chaque méthode d'encodage
    """
    print("" + "="*80)
    print("ÉVALUATION DES PERFORMANCES")
    print("="*80)
    
    # Modèles à tester
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42, n_estimators=50)
    }
    
    results = []
    detailed_results = {}
    
    for encoding_name, data in encoded_datasets.items():
        print(f"🔍 Évaluation de {encoding_name}:")
        print(f"   Nombre de features: {data.shape[1] - 1}")
        
        # Préparer les données
        X = data.drop(columns=[target_col])
        y = data[target_col]
        
        # Gérer les valeurs infinies ou NaN
        X = X.replace([np.inf, -np.inf], np.nan)
        if X.isnull().sum().sum() > 0:
            X = X.fillna(X.mean())
        
        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        encoding_results = {}
        
        for model_name, model in models.items():
            try:
                # Entraînement
                model.fit(X_train, y_train)
                
                # Évaluation
                train_score = model.score(X_train, y_train)
                test_score = model.score(X_test, y_test)
                
                # Validation croisée
                cv_scores = cross_val_score(model, X_train, y_train, cv=5)
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()
                
                encoding_results[model_name] = {
                    'train_score': train_score,
                    'test_score': test_score,
                    'cv_mean': cv_mean,
                    'cv_std': cv_std
                }
                
                results.append({
                    'Encoding': encoding_name,
                    'Model': model_name,
                    'Train_Accuracy': train_score,
                    'Test_Accuracy': test_score,
                    'CV_Mean': cv_mean,
                    'CV_Std': cv_std,
                    'Features_Count': X.shape[1],
                    'Overfitting': train_score - test_score
                })
                
                print(f"   {model_name}:")
                print(f"     Train: {train_score:.4f}, Test: {test_score:.4f}")
                print(f"     CV: {cv_mean:.4f} ± {cv_std:.4f}")
                
            except Exception as e:
                print(f"   ❌ {model_name}: Erreur - {e}")
        
        detailed_results[encoding_name] = encoding_results
    
    return pd.DataFrame(results), detailed_results

def visualize_encoding_analysis(results_df):
    """
    Visualise l'analyse comparative des encodages
    """
    print("" + "="*80)
    print("VISUALISATION DES RÉSULTATS")
    print("="*80)
    
    # Créer des visualisations complètes
    fig, axes = plt.subplots(3, 2, figsize=(20, 18))
    fig.suptitle('Analyse Comparative des Méthodes d\'Encodage', fontsize=16)
    
    # 1. Heatmap des performances de test
    pivot_test = results_df.pivot(index='Encoding', columns='Model', values='Test_Accuracy')
    sns.heatmap(pivot_test, annot=True, fmt='.3f', cmap='viridis', ax=axes[0, 0])
    axes[0, 0].set_title('Accuracy de Test par Encodage et Modèle')
    
    # 2. Comparaison Train vs Test (Overfitting)
    pivot_overfit = results_df.pivot(index='Encoding', columns='Model', values='Overfitting')
    sns.heatmap(pivot_overfit, annot=True, fmt='.3f', cmap='Reds', ax=axes[0, 1])
    axes[0, 1].set_title('Overfitting (Train - Test) par Encodage et Modèle')
    
    # 3. Box plot des performances par encodage
    sns.boxplot(data=results_df, x='Encoding', y='Test_Accuracy', ax=axes[1, 0])
    axes[1, 0].set_title('Distribution des Performances par Encodage')   
    axes[1, 0].tick_params(axis='x', rotation=45)
    # 4. Nombre de features par encodage
    features_by_encoding = results_df.groupby('Encoding')['Features_Count'].first()
    features_by_encoding.plot(kind='bar', ax=axes[1, 1], color='orange')    
    axes[1, 1].set_title('Nombre de Features par Encodage')    
    axes[1, 1].tick_params(axis='x', rotation=45)    
    axes[1, 1].set_ylabel('Nombre de Features')        
    # 5. Performance vs Complexité    
    avg_perf = results_df.groupby('Encoding').agg({        'Test_Accuracy': 'mean',        'Features_Count': 'first',        'CV_Mean': 'mean'    }).reset_index()        
    scatter = axes[2, 0].scatter(avg_perf['Features_Count'], avg_perf['Test_Accuracy'],                                
                                 s=150, alpha=0.7, c=avg_perf['CV_Mean'], cmap='viridis')    
    for i, row in avg_perf.iterrows():        
        axes[2, 0].annotate(row['Encoding'],                           
                            (row['Features_Count'], row['Test_Accuracy']),                          
                            xytext=(5, 5), textcoords='offset points', fontsize=9)    
        axes[2, 0].set_xlabel('Nombre de Features')    
        axes[2, 0].set_ylabel('Accuracy Moyenne')    
        axes[2, 0].set_title('Performance vs Complexité')    
        plt.colorbar(scatter, ax=axes[2, 0], label='CV Mean')        
        # 6. Variance des performances (stabilité)    
        cv_std_by_encoding = results_df.groupby('Encoding')['CV_Std'].mean()    
        cv_std_by_encoding.plot(kind='bar', ax=axes[2, 1], color='lightcoral')    
        axes[2, 1].set_title('Stabilité des Performances (CV Std)')    
        axes[2, 1].tick_params(axis='x', rotation=45)    
        axes[2, 1].set_ylabel('Écart-type CV')
        
        plt.tight_layout()
        plt.savefig('comprehensive_encoding_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

def generate_final_recommendations(results_df):
    """
    Génère des recommandations finales basées sur l'analyse
    """
    print("\n" + "="*80)
    print("RECOMMANDATIONS FINALES")
    print("="*80)
    
    # Analyser les résultats
    avg_performance = results_df.groupby('Encoding').agg({
    'Test_Accuracy': ['mean', 'std'],
    'CV_Mean': 'mean',
    'CV_Std': 'mean',
    'Features_Count': 'first',
    'Overfitting': 'mean'
    }).round(4)
    
    avg_performance.columns = ['Test_Mean', 'Test_Std', 'CV_Mean', 'CV_Std', 'Features', 'Overfitting']
    
    print("📊 RÉSUMÉ DES PERFORMANCES:")
    print(avg_performance)
    
    # Identifier les meilleures méthodes
    best_accuracy = avg_performance['Test_Mean'].idxmax()
    most_stable = avg_performance['CV_Std'].idxmin()
    least_overfitting = avg_performance['Overfitting'].idxmin()
    most_efficient = avg_performance.loc[avg_performance['Test_Mean'] > avg_performance['Test_Mean'].quantile(0.7), 'Features'].idxmin()
    
    print(f"\n🏆 LAURÉATS PAR CATÉGORIE:")
    print(f"• Meilleure performance: {best_accuracy} ({avg_performance.loc[best_accuracy, 'Test_Mean']:.4f})")
    print(f"• Plus stable: {most_stable} (CV std: {avg_performance.loc[most_stable, 'CV_Std']:.4f})")
    print(f"• Moins d'overfitting: {least_overfitting} ({avg_performance.loc[least_overfitting, 'Overfitting']:.4f})")
    print(f"• Plus efficace: {most_efficient} ({avg_performance.loc[most_efficient, 'Features']:.0f} features)")
    
    print(f"\n💡 RECOMMANDATIONS SPÉCIFIQUES:")
    print("-" * 50)
    
    print(f"\n🎯 POUR LA PRODUCTION:")
    if avg_performance.loc[best_accuracy, 'Overfitting'] < 0.05 and avg_performance.loc[best_accuracy, 'CV_Std'] < 0.02:
            print(f"   Recommandé: {best_accuracy} (meilleure performance + stabilité)")
    else:
        print(f"   Recommandé: {most_stable} (privilégier la stabilité)")
    
    print(f"\n🚀 POUR LE PROTOTYPAGE RAPIDE:")
    simple_methods = ['Label_Encoding', 'Frequency_Encoding']
    simple_available = [m for m in simple_methods if m in avg_performance.index]
    if simple_available:
        best_simple = avg_performance.loc[simple_available, 'Test_Mean'].idxmax()
    print(f"   Recommandé: {best_simple} (simplicité + performance)")
    
    print(f"\n📈 POUR L'OPTIMISATION:")
    if 'OneHot_Encoding' in avg_performance.index:
        onehot_perf = avg_performance.loc['OneHot_Encoding', 'Test_Mean']
    if onehot_perf >= avg_performance['Test_Mean'].quantile(0.8):
        print(f"   OneHot_Encoding recommandé (interprétabilité + performance)")
    else:
        print(f"   {best_accuracy} recommandé (performance maximale)")
    
    print(f"\n⚠️  POINTS D'ATTENTION:")
    high_overfit = avg_performance[avg_performance['Overfitting'] > 0.1].index.tolist()
    if high_overfit:
        print(f"   Risque d'overfitting: {high_overfit}")
    
    high_variance = avg_performance[avg_performance['CV_Std'] > 0.02].index.tolist()
    if high_variance:
        print(f"   Performances instables: {high_variance}")
    
    many_features = avg_performance[avg_performance['Features'] > avg_performance['Features'].quantile(0.8)].index.tolist()
    if many_features:
        print(f"   Complexité élevée: {many_features}")

def main():
    """
    Fonction principale pour l'analyse complète
    """
    print("ANALYSE COMPLÈTE DES TECHNIQUES D'ENCODAGE CATÉGORIEL")
    print("="*80)
    
    # 1. Créer le dataset
    df = create_adult_like_dataset()
    
    # 2. Afficher les informations sur les variables catégorielles
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    print(f"\n📊 ANALYSE DES VARIABLES CATÉGORIELLES:")
    for col in categorical_cols:
        n_unique = df[col].nunique()
    most_frequent = df[col].mode().iloc[0]
    print(f"   {col}: {n_unique} catégories, plus fréquente: '{most_frequent}'")
    
    # 3. Appliquer tous les encodages
    encoded_datasets = comprehensive_encoding_comparison(df)
    
    # 4. Évaluer les performances
    results_df, detailed_results = evaluate_encodings_performance(encoded_datasets)
    
    # 5. Visualiser les résultats
    visualize_encoding_analysis(results_df)
    
    # 6. Générer les recommandations
    generate_final_recommendations(results_df)
    
    print("\n" + "="*80)
    print("✅ ANALYSE TERMINÉE!")
    print("📚 Vous avez maintenant une compréhension approfondie des techniques d'encodage.")
    print("="*80)
    
    return df, encoded_datasets, results_df

if __name__ == "__main__":
    df, encodings, results = main()
