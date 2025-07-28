import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
import seaborn as sns

print("ANALYSE K-MEANS DU DATASET IRIS")
print("=" * 50)

# 1. Chargement du dataset Iris
print("\n1. Chargement du dataset Iris...")

iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

print(f"Données chargées: {X.shape[0]} échantillons, {X.shape[1]} caractéristiques")
print(f"Caractéristiques: {feature_names}")
print(f"Espèces: {target_names}")

# Création d'un DataFrame pour faciliter la manipulation
df = pd.DataFrame(X, columns=feature_names)
df['species'] = [target_names[i] for i in y]

print("\nPremiers échantillons:")
print(df.head())

# 2. Séparation en ensemble d'entraînement et de test
print("\n2. Séparation des données...")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"Ensemble d'entraînement: {X_train.shape[0]} échantillons")
print(f"Ensemble de test: {X_test.shape[0]} échantillons")

# 3. Standardisation des données
print("\n3. Standardisation des données...")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Variables standardisées (moyenne = 0, écart-type = 1)")

# 4. Application de l'algorithme K-Means
print("\n4. Application de K-Means avec 3 clusters...")

# Utilisation de 3 clusters car nous avons 3 espèces d'iris
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X_train_scaled)

# Prédiction sur l'ensemble de test
y_pred_test = kmeans.predict(X_test_scaled)
y_pred_train = kmeans.predict(X_train_scaled)

print("Modèle K-Means entraîné avec succès")
print(f"Centres des clusters:")
for i, center in enumerate(kmeans.cluster_centers_):
    print(f"Cluster {i}: {center}")

# 5. Attribution des étiquettes de classe aux clusters
print("\n5. Attribution des étiquettes aux clusters...")

# Analyser la correspondance entre clusters et vraies classes sur l'ensemble d'entraînement
cluster_to_species = {}
for cluster in range(3):
    # Trouver les indices des échantillons dans ce cluster
    cluster_indices = np.where(y_pred_train == cluster)[0]
    # Trouver les vraies espèces correspondantes
    true_species = y_train[cluster_indices]
    # Attribuer l'espèce la plus fréquente à ce cluster
    most_common_species = np.bincount(true_species).argmax()
    cluster_to_species[cluster] = most_common_species
    
    print(f"Cluster {cluster} -> Espèce {target_names[most_common_species]}")
    
    # Statistiques détaillées pour ce cluster
    unique, counts = np.unique(true_species, return_counts=True)
    for species_idx, count in zip(unique, counts):
        percentage = count / len(cluster_indices) * 100
        print(f"  - {target_names[species_idx]}: {count} échantillons ({percentage:.1f}%)")

# Convertir les prédictions de clusters en prédictions d'espèces
y_pred_species_test = np.array([cluster_to_species[cluster] for cluster in y_pred_test])
y_pred_species_train = np.array([cluster_to_species[cluster] for cluster in y_pred_train])

# 6. Évaluation de la performance
print("\n6. Évaluation de la performance...")

# Métriques sur l'ensemble de test
accuracy_test = np.mean(y_pred_species_test == y_test)
ari_test = adjusted_rand_score(y_test, y_pred_test)
nmi_test = normalized_mutual_info_score(y_test, y_pred_test)
silhouette_test = silhouette_score(X_test_scaled, y_pred_test)

print(f"Performance sur l'ensemble de test:")
print(f"  - Précision: {accuracy_test:.3f}")
print(f"  - Adjusted Rand Index: {ari_test:.3f}")
print(f"  - Normalized Mutual Information: {nmi_test:.3f}")
print(f"  - Silhouette Score: {silhouette_test:.3f}")

# Métriques sur l'ensemble d'entraînement
accuracy_train = np.mean(y_pred_species_train == y_train)
ari_train = adjusted_rand_score(y_train, y_pred_train)
silhouette_train = silhouette_score(X_train_scaled, y_pred_train)

print(f"\nPerformance sur l'ensemble d'entraînement:")
print(f"  - Précision: {accuracy_train:.3f}")
print(f"  - Adjusted Rand Index: {ari_train:.3f}")
print(f"  - Silhouette Score: {silhouette_train:.3f}")

# 7. Visualisation des résultats
print("\n7. Génération des visualisations...")

# Configuration de la figure
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Analyse K-Means du Dataset Iris', fontsize=16, fontweight='bold')

# Couleurs pour les visualisations
colors_true = ['red', 'blue', 'green']
colors_pred = ['orange', 'purple', 'cyan']

# 1. Distribution des vraies classes
ax1 = axes[0, 0]
unique_train, counts_train = np.unique(y_train, return_counts=True)
ax1.bar([target_names[i] for i in unique_train], counts_train, color=colors_true)
ax1.set_title('Distribution des vraies classes\n(Ensemble d\'entraînement)')
ax1.set_ylabel('Nombre d\'échantillons')

# 2. Distribution des clusters prédits
ax2 = axes[0, 1]
unique_pred, counts_pred = np.unique(y_pred_train, return_counts=True)
ax2.bar([f'Cluster {i}' for i in unique_pred], counts_pred, color=colors_pred)
ax2.set_title('Distribution des clusters prédits\n(Ensemble d\'entraînement)')
ax2.set_ylabel('Nombre d\'échantillons')

# 3. Matrice de confusion
ax3 = axes[0, 2]
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred_species_test)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=target_names, yticklabels=target_names, ax=ax3)
ax3.set_title('Matrice de confusion\n(Ensemble de test)')
ax3.set_xlabel('Prédictions')
ax3.set_ylabel('Vraies classes')

# 4. Visualisation 2D des vraies classes (premières 2 caractéristiques)
ax4 = axes[1, 0]
for i, species in enumerate(target_names):
    mask = y_test == i
    ax4.scatter(X_test[mask, 0], X_test[mask, 1], 
               c=colors_true[i], label=species, alpha=0.7)
ax4.set_xlabel(feature_names[0])
ax4.set_ylabel(feature_names[1])
ax4.set_title('Vraies classes\n(2 premières caractéristiques)')
ax4.legend()

# 5. Visualisation 2D des clusters prédits
ax5 = axes[1, 1]
for i in range(3):
    mask = y_pred_test == i
    ax5.scatter(X_test[mask, 0], X_test[mask, 1], 
               c=colors_pred[i], label=f'Cluster {i}', alpha=0.7)
# Ajouter les centres des clusters
centers_original = scaler.inverse_transform(kmeans.cluster_centers_)
ax5.scatter(centers_original[:, 0], centers_original[:, 1], 
           c='black', marker='x', s=200, linewidths=3, label='Centres')
ax5.set_xlabel(feature_names[0])
ax5.set_ylabel(feature_names[1])
ax5.set_title('Clusters prédits\n(2 premières caractéristiques)')
ax5.legend()

# 6. Visualisation 2D avec les 2 dernières caractéristiques
ax6 = axes[1, 2]
for i, species in enumerate(target_names):
    mask = y_test == i
    ax6.scatter(X_test[mask, 2], X_test[mask, 3], 
               c=colors_true[i], label=species, alpha=0.7, marker='o')
for i in range(3):
    mask = y_pred_test == i
    ax6.scatter(X_test[mask, 2], X_test[mask, 3], 
               c=colors_pred[i], alpha=0.3, marker='s', s=100)
# Centres des clusters
ax6.scatter(centers_original[:, 2], centers_original[:, 3], 
           c='black', marker='x', s=200, linewidths=3)
ax6.set_xlabel(feature_names[2])
ax6.set_ylabel(feature_names[3])
ax6.set_title('Comparaison classes/clusters\n(2 dernières caractéristiques)')

plt.tight_layout()
plt.savefig('iris_kmeans_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# 8. Analyse détaillée des erreurs
print("\n8. Analyse détaillée des erreurs...")

# Identifier les échantillons mal classés
misclassified = np.where(y_pred_species_test != y_test)[0]
print(f"Nombre d'échantillons mal classés: {len(misclassified)} sur {len(y_test)}")

if len(misclassified) > 0:
    print("\nDétails des erreurs de classification:")
    for idx in misclassified:
        true_species = target_names[y_test[idx]]
        pred_species = target_names[y_pred_species_test[idx]]
        cluster = y_pred_test[idx]
        print(f"  Échantillon {idx}: Vraie={true_species}, Prédite={pred_species}, Cluster={cluster}")

# 9. Résumé final
print("\n" + "="*50)
print("RÉSUMÉ DE L'ANALYSE K-MEANS")
print("="*50)
print(f"Dataset: Iris ({X.shape[0]} échantillons, {X.shape[1]} caractéristiques)")
print(f"Algorithme: K-Means avec {kmeans.n_clusters} clusters")
print(f"Précision sur l'ensemble de test: {accuracy_test:.1%}")
print(f"Silhouette Score: {silhouette_test:.3f}")
print("\nConclusion:")
if accuracy_test > 0.8:
    print("L'algorithme K-Means a bien réussi à identifier les groupes naturels dans les données Iris.")
elif accuracy_test > 0.6:
    print("L'algorithme K-Means a partiellement réussi à identifier les groupes dans les données.")
else:
    print("L'algorithme K-Means a eu des difficultés à identifier correctement les groupes.")

print("\nLes résultats montrent que K-Means peut effectivement découvrir")
print("les structures naturelles dans les données, même sans supervision.")
