# 📘 Cheat Sheet : Jupyter Notebook & NumPy Einsum

Ce document centralise les outils indispensables pour le développement en Python (Data Science & IA) au sein de VS Code.

---

## I. Jupyter Notebook dans VS Code

Le Notebook est votre espace de travail interactif. Voici comment le maîtriser.

### Raccourcis clavier (Mode Commande - `Echap`)
*   **`A`** : Ajouter une cellule au-dessus de la sélection.
*   **`B`** : Ajouter une cellule en dessous de la sélection.
*   **`D, D`** : Supprimer la cellule sélectionnée.
*   **`Z`** : Annuler la suppression d'une cellule.
*   **`Y`** : Convertir la cellule en format **Code**.
*   **`M`** : Convertir la cellule en format **Markdown** (pour la documentation).
*   **`Maj + Entrée`** : Exécuter la cellule et passer à la suivante.

### Commandes Magiques (Utilitaires)
*   **`%time`** : Affiche le temps d'exécution d'une ligne de code.
*   **`%%time`** : Affiche le temps d'exécution total de la cellule.
*   **`%who`** : Affiche la liste de toutes les variables créées dans votre session.
*   **`%pip install <nom>`** : Installe une bibliothèque sans quitter le notebook (utile si `uv` est mal configuré).
*   **`!commande`** : Permet de lancer une commande shell système (ex: `!ls` sur Mac/Linux ou `!dir` sur Windows).

---

## II. Sommation d'Einstein (`np.einsum`)

La convention d'Einstein permet de simplifier les opérations complexes sur les tenseurs en définissant explicitement les axes de sommation.

### 1. La règle d'or
La chaîne de caractères `'ij,jk->ik'` se décompose ainsi :
1.  **Entrées** (avant la virgule) : Les lettres sont les indices des tenseurs d'entrée.
2.  **Sortie** (après `->`) : Les lettres sont les indices que l'on veut conserver.
3.  **Contraction** : Tout indice présent dans l'entrée mais **absent** de la sortie est automatiquement sommé.

### 2. Tableau de correspondance (Exemples)

| Opération | Notation Mathématique | Chaîne `einsum` | Explication |
| :--- | :--- | :--- | :--- |
| **Trace** | $\sum A_{ii}$ | `'ii->'` | Somme des éléments diagonaux. |
| **Somme totale** | $\sum_{i,j} A_{ij}$ | `'ij->'` | Somme de tous les éléments. |
| **Somme colonnes** | $\sum_{i} A_{ij}$ | `'ij->j'` | Somme sur l'axe des lignes (résultat = vecteur). |
| **Transposée** | $A_{ji} = A_{ij}$ | `'ij->ji'` | Inverse l'ordre des indices. |
| **Produit scalaire** | $\sum u_i v_i$ | `'i,i->'` | Produit et somme de deux vecteurs. |
| **Produit matriciel** | $\sum_k A_{ik} B_{kj}$ | `'ik,kj->ij'` | Somme sur l'indice commun $k$. |
| **Attention (IA)** | $\sum_d Q_{hd} K_{hd}$ | `'bhqd, bhkd -> bhqk'` | Contraction sur l'indice de dimension `d`. |

---

## III. Conseils de structuration pour vos rapports

Pour que vos Notebooks soient des documents académiques et professionnels (comme demandé dans votre TP) :

1.  **Utilisez LaTeX pour les formules :**
    *   `$ ... $` pour une formule intégrée au texte (ex : $A \in \mathbb{R}^{n \times n}$).
    *   `$$ ... $$` pour une formule centrée.
2.  **Affichage propre des résultats :**
    *   Ne vous contentez jamais de `print(matrice)`.
    *   Utilisez `sp.pprint(matrice)` (pour SymPy) ou `print(np.round(matrice, 4))` pour avoir un affichage lisible de vos calculs.
3.  **Documentation :**
    *   Chaque bloc de code doit être précédé d'une cellule **Markdown** expliquant l'intention mathématique (ex : "Calcul du gradient pour minimiser la fonction de coût").
    *   Utilisez les blocs `###` pour hiérarchiser vos exercices, exactement comme dans votre TP.

---
*Astuce : Si `einsum` devient trop complexe, utilisez `np.einsum(..., optimize=True)`. Cela laisse NumPy trouver la manière la plus efficace (l'ordre des calculs) pour exécuter l'opération sur vos données.*