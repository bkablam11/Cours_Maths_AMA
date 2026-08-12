# TP — Algèbre linéaire numérique et expériences

Ce dépôt contient les travaux pratiques (TP) et scripts associés pour les
exercices d'algèbre linéaire numérique réalisés dans le cadre du cours.
Le point d'attention principal : la plupart des calculs matriciels sont
explicitement écrits avec la convention d'Einstein via `numpy.einsum`.

**Racine du dépôt**: contient notebooks, un petit module utilitaire et des
ressources pour les expérimentations numériques et visuelles.

**Versions / dépendances**
- Python 3.8+ recommandé
- numpy, scipy, matplotlib, jupyterlab / notebook
- Les dépendances exactes peuvent être consultées dans `pyproject.toml`.

**Installation rapide**
1. Créer un environnement virtuel : `uv init`
2. Activer : `source .venv/bin/activate` (macOS / Linux)
3. Installer dépendances : `uv add -r requirements.txt` ou
	 `uv add numpy scipy matplotlib jupyterlab`

**Fichiers importants**
- [TP2_algebre_Lineaire.ipynb](TP2_algebre_Lineaire.ipynb) : Notebook principal.
	- Partie I : exercices théoriques et démonstrations (SVD, propriétés matricielles).
	- Partie II : implémentations pratiques — `np.einsum` partout, visualisations
		("ravin"), power-iteration, self-attention, PageRank, et benchmarks.
- [projet_gradient.ipynb](projet_gradient.ipynb) & [projet_gradient.html](projet_gradient.html) : projet sur les méthodes de gradient (visualisations HTML incluse).
- [main.py](main.py) : script d'exemple / point d'entrée minimal (si présent).
- [engine.py](engine.py) : module utilitaire pédagogique (produit matrice‑vecteur,
	PageRank minimal). Contient des implémentations didactiques — préférer
	NumPy/SciPy pour la production.
- [pyproject.toml](pyproject.toml) : métadonnées du projet.
- [README.md](README.md) : ce fichier.
- Notebooks additionnels : [tp1.ipynb](tp1.ipynb), [test.ipynb](test.ipynb).

**Expériences et recommandations**
- Benchmarks lourds : certaines cellules du notebook utilisent des matrices
	aléatoires 1000×1000 pour comparer `np.einsum` vs opérateur `@`, et
	SVD vs power-iteration. Ces expériences sont coûteuses en CPU/mémoire —
	exécuter sur une machine avec suffisamment de RAM et prévoir du temps.
- Seeds : les notebooks utilisent des seeds (ex. 1, 3, 4) pour reproductibilité.
- Warm-up : les timings intègrent des exécutions de warm-up pour réduire
	l'impact du JIT/caches du système.

**Exemples d'exécution**
- Lancer le notebook principal :

```bash
jupyter lab TP2_algebre_Lineaire.ipynb
```

- Exécuter rapidement PageRank à petite échelle (dans un REPL Python) :

```python
from engine import page_rank_scratch
adj = [[0,1,0],[0,0,1],[1,0,0]]
print(page_rank_scratch(adj))
```

**Remarques pédagogiques**
- Les fonctions présentes sont volontairement explicites et lisibles pour
	l'apprentissage ; elles ne visent pas l'optimalité. Pour des graphes réels
	ou de très grande taille, convertir en tableaux NumPy et utiliser
	`scipy.sparse` est fortement recommandé.
- Le notebook illustre aussi la connexion entre la méthode de puissance et
	l'algorithme PageRank (itération sur la matrice de Google).

Souhaitez-vous que je :
- exécute le notebook et collecte les temps de benchmark ?
- ajoute un `requirements.txt` et un exemple de script d'exécution ?

