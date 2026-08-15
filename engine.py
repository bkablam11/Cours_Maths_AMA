# Auteur : PIIA - Machine Learning
# Description : Moteur PageRank implémenté "from scratch"
import csv

def charger_graphe_depuis_csv(fichier_csv):
    """Lit un CSV et renvoie la matrice d'adjacence et les pages."""
    try:
        liens = []
        pages = set()
        with open(fichier_csv, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Vérification de l'existence des clés
                if 'source' in row and 'cible' in row:
                    liens.append((row['source'], row['cible']))
                    pages.add(row['source'])
                    pages.add(row['cible'])
        
        if not pages:
            raise ValueError("Le fichier CSV est vide ou mal formaté.")
            
        liste_pages = sorted(list(pages))
        page_to_id = {page: i for i, page in enumerate(liste_pages)}
        N = len(liste_pages)
        
        adj = [[0 for _ in range(N)] for _ in range(N)]
        for source, cible in liens:
            i, j = page_to_id[source], page_to_id[cible]
            adj[i][j] = 1
            
        return adj, liste_pages
    except FileNotFoundError:
        print(f"Erreur : Le fichier {fichier_csv} n'existe pas.")
        return [], []

def produit_matrice_vecteur(matrice, vecteur):
    """Calcule le produit v * M (vecteur ligne * matrice).

    Args:
        matrice (_type_): _description_
        vecteur (_type_): _description_

    Returns:
        _type_: _description_
    """
    N = len(matrice)
    resultat = [0.0] * N
    # On itère pour v * G
    for j in range(N):
        for i in range(N):
            resultat[j] += vecteur[i] * matrice[i][j]
    return resultat

def page_rank_scratch(adj, alpha=0.85, tol=1e-10, max_iter=1000):
    """
    Calcule le score PageRank d'un graphe à partir de sa matrice d'adjacence.

    Args:
        adj (list): Liste de listes, Matrice d'adjacence du graphe.
        alpha (float, optional): Facteur d'amortissement. Defaults to 0.85.
        tol (float, optional): Tolérance de convergence. Defaults to 1e-10.
        max_iter (int, optional): Nombre maximum d'itérations. Defaults to 1000.

    Returns:
        list: Vecteur contenant les scores d'importance de chaque page.
    
    """
    N = len(adj)
    
    # 1. Calcul des degrés sortants (somme de chaque ligne)
    out_degrees = [sum(row) for row in adj]
    
    # 2. Construction de P (matrice stochastique)
    P = [[0.0 for _ in range(N)] for _ in range(N)]
    for i in range(N):
        if out_degrees[i] > 0:
            for j in range(N):
                P[i][j] = adj[i][j] / out_degrees[i]
        else:
            # Traitement des 'dangling nodes' : distribution uniforme
            for j in range(N):
                P[i][j] = 1.0 / N
                
    # 3. Construction de la matrice de Google G = alpha * P + (1 - alpha) / N
    G = [[0.0 for _ in range(N)] for _ in range(N)]
    damping_factor = (1.0 - alpha) / N
    for i in range(N):
        for j in range(N):
            G[i][j] = alpha * P[i][j] + damping_factor
            
    # 4. Power Iteration : v_next = v * G
    v = [1.0 / N] * N
    for _ in range(max_iter):
        v_next = produit_matrice_vecteur(G, v)
        
        # Calcul de la norme L1 (distance entre deux vecteurs)
        diff = sum(abs(v_next[i] - v[i]) for i in range(N))
        
        v = v_next
        if diff < tol:
            break
            
    return v

# --- Test de vérification ---
""" if __name__ == "__main__":
    adj_test = [
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0]
    ]
    scores = page_rank_scratch(adj_test)
    print(f"Scores PageRank : {[round(s, 4) for s in scores]}") """