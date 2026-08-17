import numpy as np
import time
import matplotlib.pyplot as plt

def fonction_cout(theta, X, y):
    """
    Calcule l'Erreur Quadratique Moyenne (MSE - Mean Squared Error).
    Cette fonction représente la "surface de coût" que l'on cherche à minimiser.
    
    Paramètres :
    -----------
    theta : numpy.ndarray
        Le vecteur des coefficients (poids) du modèle.
    X : numpy.ndarray
        La matrice des caractéristiques (features), de dimension (m_exemples, n_features).
    y : numpy.ndarray
        Le vecteur des valeurs réelles cibles, de dimension (m_exemples,).
        
    Retourne :
    ---------
    float
        La valeur scalaire de l'erreur moyenne commise par le modèle.
    """
    m = len(y)
    return (1 / (2 * m)) * np.sum((X @ theta - y) ** 2)

def gradient(theta, X, y):
    """
    Calcule le gradient analytique de la fonction de coût par rapport à theta.
    Il indique la direction de la plus forte pente (la montée). L'optimiseur 
    ira donc dans la direction opposée pour minimiser l'erreur.
    
    Paramètres :
    -----------
    theta : numpy.ndarray
        Le vecteur actuel des coefficients du modèle.
    X : numpy.ndarray
        La matrice des caractéristiques des données.
    y : numpy.ndarray
        Le vecteur des valeurs cibles réelles.
        
    Retourne :
    ---------
    numpy.ndarray
        Un vecteur contenant les dérivées partielles par rapport à chaque paramètre.
    """
    m = len(y)
    return (X.T @ (X @ theta - y)) / m

def optimiseur_adam(X, y, theta_init, alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, n_iter=100):
    """
    Implémente l'algorithme d'optimisation Adam (Adaptive Moment Estimation).
    Il combine l'inertie du mouvement (Momentum) et des taux d'apprentissage adaptatifs 
    par paramètre, avec une correction de biais pour les premières itérations.
    
    Paramètres :
    -----------
    X : numpy.ndarray
        Matrice des données d'entrée.
    y : numpy.ndarray
        Vecteur des étiquettes/valeurs cibles.
    theta_init : numpy.ndarray
        Point de départ (initialisation des poids).
    alpha : float
        Le taux d'apprentissage (stepsize), contrôle la taille des pas.
    beta1 : float
        Taux de désintégration exponentielle pour le 1er moment (moyenne des gradients).
    beta2 : float
        Taux de désintégration exponentielle pour le 2ème moment (variance des gradients).
    epsilon : float
        Petite constante de sécurité pour éviter de diviser par zéro.
    n_iter : int
        Nombre total d'itérations (époques) de l'optimisation.
        
    Retourne :
    ---------
    tuple (numpy.ndarray, list)
        Le vecteur final des paramètres optimisés et l'historique des coûts par itération.
    """
    theta = theta_init.copy()
    m = np.zeros_like(theta)  # Initialisation du 1er moment (moyenne) à 0
    v = np.zeros_like(theta)  # Initialisation du 2ème moment (variance non centrée) à 0
    historique_cout = []
    
    for t in range(1, n_iter + 1):
        gt = gradient(theta, X, y) # Étape 1 : Calcul du gradient actuel
        
        # Étape 2 : Mise à jour des moyennes mobiles
        m = beta1 * m + (1 - beta1) * gt
        v = beta2 * v + (1 - beta2) * (gt ** 2)
        
        # Étape 3 : Correction du biais d'initialisation (très important au début quand m et v sont proches de 0)
        m_hat = m / (1 - (beta1 ** t))
        v_hat = v / (1 - (beta2 ** t))
        
        # Étape 4 : Mise à jour des paramètres du modèle
        theta = theta - (alpha / (np.sqrt(v_hat) + epsilon)) * m_hat
        historique_cout.append(fonction_cout(theta, X, y))
        
    return theta, historique_cout

def optimiseur_l_bfgs(X, y, theta_init, m_history=10, n_iter=100):
    """
    Implémente l'algorithme quasi-Newton L-BFGS (Limited-memory BFGS).
    Il approxime l'inverse de la matrice Hessienne (la courbure de la fonction) 
    en n'utilisant qu'un historique limité de m pas précédents, ce qui économise énormément de mémoire.
    
    Paramètres :
    -----------
    X : numpy.ndarray
        Matrice des données d'entrée.
    y : numpy.ndarray
        Vecteur des valeurs cibles.
    theta_init : numpy.ndarray
        Point de départ (initialisation des poids).
    m_history : int
        Taille de la mémoire (nombre de corrections passées s et y conservées).
    n_iter : int
        Nombre total d'itérations de l'optimisation.
        
    Retourne :
    ---------
    tuple (numpy.ndarray, list)
        Le vecteur final des paramètres optimisés et l'historique des coûts par itération.
    """
    theta = theta_init.copy()
    historique_cout = []
    s_list, y_list = [], [] # Listes pour stocker l'historique des variations (s_k et y_k)
    
    for k in range(n_iter):
        gk = gradient(theta, X, y)
        historique_cout.append(fonction_cout(theta, X, y))
        
        if k == 0:
            pk = -gk  # Première itération : on se comporte comme une simple descente de gradient
        else:
            # Algorithme 3 : Récursion à deux boucles (Two-loop recursion) pour trouver la direction pk
            q = gk.copy()
            alphas = []
            for s, y_vec in reversed(list(zip(s_list, y_list))):
                rho = 1.0 / (y_vec @ s)
                alpha = rho * (s @ q)
                alphas.append(alpha)
                q -= alpha * y_vec
            
            # Estimation initiale de la matrice de mise à l'échelle H0 par un scalaire (gamma)
            gamma = (s_list[-1] @ y_list[-1]) / (y_list[-1] @ y_list[-1])
            r = gamma * q
            
            # Seconde boucle de récursion pour combiner l'historique et trouver la direction finale
            for (s, y_vec), alpha in zip(zip(s_list, y_list), reversed(alphas)):
                rho = 1.0 / (y_vec @ s)
                beta = rho * (y_vec @ r)
                r += s * (alpha - beta)
            pk = -r  # Direction de recherche finale corrigée par la courbure
            
        # Application d'un pas fixe (line search simplifiée) pour avancer dans la direction pk
        alpha_step = 0.01
        theta_next = theta + alpha_step * pk
        
        # Mémorisation des deltas de position (s) et de gradient (y) pour l'historique L-BFGS
        sk = theta_next - theta
        yk_vec = gradient(theta_next, X, y) - gk
        
        s_list.append(sk)
        y_list.append(yk_vec)
        
        # Si l'historique dépasse la taille maximale 'm_history', on supprime le plus ancien (FIFO)
        if len(s_list) > m_history:
            s_list.pop(0)
            y_list.pop(0)
            
        theta = theta_next
        
    return theta, historique_cout

# --- Génération de données de test synthétiques ---
np.random.seed(42)
X_synth = np.c_[np.ones(500), np.random.rand(500, 5)] # 500 exemples, 5 features + biais
vrai_theta = np.array([2.0, -1.0, 3.5, 0.5, -2.0, 1.0])
y_synth = X_synth @ vrai_theta + np.random.randn(500) * 0.1 # Ajout d'un léger bruit gaussien
theta_0 = np.zeros(6) # Initialisation des poids à zéro

# --- Exécution et Mesure du Temps d'entraînement pour Adam ---
t_start_adam = time.time()
th_adam, cout_adam = optimiseur_adam(X_synth, y_synth, theta_0, alpha=0.05, n_iter=200)
t_adam = time.time() - t_start_adam

# --- Exécution et Mesure du Temps d'entraînement pour L-BFGS ---
t_start_lbfgs = time.time()
th_lbfgs, cout_lbfgs = optimiseur_l_bfgs(X_synth, y_synth, theta_0, m_history=5, n_iter=200)
t_lbfgs = time.time() - t_start_lbfgs

# Affichage des performances temporelles dans la console
print(f"Temps Adam : {t_adam:.4f}s | Temps L-BFGS : {t_lbfgs:.4f}s")

# --- Configuration et Affichage des graphiques comparatifs de convergence ---
plt.figure(figsize=(7, 3.5))
plt.plot(cout_adam, label=f"Adam (Temps: {t_adam:.3f}s)", color="red")
plt.plot(cout_lbfgs, label=f"L-BFGS (Temps: {t_lbfgs:.3f}s)", color="blue")
plt.xlabel("Itérations")
plt.ylabel("Fonction de Coût (MSE)")
plt.legend()
plt.title("Comparaison de Convergence : Adam vs L-BFGS")
plt.grid(True)
plt.show()