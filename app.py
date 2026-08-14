import streamlit as st
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.cm as cm
from engine import charger_graphe_depuis_csv, page_rank_scratch

st.title(" PageRank : Analyse du Web et Visualisation")

# Chargement des données
try:
    adj, noms_pages = charger_graphe_depuis_csv('liens.csv')
    
    # Calcul du PageRank
    scores = page_rank_scratch(adj)
    
    # --- SECTION 1 : VISUALISATION DU GRAPHE STYLE "IMAGE 2" ---
    st.subheader(" Visualisation interactive du PageRank")
    
    # Création du graphe orienté
    G = nx.DiGraph()
    N = len(noms_pages)
    for i in range(N):
        for j in range(N):
            if adj[i][j] == 1:
                G.add_edge(noms_pages[i], noms_pages[j])
                
    fig_graph, ax_graph = plt.subplots(figsize=(9, 7))
    
    # Algorithme de disposition
    pos = nx.spring_layout(G, seed=42, k=0.8)
    
    # 1. Calcul des tailles de nœuds proportionnelles
    node_sizes = [max(v * 12000, 800) for v in scores]
    
    # 2. Attribution des couleurs selon les scores
    norm_scores = [(s - min(scores)) / (max(scores) - min(scores) + 1e-9) for s in scores]
    node_colors = [cm.plasma(ns) for ns in norm_scores]
    
    # 3. Dessin des arêtes (FLÈCHES RENDUES TRÈS VISIBLES)
    nx.draw_networkx_edges(
        G, pos, 
        ax=ax_graph, 
        edge_color="#2c3e50",       # Couleur sombre bien contrastée
        width=1.8,                  # Épaisseur des lignes augmentée
        arrows=True, 
        arrowsize=25,               # Taille des pointes de flèches bien plus grande
        arrowstyle='-|>', 
        connectionstyle='arc3,rad=0.2' # Courbure accentuée pour éviter les superpositions
    )
    
    # 4. Dessin des nœuds (cercles colorés et proportionnels)
    nodes = nx.draw_networkx_nodes(
        G, pos, 
        node_size=node_sizes, 
        node_color=node_colors, 
        ax=ax_graph, 
        alpha=0.9
    )
    
    # 5. Étiquettes personnalisées
    labels = {noms_pages[i]: f"{noms_pages[i]}\n({scores[i]*100:.1f}%)" for i in range(N)}
    nx.draw_networkx_labels(
        G, pos, 
        labels=labels, 
        font_size=10, 
        font_weight='bold', 
        font_color='black', 
        ax=ax_graph
    )
    
    ax_graph.axis('off')
    st.pyplot(fig_graph)
    
    # --- SECTION 2 : CLASSEMENT TEXTUEL & BARRES ---
    st.subheader(" Classement et Scores Détaillés")
    indices_tries = sorted(range(len(scores)), key=lambda k: scores[k], reverse=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Top Pages")
        for i in indices_tries:
            st.write(f"**{noms_pages[i]}** : `{scores[i]*100:.2f}%` ({scores[i]:.4f})")
            
    with col2:
        st.markdown("### Graphique des Scores")
        fig_bar, ax_bar = plt.subplots(figsize=(5, 4))
        # Trier les noms et scores pour le diagramme en barres
        noms_tries = [noms_pages[i] for i in indices_tries]
        scores_tries = [scores[i] for i in indices_tries]
        
        ax_bar.barh(noms_tries[::-1], scores_tries[::-1], color='orange')
        ax_bar.set_xlabel('Score PageRank')
        st.pyplot(fig_bar)
        
except FileNotFoundError:
    st.error("Le fichier 'liens.csv' est introuvable. Vérifie sa présence dans le dossier.")