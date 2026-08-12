import streamlit as st
from engine import charger_graphe_depuis_csv, page_rank_scratch

st.title(" PageRank : Analyse du Web")

# Chargement des données
try:
    adj, noms_pages = charger_graphe_depuis_csv('liens.csv')
    
    # Calcul du PageRank
    scores = page_rank_scratch(adj)
    
    st.subheader("Classement des pages")
    # On trie les résultats pour afficher les plus populaires en premier
    indices_tries = sorted(range(len(scores)), key=lambda k: scores[k], reverse=True)
    
    for i in indices_tries:
        st.write(f"**{noms_pages[i]}** : {scores[i]:.4f}")
        
except FileNotFoundError:
    st.error("Le fichier 'liens.csv' est introuvable. Créez-le dans le dossier du projet.")