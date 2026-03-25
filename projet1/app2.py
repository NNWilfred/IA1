import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro

# CHARGEMENT 
st.cache_data
def load_data():
    data = pd.read_csv("BeansDataSet.csv")
    data.columns = data.columns.str.strip()
    return data

df = load_data()

#  TITRE 
st.title("Dashboard Beans & Pods")

#  MENU (RADIO au lieu de selectbox) 
choix = st.sidebar.radio(
    "Menu",
    ["Données", "Statistiques", "Visualisation", "Interprétation", "GitHub"]
)

#  DONNÉES 
if choix == "Données":
    st.header("Aperçu du dataset")

    if st.checkbox("Afficher les 5 premières lignes"):
        st.dataframe(df.head())

    if st.checkbox("Afficher toutes les colonnes"):
        st.write(df.columns)

#  STATISTIQUES 
elif choix == "Statistiques":
    st.header("Analyse statistique")

    colonnes = df.select_dtypes(include="number").columns
    col = st.selectbox("Choisir une variable :", colonnes)

    st.write("Moyenne :", df[col].mean())
    st.write("Médiane :", df[col].median())
    st.write("Variance :", df[col].var())
    st.write("Écart-type :", df[col].std())

# VISUALISATION
elif choix == "Visualisation":
    st.header("Graphiques")

    type_graph = st.selectbox(
        "Choisir un graphique",
        ["Histogramme", "Boxplot", "Barres", "Corrélation"]
    )

    if type_graph == "Histogramme":
        col = st.selectbox("Variable :", df.select_dtypes("number").columns)
        fig, ax = plt.subplots()
        df[col].hist(bins=20, ax=ax)
        st.pyplot(fig)

    elif type_graph == "Boxplot":
        col = st.selectbox("Variable :", df.select_dtypes("number").columns)
        fig, ax = plt.subplots()
        ax.boxplot(df[col], vert=False)
        st.pyplot(fig)

    elif type_graph == "Barres":
        group = st.selectbox("Regrouper par :", ["Channel", "Region"])
        fig, ax = plt.subplots()
        df.groupby(group).sum(numeric_only=True).plot(kind="bar", ax=ax)
        st.pyplot(fig)

    else:
        fig, ax = plt.subplots()
        sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

#  INTERPRÉTATION 
elif choix == "Interprétation":
    st.header("Analyse et conclusions")

    # calculs
    produit = df.sum(numeric_only=True).idxmax()
    canal = df.groupby("Channel").sum(numeric_only=True).sum(axis=1).idxmax()
    region = df.groupby("Region").sum(numeric_only=True).sum(axis=1).idxmax()

    st.write(f"Le produit le plus vendu est : {produit}")
    st.write(f"Le canal dominant est : {canal}")
    st.write(f"La région dominante est : {region}")

    st.subheader("Recommandations")
    st.write("Investir dans le canal le plus performant")
    st.write("Mettre en avant le produit le plus vendu")
    st.write("Renforcer les ventes dans la meilleure région")

    # Test normalité
    st.subheader("Test de normalité")
    stat, p = shapiro(df["Espresso"])

    if p > 0.05:
        st.write("Distribution normale")
    else:
        st.write("Distribution non normale")

# Lien GITHUB 

    "https://github.com/NNWilfred/IA1.git:"
