import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.decomposition import PCA
import plotly.express as px

# Charger les données Iris
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
iris = pd.read_csv(url, header=None, names=columns)

# Configuration de la page
st.set_page_config(page_title="Exploration Iris", page_icon="🌸")

st.title("Exploration de la base de données Iris 🌸")

# **Barre latérale (sidebar) pour les options**
st.sidebar.header("Options")
species = st.sidebar.selectbox(
    "Choisissez une espèce", 
    options=["Toutes les espèces"] + list(iris['species'].unique())
)

graph_choice = st.sidebar.selectbox(
    "Choisissez un graphique à afficher", 
    ["Distribution de la longueur des sépales", 
     "Relation entre longueur et largeur des pétales", 
     "PCA en 3D",
     "Comparaison de la largeur des sépales par espèce"]
)

# Filtrage des données en fonction du choix
if species == "Toutes les espèces":
    filtered_data = iris  # Toutes les données
else:
    filtered_data = iris[iris['species'] == species]

# Affichage des données filtrées
st.subheader(f"Données pour : {species}")
st.dataframe(filtered_data)

#Statistique descriptive
st.subheader("Statistiques")
st.write(iris.describe())

# **Affichage des graphiques selon le choix**
if graph_choice == "Distribution de la longueur des sépales":
    st.subheader("Graphe 📊: Distribution de la longueur des sépales")
    sns.set_theme(style="darkgrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data=iris, x="sepal_length", hue="species", kde=True, ax=ax)
    st.pyplot(fig)

elif graph_choice == "Relation entre longueur et largeur des pétales":
    st.subheader("Graphe 📊: Relation entre longueur et largeur des pétales")
    sns.set_theme(style="darkgrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=iris, x="petal_length", y="petal_width", hue="species", ax=ax)
    st.pyplot(fig)

elif graph_choice == "PCA en 3D":
    st.subheader("Graphe 📊: Projection PCA en 3D")

    # Encodage des espèces
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    iris['species_encoded'] = label_encoder.fit_transform(iris['species'])

    # Préparation des données pour PCA
    X = iris.drop(columns=["species", "species_encoded"])
    y = iris["species_encoded"]

    # PCA
    pca = PCA(n_components=3)
    X_reduced = pca.fit_transform(X)

    # Graphique interactif 3D
    fig = px.scatter_3d(
        x=X_reduced[:, 0], 
        y=X_reduced[:, 1], 
        z=X_reduced[:, 2], 
        color=iris["species"],
        labels={"x": "1ère CP", "y": "2ème CP", "z": "3ème CP"},
        title="Premières composantes principales du PCA"
    )
    st.plotly_chart(fig)
elif graph_choice == "Comparaison de la largeur des sépales par espèce":
    st.subheader("Graphe 📊: Comparaison de la largeur des sépales par espèce")
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(
        data=filtered_data,
        x="species",
        y="sepal_width",
        notch=True,
        width=0.5,
        boxprops=dict(facecolor="blue", edgecolor="blue"),
        medianprops=dict(color="red", linewidth=1),
        whiskerprops=dict(color="blue"),
        capprops=dict(color="blue"),
        flierprops=dict(marker="x", color="red", markersize=6),
        ax=ax
    )
    ax.set_ylim(2, 4.5)
    ax.set_title("Comparaison des espèces dans la base de données Iris", fontsize=12, loc="center", pad=15)
    ax.set_ylabel("Largeur des sépales (en mm)", fontsize=10)
    ax.set_xlabel("")
    st.pyplot(fig)
