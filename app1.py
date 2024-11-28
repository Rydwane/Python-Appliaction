import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.decomposition import PCA
import mpl_toolkits.mplot3d # pour la 3D
import plotly.express as px

#Télécharger la base de données
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
iris = pd.read_csv(url,header=None, names= columns)

st.set_page_config(page_title="Exploration Iris", page_icon="🌸")

st.title("Exploration de la base de données Iris")

#Données
st.subheader("Données Iris 📊")
st.dataframe(iris.head(50))

#Statistique descriptive
st.subheader("Statistiques")
st.write(iris.describe())

#Filtrage
species = st.selectbox("Choisissez une espèce", iris['species'].unique())
filtered_data = iris[iris['species'] == species]
st.subheader(f"Filtrer les données pour l'espèce : {species}")
st.dataframe(filtered_data)

#Choix du graphe
graph_choice = st.selectbox("Choisissez le graphique à afficher", ["Distribution de la longueur des sépales", 
                                                                   "Relation entre longueur et largeur des pétales",
                                                                   "PCA en 3D"])


if graph_choice == "Distribution de la longueur des sépales":
    st.subheader("Graphe 📊: Distribution de la longueur des sépales")
    sns.set_theme(style="darkgrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data=iris, x="sepal_length", hue="species", kde=True, ax= ax)
    st.pyplot(fig)
elif graph_choice == "Relation entre longueur et largeur des pétales":
    st.subheader("Graphe 📊: Relation entre longueur et largeur des pétales")
    sns.set_theme(style="darkgrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=iris, x="petal_length", y="petal_width",hue="species", ax= ax)
    st.pyplot(fig)
elif graph_choice == "PCA en 3D" :
    st.subheader("Graphe 📊: Projection PCA en 3D")

    # Préparer les données pour PCA
    from sklearn.preprocessing import LabelEncoder
    # Préparer les données pour PCA
    label_encoder = LabelEncoder()
    iris['species_encoded'] = label_encoder.fit_transform(iris['species'])

    X = iris.drop(columns=["species", "species_encoded"])
    y = iris["species_encoded"]

    # Ajouter un sélecteur de nombre de composantes principales
    num_components = 3

    # Appliquer PCA avec le nombre de composantes choisi
    pca = PCA(n_components=num_components)
    X_reduced = pca.fit_transform(X)

    # Créer un graphique 3D avec Plotly
    fig = px.scatter_3d(
        x=X_reduced[:, 0], 
        y=X_reduced[:, 1], 
        z=X_reduced[:, 2], 
        color=iris["species"],
        labels={"x": "1ère Eigenvector", "y": "2ème Eigenvector", "z": "3ème Eigenvector" if num_components == 3 else None},
        title="Premières composantes principales du PCA"
    )

    # Dynamique avec les options d'interaction Plotly
    fig.update_layout(
        scene=dict(
            xaxis_title="1ère Eigenvector",
            yaxis_title="2ème Eigenvector",
            zaxis_title="3ème Eigenvector" if num_components == 3 else None
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    # Afficher le graphique interactif
    st.plotly_chart(fig)