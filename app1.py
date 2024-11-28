import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.decomposition import PCA
import mpl_toolkits.mplot3d # pour la 3D

#Télécharger la base de données
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
iris = pd.read_csv(url,header=None, names= columns)

st.title("Exploration de la base de données Iris")

#Données
st.subheader("Données Iris")
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
    st.subheader("Graphe: Distribution de la longueur des sépales")
    fig, ax = plt.subplots()
    sns.histplot(data=iris, x="sepal_length", hue="species", kde=True, ax= ax)
    st.pyplot(fig)
elif graph_choice == "Relation entre longueur et largeur des pétales":
    st.subheader("Graphe: Relation entre longueur et largeur des pétales")
    fig, ax = plt.subplots()
    sns.scatterplot(data=iris, x="petal_length", y="petal_width",hue="species", ax= ax)
    st.pyplot(fig)
elif graph_choice == "PCA en 3D" :
    st.subheader("Graphique : Projection PCA en 3D")

    # Préparer les données pour PCA
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    iris['species_encoded'] = label_encoder.fit_transform(iris['species'])

    X = iris.drop(columns=["species", "species_encoded"])
    y = iris["species_encoded"]

    # Appliquer PCA pour réduire les dimensions à 3
    pca = PCA(n_components=3)
    X_reduced = pca.fit_transform(X)

    # Créer le graphique 3D
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d", elev=-150, azim=110)

    # Tracer les points dans l'espace 3D
    ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=y, s=40, cmap='viridis')

    # Titres et labels
    ax.set_title("Premières trois dimensions PCA")
    ax.set_xlabel("1ère Eigenvector")
    ax.set_ylabel("2ème Eigenvector")
    ax.set_zlabel("3ème Eigenvector")

    st.pyplot(fig)  # Afficher le graphique avec Streamlit
    