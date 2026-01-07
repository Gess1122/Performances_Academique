# Manipulation & calcul
import numpy as np
import pandas as pd
import streamlit as st

# Visualisation
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Statistiques
import scipy.stats as stats
import statsmodels.api as sm

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# Divers
import warnings
warnings.filterwarnings("ignore")

st.markdown("""
    <h2 style="
    text-align:center;
    background:linear-gradient(135deg,#0f2027,#203a43,#2c5364);
    border-radius:14px;
    margin-bottom:2.2em;
    color:white;
    font-family:'Playfair Display','Times New Roman',serif;
    padding:22px;
    letter-spacing:0.9px;
    box-shadow:0 6px 20px rgba(0,0,0,0.25);
">
    Dashboard interactif – Analyse des performances académiques des étudiants
</h2>

""", unsafe_allow_html=True)


df =pd.read_csv("data\student_500.csv")

st.sidebar.header("Apercus avant et apres  ")

### APERCU DES DATASET AVANT ET APRES NETTOYAGE

if st.sidebar.checkbox("DataSet Brute ") :
    
    st.write("#### DataSet Non Exploitable")
    st.write(df.head())

### nettoyage des donnees : Trouver les valeurs manquantes 

if df is not None :
    val_manq = df.isnull().sum() / df.shape[0]
    val_manq = val_manq[val_manq > 0].sort_values(ascending=False)

### supprimer ceux ayant un suil > 0.70
    seuil = 0.70

    supp_col = val_manq[val_manq > seuil].index
    df = df.drop(columns=supp_col)

### separer les valeurs numeriques des valeurs categorielles 
    df_num = df.select_dtypes(include=['int64','float64','bool'])
    df_cat = df.select_dtypes(exclude=['int64','float64','bool'])

### imputation des differentes valeurs par la moyenne(numerique) et le mode (categorielle)
    df_num = df_num.fillna(df_num.mean())
    df_cat = df_cat.fillna(df_cat.mode().iloc[0])

### rassemblertous les variables en un seul dataset 
    df = pd.concat([df_num, df_cat], axis = 1)

### apercu du dataset nettoyer
if st.sidebar.checkbox("DataSet Exploitable") :
    st.write("#### DataSet Exploitable")
    st.write(df.head())



st.sidebar.header("Filtres de Recherche 🔍")

if st.sidebar.checkbox("Statistiques descriptive"):
    st.write("Statistique numerique")
    st.write(df.describe())

    st.write("Statistique categorielles")
    st.write(df.describe(include=object))

if st.sidebar.checkbox("Nombre Femme et Homme"):
    st.write("## Nombre de Femme et d'Homme")
    
    femme = df[df['gender'] == "Femme"].shape[0]
    

    homme = df[df['gender'] == "Homme"].shape[0]

    st.write(f"Femmes : {femme}")
    st.write(f"Hommes : {homme}")

if st.sidebar.checkbox("Nombre d'echec et de reussite"):
    st.write("## Nombre d'echec et de reussite")
    
    reussi = df[df['final_result'] == "Réussite"].shape[0]
    

    echec = df[df['final_result'] == "Échec"].shape[0]

    st.write(f"Réussite : {reussi}")
    st.write(f"Échec : {echec}")


tab1, tab2, tab3 = st.tabs([
    "Vue d'ensemble",
    "performance academiques",
    "Analyse par profile",
    
])

with tab1 :
    st.subheader("Repartition reussite/ echec")

    #Filtres
    genre = st.selectbox("Genre", ["Tous"] + df["gender"].unique().tolist())
    inter = st.selectbox("Accès Internet", ["Tous", "Oui", "Non"])

    df_fil = df.copy()

    if genre != "Tous":
        df_fil = df_fil[df_fil["gender"] == genre]

    if inter != "Tous":
        df_fil = df_fil[df_fil["internet_access"] == inter]

    result = df_fil["final_result"].value_counts()

    fig, ax = plt.subplots()
    ax.bar(result.index, result.values)
    st.pyplot(fig)

with tab2:
    st.subheader("Analyse des performances académiques selon la réussite")

    sujets = {
        "Mathématiques": "math_score",
        "Programmation": "programming_score",
        "Statistiques": "statistics_score"
    }

    sujet_label = st.selectbox(
        " Choisir une matière",
        list(sujets.keys())
    )

    sujets = sujets[sujet_label]  # vraie colonne

    col1, col2 = st.columns(2)

    #  BOXPLOT
    with col1:
        fig, ax = plt.subplots(figsize=(5,4))
        df.boxplot(column=sujets, by="final_result", ax=ax)
        plt.suptitle("")
        ax.set_title(f"Boxplot – {sujet_label}")
        ax.set_xlabel("Résultat final")
        ax.set_ylabel("Note")
        st.pyplot(fig)

    # VIOLIN PLOT
    with col2:
        fig, ax = plt.subplots(figsize=(5,4))
        sns.violinplot(
            data=df,
            x="final_result",
            y=sujets,
            inner="quartile",
            ax=ax
        )
        ax.set_title(f"Distribution – {sujet_label}")
        ax.set_xlabel("Résultat final")
        ax.set_ylabel("Note")
        st.pyplot(fig)

    #  INTERPRÉTATION
    mean_scores = df.groupby("final_result")[sujets].mean()

    st.info(
        f"**Interprétation** : la moyenne en **{sujet_label}** est de "
        f"**{mean_scores['Réussite']:.2f}** pour les étudiants en réussite "
        f"contre **{mean_scores['Échec']:.2f}** pour ceux en échec."
    )

with tab3:
    st.subheader("Analyse par profil étudiant")

    col1, col2 = st.columns(2)

    #  ÂGE
    with col1:
        fig, ax = plt.subplots(figsize=(5,4))
        ax.hist(df["age"], bins=10)
        ax.set_title("Distribution des âges")
        ax.set_xlabel("Âge")
        ax.set_ylabel("Effectif")
        st.pyplot(fig)

    #  GENRE vs RÉSULTAT
    with col2:
        genre_resul = df.groupby(["gender", "final_result"]).size().unstack()
        fig, ax = plt.subplots(figsize=(5,4))
        genre_resul.plot(kind="bar", ax=ax)
        ax.set_title("Réussite selon le genre")
        ax.set_xlabel("Genre")
        ax.set_ylabel("Nombre d'étudiants")
        st.pyplot(fig)

    st.divider() ### on va a la ligne

    col3, col4 = st.columns(2)

    #  BOURSE vs RÉSULTAT
    with col3:
        bourse_resul = df.groupby(
            ["scholarship", "final_result"]
        ).size().unstack()

        fig, ax = plt.subplots(figsize=(5,4))
        bourse_resul.plot(kind="bar", ax=ax)
        ax.set_title("Impact de la bourse sur la réussite")
        ax.set_xlabel("Bourse")
        ax.set_ylabel("Nombre d'étudiants")
        st.pyplot(fig)

    # INTERNET vs RÉSULTAT
    with col4:
        intern_resul = df.groupby(
            ["internet_access", "final_result"]
        ).size().unstack()

        fig, ax = plt.subplots(figsize=(5,4))
        intern_resul.plot(kind="bar", ax=ax)
        ax.set_title("Accès à Internet et réussite")
        ax.set_xlabel("Accès Internet")
        ax.set_ylabel("Nombre d'étudiants")
        st.pyplot(fig)

    st.divider()

    # PRÉSENCE EN COURS
    fig, ax = plt.subplots(figsize=(6,4))
    df.boxplot(column="attendance_rate", by="final_result", ax=ax)
    plt.suptitle("")
    ax.set_title("Taux de présence selon le résultat")
    ax.set_xlabel("Résultat final")
    ax.set_ylabel("Taux de présence (%)")
    st.pyplot(fig)

    #  INTERPRÉTATION AUTOMATIQUE
    success_rate = (
        df[df["final_result"] == "Réussite"]
        .groupby("scholarship")
        .size()
    )

    st.info(
        " **Interprétation générale** :\n"
        "- Les étudiants ayant un **taux de présence élevé** réussissent davantage.\n"
        "- L’**accès à Internet** et la **bourse** semblent avoir un impact positif.\n"
        "- Des différences apparaissent selon le **genre** et l’**âge**, mais avec un effet plus modéré."
    )

