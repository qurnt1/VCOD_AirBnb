import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
from geopy.distance import geodesic
import numpy as np

# Configuration de la page Streamlit
st.set_page_config(page_title="Airbnb Paris Dashboard", layout="wide")

# Désactivation de l'affichage des erreurs côté client
st.set_option('client.showErrorDetails', False)

# Application de styles CSS pour la personnalisation de l'interface utilisateur
st.markdown(
    """
    <style>
        /* 🌑 Mode sombre global */
        html, body, .stApp {
            background-color: #0E1117 !important;
            color: white !important;
        }
    
        /* 🔲 Conteneur principal */
        [data-testid="stAppViewContainer"], [data-testid="block-container"] {
            background-color: #0E1117 !important;
        }
    
        /* 🌑 Sidebar et son icône */
        [data-testid="stSidebar"], .css-1d391kg {
            background-color: #0E1117 !important;
            color: white !important;
        }
    
        /* 🎨 Titres et texte */
        h1, h2, h3, h4, h5, h6, p, label, span {
            color: white !important;
        }
    
        /* 📌 Fix de la barre blanche en haut */
        [data-testid="stHeader"], header {
            background: #0E1117 !important;
        }
    
        /* 📌 Multiselect & autres inputs */
        .stMultiSelect div[data-baseweb="select"] {
            background-color: #1E222A !important;
            color: white !important;
        }
        .stMultiSelect div[data-baseweb="select"] div {
            background-color: #1E222A !important;
            color: white !important;
        }
    
        /* 📊 Valeurs des 'metric()' en blanc */
        div[data-testid="metric-container"] {
            color: white !important;
        }
        div[data-testid="metric-container"] > label {
            color: white !important;
        }
    
        /* 🎨 Légende des graphiques */
        .legendtext {
            fill: white !important;
        }

        /* 🏠 Barre de navigation */
        .nav-links {
            background-color: #0E1117; /* Fond sombre */
            padding: 10px; /* Espacement autour des éléments */
            text-align: center; /* Alignement du texte au centre */
            margin-bottom: 20px; /* Marge en bas */
        }
        .nav-links a {
            margin: 0 15px; /* Espacement entre les liens */
            text-decoration: none; /* Retirer la décoration par défaut */
            font-weight: bold; /* Texte en gras */
            color: white !important; /* Texte en blanc */
            background-color: #ff5d57; /* Fond des boutons */
            padding: 10px 20px; /* Espacement interne pour les boutons */
            border-radius: 25px; /* Bord arrondi pour rendre les boutons ronds */
            display: inline-block; /* Disposition des liens en ligne */
            transition: background-color 0.3s ease; /* Effet de transition pour le survol */
        }
        .nav-links a:hover {
            background-color: #e04b45; /* Changer la couleur de fond au survol */
        }

        /* 🎨 Personnalisation du bouton de téléchargement */
        .stDownloadButton button {
            background-color: #0E1117 !important;
            color: #ff5d57 !important;
            font-weight: bold;
            padding: 10px 20px;
            border-radius: 25px; /* Rendre le bouton rond */
        }
        .stDownloadButton button:hover {
            background-color: #ff5d57 !important;
            color: white !important;
        }
    
    </style>
    """, unsafe_allow_html=True
)

# Détecter le paramètre de l'URL pour déterminer la page active
query_params = st.experimental_get_query_params()

# Récupération du paramètre "page" de l'URL, valeur par défaut = "visualisation"
current_page = query_params.get("page", ["visualisation"])[0]

# Si le paramètre "page" est absent ou mal formé, définir "visualisation" comme page par défaut
if current_page not in ["page1", "page2", "page3", "page4"]:
    current_page = "page1"

# Affichage des liens de navigation en haut de page
st.markdown("""
<div class="nav-links">
    <a href="?page=page1" target="_self">Tableau de Bord</a>
    <a href="?page=page2" target="_self">Carte Choroplèthe</a>
    <a href="?page=page3" target="_self">Graphique Small Multiples</a>
    <a href="?page=page4" target="_self">Distance Monuments</a>
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------ 
# Chargement des données depuis listings.csv avec mise en cache pour améliorer la performance
# ------------------------------------------------------------------------------
@st.cache_data
def load_data():
    # Chargement du fichier CSV contenant les données des annonces Airbnb
    df = pd.read_csv("datas/data/listings.csv")
    
    # Conversion de la colonne 'last_review' en datetime, en gérant les erreurs
    if 'last_review' in df.columns:
        df['last_review'] = pd.to_datetime(df['last_review'], errors='coerce')
    
    # Remplacement des valeurs dans la colonne 'room_type' pour plus de lisibilité
    df['room_type'] = df['room_type'].replace({
        'Entire home/apt': 'Appartement entier',
        'Hotel room': 'Chambre d\'hôtel',
        'Private room': 'Chambre privée',
        'Shared room': 'Chambre partagée'
    })
    
    return df

# Chargement des données des listings
df = load_data()

# ------------------------------------------------------------------------------ 
# Chargement du fichier geojson pour les contours des arrondissements de Paris
# ------------------------------------------------------------------------------
@st.cache_data
def load_geojson():
    # Lecture du fichier geojson contenant les contours des arrondissements
    with open("datas/data/neighbourhoods.geojson", "r", encoding="utf-8") as f:
        geojson_data = json.load(f)
    
    return geojson_data

# Chargement des données geojson des arrondissements
geojson_data = load_geojson()

# ------------------------------------------------------------------------------ 
# Chargement des données des monuments depuis une URL externe
# ------------------------------------------------------------------------------
@st.cache_data
def load_monuments():
    # URL contenant les coordonnées des monuments nationaux
    url = "https://static.data.gouv.fr/resources/liste-des-coordonnees-gps-des-monuments-nationaux/20190729-144937/liste-coordonnees-gps-des-monuments.csv"
    
    # Chargement des données des monuments
    monuments = pd.read_csv(url, delimiter=',', encoding='latin1')
    
    # Limites géographiques de Paris pour filtrer les monuments dans cette zone
    lat_min, lat_max = 48.815573, 48.902144
    lon_min, lon_max = 2.224199, 2.469921

    # Filtrage des monuments pour ne conserver que ceux situés à Paris
    monuments = monuments[(monuments['latitude'] >= lat_min) & (monuments['latitude'] <= lat_max) &
                          (monuments['longitude'] >= lon_min) & (monuments['longitude'] <= lon_max)]
    
    return monuments

# Chargement des données des monuments
monuments = load_monuments()

# Contenu des pages
if current_page == "page1":

    # Affichage du logo et du titre
    st.image("datas/imgs/logo.png", width=150)  # Affiche le logo à gauche avec une largeur de 150px
    st.title("Airbnb Paris Dashboard")  # Affiche le titre principal
    st.markdown("### Analyse interactive des données Airbnb à Paris")  # Sous-titre descriptif
    
    # -------------------------------------------------------------------------------
    # Sidebar pour afficher les filtres
    # -------------------------------------------------------------------------------
    st.sidebar.header("Filtres")  # Titre de la sidebar

    # Filtre sur le type de logement
    room_types = df['room_type'].dropna().unique().tolist()  # Récupération des types de logement uniques dans les données
    selected_room_types = st.sidebar.multiselect("Type de logement", options=room_types, default=room_types)  # Sélection multiple des types de logement
    
    # Filtre sur le prix (slider)
    min_price = int(df['price'].min())  # Valeur minimale du prix
    max_price = int(df['price'].max())  # Valeur maximale du prix
    price_range = st.sidebar.slider("Prix (€)", min_value=min_price, max_value=max_price, 
                                    value=(min_price, max_price), step=5)  # Slider pour sélectionner une plage de prix
    
    # Filtre sur le nombre d'avis (slider)
    min_reviews = int(df['number_of_reviews'].min())  # Valeur minimale du nombre d'avis
    max_reviews = int(df['number_of_reviews'].max())  # Valeur maximale du nombre d'avis
    reviews_range = st.sidebar.slider("Nombre d'avis", min_value=min_reviews, max_value=max_reviews, 
                                    value=(min_reviews, max_reviews), step=1)  # Slider pour sélectionner une plage d'avis
    
    # Sélection des arrondissements à afficher
    arrondissement_names = [feature["properties"]["neighbourhood"] for feature in geojson_data["features"]]  # Liste des noms des arrondissements
    selected_arrondissements = st.sidebar.multiselect("Arrondissements", options=arrondissement_names, default=arrondissement_names)  # Sélection multiple des arrondissements
    
    # Sauvegarde des filtres dans un dictionnaire
    filters = {
        "selected_room_types": selected_room_types,  # Types de logement sélectionnés
        "price_range": price_range,  # Plage de prix sélectionnée
        "reviews_range": reviews_range,  # Plage d'avis sélectionnée
        "selected_arrondissements": selected_arrondissements  # Arrondissements sélectionnés
    }

    # Application des filtres sur le DataFrame
    df_filtered = df[df['room_type'].isin(filters["selected_room_types"]) &  # Filtrage selon le type de logement
                    df['price'].between(filters["price_range"][0], filters["price_range"][1]) &  # Filtrage selon le prix
                    df['number_of_reviews'].between(filters["reviews_range"][0], filters["reviews_range"][1])]  # Filtrage selon le nombre d'avis

    # Filtrage des logements selon les arrondissements sélectionnés
    if filters["selected_arrondissements"]:
        df_filtered = df_filtered[df_filtered["neighbourhood"].isin(filters["selected_arrondissements"])]  # Filtrage selon les arrondissements

    # -------------------------------------------------------------------------------
    # Affichage des indicateurs clés
    # -------------------------------------------------------------------------------
    # Affichage du titre des chiffres clés sur les logements filtrés
    st.markdown(f"### Chiffres clés sur les {df_filtered.shape[0]} logements filtrés")

    # Création de trois colonnes pour afficher des indicateurs clés
    col1, col2, col3 = st.columns(3)
    # Affichage du prix moyen des logements
    col1.metric("Prix moyen (€)", f"{df_filtered['price'].mean():.2f}")
    # Affichage de la médiane du prix des logements
    col2.metric("Médiane du prix (€)", f"{df_filtered['price'].median():.2f}")
    # Affichage du nombre moyen d'avis par logement
    col3.metric("Nombre moyen d'avis par logement", f"{df_filtered['number_of_reviews'].mean():.1f}")

    # Création de trois autres colonnes pour afficher des indicateurs supplémentaires
    col4, col5, col6 = st.columns(3)
    # Affichage de la disponibilité moyenne des logements sur une année
    col4.metric("Disponibilité moyenne (jours/an)", f"{df_filtered['availability_365'].mean():.1f}")

    # Affichage du quartier le plus populaire
    quartier_top = df_filtered['neighbourhood'].mode()[0] if not df_filtered['neighbourhood'].isna().all() else "N/A"
    col5.metric("Quartier le plus populaire", quartier_top)

    # Si la colonne 'calculated_host_listings_count' existe, afficher le taux de réponse moyen
    if "calculated_host_listings_count" in df_filtered.columns:
        taux_reponse_moyen = df_filtered["calculated_host_listings_count"].mean()
        col6.metric("Taux de réponse moyen (en %)", f"{taux_reponse_moyen:.1f}" if not df_filtered["calculated_host_listings_count"].isna().all() else "N/A")

    # Création d'une ligne avec deux colonnes pour d'autres indicateurs
    col7, col8, _ = st.columns(3)
    # Calcul du pourcentage de logements entiers
    percent_entire_home = (df_filtered["room_type"] == 'Appartement entier').sum() / df_filtered.shape[0] * 100
    col7.metric("Logements entiers (%)", f"{percent_entire_home:.1f}%")

    # Conversion du DataFrame filtré en CSV et ajout d'un bouton pour télécharger les données
    with col8:
        # Fonction pour convertir un DataFrame en CSV
        def convert_df_to_csv(df):
            return df.to_csv(index=False).encode('utf-8')

        # Création du bouton de téléchargement pour les données filtrées
        csv_data = convert_df_to_csv(df_filtered)
        st.download_button(
            label=f"Télécharger les {df_filtered.shape[0]} données filtrées",  # Texte du bouton avec le nombre de données
            data=csv_data,  # Données CSV à télécharger
            file_name=f"csv_donnees_filtrees_{'_'.join(filters['selected_room_types'])}_price_range_{str(filters['price_range']).replace(' ', '')}_reviews_range_{str(filters['reviews_range']).replace(' ', '')}.csv",  # Nom du fichier avec les filtres appliqués
            mime="text/csv"  # Type MIME du fichier
        )

    # Séparateur horizontal entre les différentes sections
    st.markdown("---")

    # Graphique 1 : Carte interactive des logements à Paris
    # Création d'une carte interactive avec Plotly
    fig_map = px.scatter_map(
        df_filtered,
        lat="latitude",  # Latitude des logements
        lon="longitude",  # Longitude des logements
        hover_name="name",  # Nom du logement au survol
        hover_data=["price", "room_type"],  # Affichage du prix et du type de logement au survol
        color="room_type",  # Couleur par type de logement
        zoom=11,  # Niveau de zoom initial
        height=600,  # Hauteur du graphique
        title="Localisation des logements à Paris",  # Titre du graphique
        labels={"room_type": "Type de logement"}  # Légende des couleurs
    )
    # Mise à jour de la mise en page de la carte interactive
    fig_map.update_layout(
        title=dict(
            text="Localisation des logements à Paris",  # Titre du graphique
            x=0.5,  # Centrer horizontalement
            xanchor="center",  # Centrer sur l'axe X
            font=dict(color="white")  # Titre en blanc
        ),
        legend=dict(
            font=dict(color="white")  # Légende en blanc
        ),
        mapbox=dict(
            style="satellite-streets",  # Utilisation du style satellite de Mapbox
            center={"lat": df_filtered["latitude"].mean(), "lon": df_filtered["longitude"].mean()},  # Centrer la carte sur les coordonnées moyennes des logements
            zoom=11  # Niveau de zoom initial
        ),
        plot_bgcolor="#0E1117",  # Fond du graphique en noir
        paper_bgcolor="#0E1117",  # Fond général en noir (contours)
        margin=dict(t=40, b=40, l=0, r=0)  # Ajuste les marges pour éviter les bords blancs
    )

    # Affichage de la carte interactive
    st.plotly_chart(fig_map, use_container_width=True)

    # Séparateur horizontal pour diviser les sections
    st.markdown("---")

    # Colonnes pour organiser les graphiques côte à côte
    col1, col2 = st.columns(2)

    # Graphique 2 : Box Plot - Distribution des prix par type de logement
    with col1:
        # Création d'un box plot pour la distribution des prix par type de logement
        fig_price = px.box(
            df_filtered, 
            x="room_type",  # Type de logement sur l'axe X
            y="price",  # Prix sur l'axe Y
            color="room_type",  # Couleur par type de logement
            labels={"room_type": "Type de logement", "price": "Prix (€)"}  # Légendes
        )

        # Mise à jour de la mise en page du box plot
        fig_price.update_layout(
            title=dict(
                text="Distribution des prix par type de logement",  # Titre du graphique
                x=0.5,  # Centrer horizontalement
                xanchor="center",  # Centrer sur l'axe X
                font=dict(color="white")  # Titre en blanc
            ),
            legend=dict(
                font=dict(color="white")  # Légende en blanc
            ),
            plot_bgcolor="#0E1117",  # Fond du graphique en noir
            paper_bgcolor="#0E1117",  # Fond général en noir (contours)
            margin=dict(t=40, b=40, l=0, r=0)  # Ajuste les marges pour éviter les bords blancs
        )

        # Affichage du box plot
        st.plotly_chart(fig_price, use_container_width=True)

    # Graphique 3 : Pie Chart - Répartition des types de logement
    with col2:
        # Comptage du nombre de logements par type
        room_counts = df_filtered['room_type'].value_counts().reset_index()
        room_counts.columns = ['room_type', 'count']  # Renommage des colonnes
        # Création d'un graphique en secteurs pour la répartition des types de logement
        fig_room = px.pie(
            room_counts, 
            names='room_type',  # Noms des types de logement
            values='count'  # Nombre de logements par type
        )

        # Mise à jour de la mise en page du graphique en secteurs
        fig_room.update_layout(
            title=dict(
                text="Répartition des types de logement",  # Titre du graphique
                x=0.5,  # Centrer horizontalement
                xanchor="center",  # Centrer sur l'axe X
                font=dict(color="white")  # Titre en blanc
            ),
            legend=dict(
                font=dict(color="white")  # Légende en blanc
            ),
            plot_bgcolor="#0E1117",  # Fond du graphique en noir
            paper_bgcolor="#0E1117",  # Fond général en noir (contours)
            margin=dict(t=40, b=40, l=0, r=0)  # Ajuste les marges pour éviter les bords blancs
        )

        # Affichage du graphique en secteurs
        st.plotly_chart(fig_room, use_container_width=True)

    # Séparateur horizontal pour diviser les sections
    st.markdown("---")

    # ----------------------------------------------------------------------------
    # Histogramme des prix moyens par arrondissement
    # ----------------------------------------------------------------------------
    # Calcul du prix médian par arrondissement
    prix_moyen_par_arr = df_filtered.groupby("neighbourhood")["price"].median().reset_index()

    # Tri des arrondissements par prix médian
    prix_moyen_par_arr = prix_moyen_par_arr.sort_values(by="price", ascending=True)  # Trier en ordre croissant

    # Création de l'histogramme des prix médians par arrondissement
    fig_hist = px.bar(
        prix_moyen_par_arr,
        x="neighbourhood",  # Axe des arrondissements
        y="price",  # Axe des prix médians
        color="price",  # Couleur en fonction du prix
        color_continuous_scale=["lightgreen", "lightblue", "#FF7F7F"],  # Dégradé de couleurs
        labels={"price": "Prix (€)", "neighbourhood": "Arrondissement"}  # Légendes des axes
    )

    # Mise à jour de la mise en page de l'histogramme
    fig_hist.update_layout(
        title=dict(
            text="Prix médian des logements par arrondissement",  # Titre du graphique
            x=0.5,  # Centrer horizontalement
            xanchor="center",  # Centrer sur l'axe X
            font=dict(color="white")  # Titre en blanc
        ),
        legend=dict(
            font=dict(color="white")  # Légende en blanc
        ),
        plot_bgcolor="#0E1117",  # Fond du graphique en noir
        paper_bgcolor="#0E1117",  # Fond général en noir (contours)
        margin=dict(t=40, b=40, l=0, r=0)  # Ajuste les marges pour éviter les bords blancs
    )

    # Affichage de l'histogramme
    st.plotly_chart(fig_hist, use_container_width=True)

    # Séparateur horizontal pour diviser les sections
    st.markdown("---")

    # Colonnes pour organiser les graphiques côte à côte
    col1, col2 = st.columns(2)

    # Graphique 1 : Moyenne des avis par type de logement
    with col1:
        # Création d'un graphique en barres pour le nombre médian d'avis par type de logement
        fig_reviews_type = px.bar(
            df_filtered.groupby("room_type")["number_of_reviews"].median().reset_index(), 
            x="room_type",  # Axe des types de logement
            y="number_of_reviews",  # Axe des avis
            color="room_type",  # Couleur par type de logement
            labels={"room_type": "Type de logement", "number_of_reviews": "Nombre médian d'avis"},  # Légendes
            title="Moyenne des avis par type de logement"  # Titre du graphique
        )

        # Mise à jour de la mise en page du graphique des avis
        fig_reviews_type.update_layout(
            title=dict(
                text="Nombre médian d'avis par type de logement",  # Titre du graphique
                x=0.5,  # Centrer horizontalement
                xanchor="center",  # Centrer sur l'axe X
                font=dict(color="white")  # Titre en blanc
            ),
            legend=dict(
                font=dict(color="white")  # Légende en blanc
            ),
            plot_bgcolor="#0E1117",  # Fond du graphique en noir
            paper_bgcolor="#0E1117",  # Fond général en noir (contours)
            margin=dict(t=40, b=40, l=0, r=0)  # Ajuste les marges pour éviter les bords blancs
        )

        # Affichage du graphique des avis
        st.plotly_chart(fig_reviews_type, use_container_width=True)

    # Graphique 2 : Histogramme de la distribution des prix par nuit
    with col2:
        # Regroupement des prix supérieurs à 20 000 dans une catégorie unique
        df_filtered['price_grouped'] = df_filtered['price'].apply(
            lambda x: x if x <= 20000 else 20000  # Regrouper les prix > 20000
        )
        
        # Définition des plages de prix pour l'histogramme
        bins = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1500, 20000]
        labels = ["0-100", "101-200", "201-300", "301-400", "401-500", "501-600", "601-700", "701-800", 
                "801-900", "901-1000", "1001-1200", "1201-1500", "1501-20000"]
        
        # Application des plages de prix
        df_filtered['price_range'] = pd.cut(df_filtered['price_grouped'], bins=bins, labels=labels, right=False)
        
        # Comptage du nombre de logements par plage de prix
        price_counts = df_filtered['price_range'].value_counts().sort_index()

        # Création de l'histogramme de la distribution des prix
        fig = go.Figure(data=[go.Bar(
            x=price_counts.index,  # Plages de prix
            y=price_counts.values,  # Nombre de logements par plage de prix
            marker=dict(color='rgb(100, 100, 255)'),  # Couleur des barres
        )])

        # Mise à jour de la mise en page de l'histogramme
        fig.update_layout(
            title=dict(
                text="Distribution des prix par nuit",  # Titre du graphique
                x=0.5,  # Centrer horizontalement
                xanchor="center",  # Centrer sur l'axe X
                font=dict(color="white")  # Titre en blanc
            ),
            xaxis=dict(
                title="Plage de prix",  # Titre de l'axe X
                tickangle=45,  # Angle des ticks
                tickfont=dict(color="white"),  # Couleur des ticks
                title_font=dict(color="white"),  # Couleur du titre de l'axe X
                showgrid=True,  # Afficher les grilles
                zeroline=False,  # Ne pas afficher la ligne zéro
                gridcolor="#444"  # Couleur des grilles
            ),
            yaxis=dict(
                title="Nombre de logements",  # Titre de l'axe Y
                tickfont=dict(color="white"),  # Couleur des ticks
                title_font=dict(color="white"),  # Couleur du titre de l'axe Y
                showgrid=True,  # Afficher les grilles
                zeroline=False,  # Ne pas afficher la ligne zéro
                gridcolor="#444"  # Couleur des grilles
            ),
            plot_bgcolor="#0E1117",  # Fond du graphique en noir
            paper_bgcolor="#0E1117",  # Fond général en noir (contours)
            legend=dict(
                font=dict(color="white"),  # Légende en blanc
                xanchor="center"  # Aligne la légende par rapport au centre
            ),
            margin=dict(t=40, b=80, l=40, r=40)  # Ajuste les marges pour plus d'espace
        )
        
        # Affichage de l'histogramme des prix
        st.plotly_chart(fig, use_container_width=True)

    # ----------------------------------------------------------------------------
    # Moyenne du prix par nombre de logements par arrondissement
    # ----------------------------------------------------------------------------

    # Calcul du prix moyen par arrondissement
    df_price_by_neighborhood = df_filtered.groupby("neighbourhood")["price"].mean().reset_index()

    # Calcul du nombre d'appartements par arrondissement
    df_apartments_by_neighborhood = df_filtered.groupby("neighbourhood")["id"].nunique().reset_index()

    # Fusionner les deux DataFrames pour obtenir prix moyen et nombre d'appartements
    df_neighborhoods = pd.merge(df_price_by_neighborhood, df_apartments_by_neighborhood, on="neighbourhood")
    df_neighborhoods.columns = ['neighbourhood', 'avg_price', 'num_apartments']

    # Créer un graphique combiné avec deux axes Y
    fig_combined = go.Figure()

    # Tracer le nombre d'appartements (en histogramme) sur l'axe Y principal
    fig_combined.add_trace(go.Bar(
        x=df_neighborhoods['neighbourhood'],  # Arrondissements
        y=df_neighborhoods['num_apartments'],  # Nombre d'appartements
        name="Nombre d'appartements",  # Nom de la légende
        marker=dict(color='rgb(100, 100, 255)'),  # Couleur des barres
        yaxis="y1"  # Utilisation de l'axe Y principal
    ))

    # Tracer le prix moyen (en ligne) sur l'axe Y secondaire
    fig_combined.add_trace(go.Scatter(
        x=df_neighborhoods['neighbourhood'],  # Arrondissements
        y=df_neighborhoods['avg_price'],  # Prix moyen
        name="Prix moyen",  # Nom de la légende
        mode='lines+markers',  # Lignes et points
        marker=dict(color='rgb(255, 100, 100)'),  # Couleur de la ligne
        yaxis="y2"  # Utilisation de l'axe Y secondaire
    ))

    # Configuration des axes
    fig_combined.update_layout(
        title=dict(
            text="Prix moyen et Nombre d'appartements par arrondissement",  # Titre du graphique
            x=0.5,  # Centrer horizontalement
            xanchor="center",  # Centrer sur l'axe X
            font=dict(color="white")  # Titre en blanc
        ),
        xaxis=dict(
            title="Arrondissements",  # Titre de l'axe X
            tickangle=45,  # Angle des ticks
            tickmode="array",  # Mode de gestion des ticks
            tickvals=df_neighborhoods['neighbourhood'],  # Valeurs des ticks
            ticktext=df_neighborhoods['neighbourhood'],  # Textes des ticks (arrondissements)
            showgrid=True,  # Afficher les grilles
            zeroline=False,  # Ne pas afficher la ligne zéro
            gridcolor="#444",  # Couleur des grilles
            tickfont=dict(color="white"),  # Couleur des ticks
            title_font=dict(color="white")  # Couleur du titre de l'axe X
        ),
        yaxis=dict(
            title="Nombre d'appartements",  # Titre de l'axe Y principal
            showgrid=True,  # Afficher les grilles
            zeroline=False,  # Ne pas afficher la ligne zéro
            gridcolor="#444",  # Couleur des grilles
            tickfont=dict(color="white"),  # Couleur des ticks
            title_font=dict(color="white")  # Couleur du titre de l'axe Y principal
        ),
        yaxis2=dict(
            title="Prix moyen",  # Titre de l'axe Y secondaire
            showgrid=True,  # Afficher les grilles
            zeroline=False,  # Ne pas afficher la ligne zéro
            gridcolor="#444",  # Couleur des grilles
            tickfont=dict(color="white"),  # Couleur des ticks
            title_font=dict(color="white"),  # Couleur du titre de l'axe Y secondaire
            overlaying="y",  # Superposition avec l'axe Y principal
            side="right"  # Placer l'axe secondaire à droite
        ),
        plot_bgcolor="#0E1117",  # Fond du graphique en noir
        paper_bgcolor="#0E1117",  # Fond général en noir (contours)
        legend=dict(
            font=dict(color="white"),  # Légende en blanc
            x=1.05,  # Décale la légende à droite
            y=0.5,  # Centrer verticalement la légende
            xanchor="left",  # Ancre à gauche de la légende
            yanchor="middle",  # Centrer la légende verticalement
            traceorder="normal",  # Ordre normal des traces
            bgcolor="rgba(0, 0, 0, 0)",  # Fond transparent pour la légende
            bordercolor="rgba(0, 0, 0, 0)",  # Pas de bordure autour de la légende
        ),
        margin=dict(t=40, b=80, l=40, r=40)  # Ajuste les marges pour éviter les chevauchements
    )

    # Affichage du graphique combiné
    st.plotly_chart(fig_combined, use_container_width=True)



elif current_page == "page2":
    st.title("Airbnb Paris Dashboard - Page 2")

    # Calcul du prix moyen par arrondissement
    prix_moyen_par_arr = df.groupby("neighbourhood")["price"].mean().reset_index()

    # Calcul du nombre de logements par arrondissement
    nombre_logements_par_arr = df["neighbourhood"].value_counts().reset_index()
    nombre_logements_par_arr.columns = ["neighbourhood", "count"]

    # Filtrage des logements de type "Appartement entier" et calcul du nombre par arrondissement
    part_logements_entiers_par_arr = df[df["room_type"] == "Appartement entier"].groupby("neighbourhood").size().reset_index(name='count')

    # Jointure des données pour inclure tous les quartiers et remplissage des valeurs manquantes par 0
    part_logements_entiers_par_arr = nombre_logements_par_arr.merge(
        part_logements_entiers_par_arr, on="neighbourhood", how="left"
    ).fillna(0)

    # Calcul de la part de logements entiers par rapport au total des logements dans chaque arrondissement
    part_logements_entiers_par_arr["part"] = part_logements_entiers_par_arr["count_y"] / part_logements_entiers_par_arr["count_x"]

    # Renommage des colonnes pour une meilleure lisibilité
    part_logements_entiers_par_arr = part_logements_entiers_par_arr.rename(columns={"count_x": "total_logements", "count_y": "logements_entiers"})

    # Sélection de la variable à afficher sur la carte
    variable = st.selectbox("Choisir la variable à afficher sur la carte", 
                            ["Prix moyen", "Nombre de logements", "Part de logements entiers"])

    # Sélection des données et des couleurs selon la variable choisie
    if variable == "Prix moyen":
        data = prix_moyen_par_arr
        color = "price"
        title = "Carte de chaleur des prix moyens par arrondissement"
    elif variable == "Nombre de logements":
        data = nombre_logements_par_arr
        color = "count"
        title = "Carte de chaleur du nombre de logements par arrondissement"
    else:
        data = part_logements_entiers_par_arr
        color = "part"
        title = "Carte de chaleur de la part de logements entiers par arrondissement"

    # Création de la carte de chaleur avec Plotly
    fig_heatmap = px.choropleth_mapbox(
        data,
        geojson=geojson_data,  # Données géographiques de Paris (geojson)
        locations="neighbourhood",  # Colonne contenant les arrondissements
        featureidkey="properties.neighbourhood",  # Clé des propriétés pour la correspondance
        color=color,  # Variable pour colorer la carte
        color_continuous_scale=["lightblue", "yellow", "red"],  # Palette de couleurs
        mapbox_style="carto-positron",  # Style de la carte
        zoom=10,  # Niveau de zoom sur Paris
        center={"lat": 48.8566, "lon": 2.3522},  # Coordonnées géographiques du centre de Paris
        opacity=0.6,  # Opacité de la carte
        title=title  # Titre de la carte
    )

    # Affichage de la carte de chaleur
    st.plotly_chart(fig_heatmap, use_container_width=True)
elif current_page == "page3":
    st.title("Airbnb Paris Dashboard - Page 3")
    st.markdown("### Parts de type de logement par arrondissement")

    # Filtre global pour sélectionner les types de logement disponibles
    room_types = df['room_type'].dropna().unique().tolist()  # Récupère les types de logement uniques sans valeurs manquantes
    selected_room_types = st.multiselect("Sélectionnez les types de logement à afficher", options=room_types, default=room_types)  # Permet à l'utilisateur de sélectionner plusieurs types de logement

    # Filtrage du DataFrame en fonction des types de logement sélectionnés
    df_filtered = df[df['room_type'].isin(selected_room_types)]

    # Calcul des parts de chaque type de logement par arrondissement
    parts_par_arr = df_filtered.groupby(["neighbourhood", "room_type"]).size().reset_index(name='count')  # Nombre de logements par type et par arrondissement
    total_par_arr = df_filtered.groupby("neighbourhood").size().reset_index(name='total')  # Total des logements par arrondissement
    parts_par_arr = parts_par_arr.merge(total_par_arr, on="neighbourhood")  # Jointure pour obtenir le total des logements par arrondissement
    parts_par_arr["part"] = parts_par_arr["count"] / parts_par_arr["total"]  # Calcul de la part de chaque type de logement par arrondissement

    # Création des small multiples (barres empilées pour chaque arrondissement)
    fig_small_multiples = px.bar(
        parts_par_arr,
        x="room_type",  # Axe des x : types de logement
        y="part",  # Axe des y : répartition des parts
        barmode="stack",  # Empilement des barres
        facet_col="neighbourhood",  # Facet par arrondissement
        facet_col_wrap=4,  # Nombre de colonnes d'arrondissements par ligne
        title="Parts de type de logement par arrondissement",  # Titre du graphique
        labels={"room_type": "Type de logement", "part": "Répartition", "neighbourhood": "Arrondissement" },  # Libellés des axes
        height=1000,  # Hauteur de l'intrigue
        text_auto=".0%",  # Format d'affichage du texte sur les barres (en pourcentage)
    )

    # Personnalisation de l'apparence des graphiques
    fig_small_multiples.update_traces(marker_color="#f94941")  # Couleur des barres
    fig_small_multiples.for_each_annotation(lambda a: a.update(text=a.text.split('=')[-1].strip()))  # Formatage des annotations

    # Mise à jour de la disposition du graphique
    fig_small_multiples.update_layout(
        showlegend=False,  # Masquer la légende
        margin=dict(t=50, l=50, r=50, b=50),  # Ajustement des marges du graphique
    )

    # Personnalisation des axes y (format des ticks et plage de valeurs)
    fig_small_multiples.for_each_yaxis(
        lambda yaxis: yaxis.update(tickformat=".0%", side="left", range=[0, 1])
    )

    # Affichage du graphique
    st.plotly_chart(fig_small_multiples, use_container_width=True)

elif current_page == "page4":
    st.title("Airbnb Paris Dashboard - Page 4")
    st.markdown("### Visualisation des Airbnb autour des monuments")

    # Sélection du monument parmi ceux disponibles
    monument_names = monuments['name'].tolist()  # Liste des noms des monuments
    selected_monument = st.selectbox("Choisir un monument", options=monument_names)  # Sélection du monument

    # Sélection de la distance autour du monument
    distance = st.slider("Choisir la distance (m)", min_value=100, max_value=2000, value=200, step=100)  # Choix de la distance via un slider

    # Récupération des coordonnées du monument sélectionné
    monument_coords = monuments[monuments['name'] == selected_monument][['latitude', 'longitude']].values[0]
    mon_lat, mon_lon = monument_coords  # Latitude et longitude du monument sélectionné

    # Approximation des degrés de latitude et longitude pour créer une bounding box autour du monument
    delta_lat = distance / 1000 / 111  # 1° de latitude ≈ 111 km
    delta_lon = distance / 1000 / (111 * abs(np.cos(np.radians(mon_lat))))  # Ajustement pour la longitude en fonction de la latitude

    # Définition de la bounding box (plage de latitudes et longitudes autour du monument)
    lat_min, lat_max = mon_lat - delta_lat, mon_lat + delta_lat
    lon_min, lon_max = mon_lon - delta_lon, mon_lon + delta_lon

    # Filtrage initial des logements dans la bounding box (avant de calculer la distance exacte)
    df_filtered = df[
        (df['latitude'].between(lat_min, lat_max)) &
        (df['longitude'].between(lon_min, lon_max))
    ].copy()

    # Calcul exact de la distance en kilomètres pour chaque logement à partir des coordonnées du monument
    df_filtered['distance'] = df_filtered.apply(
        lambda row: geodesic((row['latitude'], row['longitude']), (mon_lat, mon_lon)).km, axis=1
    )

    # Filtrage final des logements en fonction de la distance sélectionnée
    df_filtered = df_filtered[df_filtered['distance'] <= distance]

    # Création du lien vers l'annonce Airbnb pour chaque logement
    df_filtered["Lien Airbnb"] = df_filtered["id"].apply(lambda x: f'<a href="https://www.airbnb.fr/rooms/{x}" target="_blank">Voir l’annonce</a>')

    # Création de la carte des Airbnb autour du monument sélectionné
    fig_map = px.scatter_mapbox(
        df_filtered,
        lat="latitude",  # Coordonnée latitude des logements
        lon="longitude",  # Coordonnée longitude des logements
        hover_name="name",  # Affichage du nom du logement au survol
        color="room_type",  # Couleur des points en fonction du type de logement
        zoom=15 - (distance / 1000),  # Niveau de zoom sur la carte, ajusté en fonction de la distance
        height=600,  # Hauteur de la carte
        custom_data=["Lien Airbnb"]  # Lien vers l'annonce Airbnb à afficher au survol
    )

    # Personnalisation de l'apparence de la carte
    fig_map.update_layout(
        plot_bgcolor="#0E1117",  # Fond du graphique
        paper_bgcolor="#0E1117",  # Fond du papier
        title=dict(
            text=f"Airbnb autour de {selected_monument} dans un rayon de {distance} m",  # Titre avec le monument et la distance
            x=0.5,  # Centrer le titre horizontalement
            xanchor="center",  # Centrer sur l'axe X
            font=dict(color="white")  # Titre en blanc
        ),
        mapbox_style="carto-positron",  # Style de la carte
        margin={"r":0, "t":40, "l":0, "b":0},  # Marges de la carte
        legend=dict(
            title="Type de logement",  # Titre de la légende
            title_font=dict(color="white"),  # Couleur du titre de la légende
            font=dict(color="white"),  # Couleur du texte de la légende
            bgcolor="#0E1117",  # Couleur de fond de la légende
            bordercolor="white",  # Couleur de la bordure de la légende
            borderwidth=1  # Épaisseur de la bordure de la légende
        )
    )

    # Ajout d'un lien vers l'annonce Airbnb au survol des points de la carte
    for i, d in enumerate(fig_map.data):
        d.hovertemplate += "<br>%{customdata[0]}"  # Affichage du lien au survol

    # Affichage de la carte avec les Airbnb autour du monument
    st.plotly_chart(fig_map, use_container_width=True)

    # Initialisation d'une variable de session pour suivre le nombre de liens affichés
    if "num_links" not in st.session_state:
        st.session_state.num_links = 5  # Initialiser à 5 annonces affichées par défaut

    # Affichage des annonces Airbnb sous forme de conteneurs
    st.markdown("### Aperçu des annonces Airbnb")
    for i, row in df_filtered.head(st.session_state.num_links).iterrows():
        st.markdown(
            f"""
            <div style="background-color: #1E222A; padding: 10px; margin-bottom: 20px; border-radius: 5px;">
                <h4 style="color: #FF5D57; margin: 0;">{row['name']}</h4>  <!-- Affichage du nom de l'annonce -->
                <p style="color: white; margin: 5px 0;">Type: {row['room_type']} | Prix: {row['price']}€</p>  <!-- Affichage du type et du prix -->
                <a href="https://www.airbnb.fr/rooms/{row['id']}" target="_blank" style="color: white; text-decoration: none; background-color: #ff5d57; padding: 10px 20px; border-radius: 5px; display: inline-block; font-weight: bold;">
                    Voir l'annonce
                </a>
            </div>
            """,
            unsafe_allow_html=True  # Permet d'inclure du HTML dans le markdown
        )

    # Ajout de styles CSS personnalisés pour le bouton
    st.markdown(
        """
        <style>
        .stButton > button {
            background-color: #FF5D57; /* Couleur de fond du bouton */
            color: white; /* Couleur du texte du bouton */
            font-weight: bold;
            padding: 10px 20px;
            border-radius: 5px;
            border: none;
            cursor: pointer;
        }
        .stButton > button:hover {
            background-color: #E04B45; /* Couleur du bouton au survol */
        }
        </style>
        """,
        unsafe_allow_html=True  # Permet d'inclure du CSS dans le markdown
    )

    # Bouton natif Streamlit pour afficher 5 annonces supplémentaires
    if st.button("Voir plus"):
        st.session_state.num_links += 5  # Augmente de 5 le nombre d'annonces affichées à chaque clic
