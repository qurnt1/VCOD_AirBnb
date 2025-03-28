import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import warnings
from geopy.distance import geodesic
import numpy as np
import warnings

st.set_page_config(page_title="Airbnb Paris Dashboard", layout="wide")

st.set_option('client.showErrorDetails', False)



# Style CSS personnalisé
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
        /* 🔼 Flèche d'ouverture/fermeture en blanc */
        [data-testid="stSidebarNavIcon"] svg {
            fill: white !important;
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
            background-color: #0E1117;
            padding: 10px;
            text-align: center;
            margin-bottom: 20px;
        }
        .nav-links a {
            margin: 0 15px;
            text-decoration: none;
            font-weight: bold;
            color: white !important;
            background-color: #ff5d57;
            padding: 10px 20px;
            border-radius: 5px;
            display: inline-block;
        }
        .nav-links a:hover {
            background-color: #e04b45;
        }
    
        /* 🔘 Boutons Streamlit */
        .st-emotion-cache-1oy2xl0 {
            background-color: #238636 !important;
            color: white !important;
            font-weight: bold;
            padding: 10px 20px;
            border-radius: 5px;
        }
        .st-emotion-cache-1oy2xl0:hover {
            background-color: #2EA043 !important;
        }
    </style>

    """, unsafe_allow_html=True
)

# Détecter le paramètre de l'URL pour déterminer la page active
query_params = st.experimental_get_query_params()

current_page = query_params.get("page", ["visualisation"])[0]

# Si le paramètre "page" est absent ou mal formé, définir "visualisation" comme page par défaut
if current_page not in ["page1", "page2", "page3", "page4"]:
    current_page = "page1"

# Affichage des liens de navigation
st.markdown("""
<div class="nav-links">
    <a href="?page=page1" target="_self">Tableau de Bord</a>
    <a href="?page=page2" target="_self">Carte Choroplèthe</a>
    <a href="?page=page3" target="_self">Graphique Small Multiples</a>
    <a href="?page=page4" target="_self">Distance Monuments</a>
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# Chargement des données depuis listings.csv (avec mise en cache)
# ------------------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("listings.csv")
    if 'last_review' in df.columns:
        df['last_review'] = pd.to_datetime(df['last_review'], errors='coerce')
    df['room_type'] = df['room_type'].replace({
        'Entire home/apt': 'Appartement entier',
        'Hotel room': 'Chambre d\'hôtel',
        'Private room': 'Chambre privée',
        'Shared room': 'Chambre partagée'
    })
    return df

df = load_data()

# ------------------------------------------------------------------------------
# Chargement du fichier geojson pour les contours des arrondissements
# ------------------------------------------------------------------------------
@st.cache_data
def load_geojson():
    with open("neighbourhoods.geojson", "r", encoding="utf-8") as f:
        geojson_data = json.load(f)
    return geojson_data

geojson_data = load_geojson()

# ------------------------------------------------------------------------------
# Chargement des données des monuments depuis l'URL
# ------------------------------------------------------------------------------
@st.cache_data
def load_monuments():
    url = "monuments.csv"
    monuments = pd.read_csv(url, delimiter=',', encoding='latin1')
    
    # Limites géographiques de Paris
    lat_min, lat_max = 48.815573, 48.902144
    lon_min, lon_max = 2.224199, 2.469921

    # Filtrer les monuments situés à Paris
    monuments = monuments[(monuments['latitude'] >= lat_min) & (monuments['latitude'] <= lat_max) &
                          (monuments['longitude'] >= lon_min) & (monuments['longitude'] <= lon_max)]
    return monuments

monuments = load_monuments()

# Contenu des pages
if current_page == "page1":

    # Affichage du logo et du titre
    st.image("logo.png", width=150)  
    st.title("Airbnb Paris Dashboard")
    st.markdown("### Analyse interactive des données Airbnb à Paris")
    
    # -------------------------------------------------------------------------------
    # Sidebar pour afficher les filtres
    # -------------------------------------------------------------------------------
    st.sidebar.header("Filtres")

    # Filtre sur le type de logement
    room_types = df['room_type'].dropna().unique().tolist()
    selected_room_types = st.sidebar.multiselect("Type de logement", options=room_types, default=room_types)
    
    # Filtre sur le prix (slider)
    min_price = int(df['price'].min())
    max_price = int(df['price'].max())
    price_range = st.sidebar.slider("Prix (€)", min_value=min_price, max_value=max_price, 
                                    value=(min_price, max_price), step=5)
    
    # Filtre sur le nombre d'avis (slider)
    min_reviews = int(df['number_of_reviews'].min())
    max_reviews = int(df['number_of_reviews'].max())
    reviews_range = st.sidebar.slider("Nombre d'avis", min_value=min_reviews, max_value=max_reviews, 
                                    value=(min_reviews, max_reviews), step=1)
    
    # Sélection des arrondissements à afficher
    arrondissement_names = [feature["properties"]["neighbourhood"] for feature in geojson_data["features"]]
    selected_arrondissements = st.sidebar.multiselect("Arrondissements", options=arrondissement_names, default=arrondissement_names)

    # Sauvegarde des filtres dans un dictionnaire
    filters = {
        "selected_room_types": selected_room_types,
        "price_range": price_range,
        "reviews_range": reviews_range,
        "selected_arrondissements": selected_arrondissements
    }

    # Application des filtres sur le DataFrame
    df_filtered = df[df['room_type'].isin(filters["selected_room_types"]) &
                    df['price'].between(filters["price_range"][0], filters["price_range"][1]) &
                    df['number_of_reviews'].between(filters["reviews_range"][0], filters["reviews_range"][1])]

    # Filtrage des logements selon les arrondissements sélectionnés
    if filters["selected_arrondissements"]:
        df_filtered = df_filtered[df_filtered["neighbourhood"].isin(filters["selected_arrondissements"])]

    # -------------------------------------------------------------------------------
    # Affichage des indicateurs clés
    # -------------------------------------------------------------------------------
    st.markdown(f"### Chiffres clés sur les {df_filtered.shape[0]} logements filtrés")

    col1, col2, col3 = st.columns(3)
    col1.metric("Prix moyen (€)", f"{df_filtered['price'].mean():.2f}")
    col2.metric("Médiane du prix (€)", f"{df_filtered['price'].median():.2f}")
    col3.metric("Nombre moyen d'avis par logement", f"{df_filtered['number_of_reviews'].mean():.1f}")

    col4, col5, col6 = st.columns(3)
    col4.metric("Disponibilité moyenne (jours/an)", f"{df_filtered['availability_365'].mean():.1f}")

    quartier_top = df_filtered['neighbourhood'].mode()[0] if not df_filtered['neighbourhood'].isna().all() else "N/A"
    col5.metric("Quartier le plus populaire", quartier_top)

    if "calculated_host_listings_count" in df_filtered.columns:
        taux_reponse_moyen = df_filtered["calculated_host_listings_count"].mean()
        col6.metric("Taux de réponse moyen (en %)", f"{taux_reponse_moyen:.1f}" if not df_filtered["calculated_host_listings_count"].isna().all() else "N/A")

    col7, _ , _ = st.columns(3)
    percent_entire_home = (df_filtered["room_type"] == 'Appartement entier').sum() / df_filtered.shape[0] * 100
    col7.metric("Logements entiers (%)", f"{percent_entire_home:.1f}%")

    st.markdown("---")

    # Graphique 1 : Carte interactive des logements à Paris
    fig_map = px.scatter_map(
        df_filtered,
        lat="latitude",
        lon="longitude",
        hover_name="name",
        hover_data=["price", "room_type"],
        color="room_type",
        zoom=11,
        height=600,
        title="Localisation des logements à Paris",
        labels={"room_type": "Type de logement"}
    )

    fig_map.update_layout(
        title=dict(
            text="Localisation des logements à Paris",
            x=0.5,  # Centrer horizontalement
            xanchor="center",  # Centrer sur l'axe X
            font=dict(color="white")  # Titre en blanc
        ),
        legend=dict(
            font=dict(color="white")  # Légende en blanc
        ),
        mapbox=dict(
            style="satellite",  # Utilisation du style satellite de Mapbox
            center={"lat": df_filtered["latitude"].mean(), "lon": df_filtered["longitude"].mean()},
            zoom=11
        ),
        plot_bgcolor="#0E1117",  # Fond du graphique en noir
        paper_bgcolor="#0E1117",  # Fond général en noir (contours)
        margin=dict(t=40, b=40, l=0, r=0)  # Ajuste les marges pour éviter les bords blancs
    )

    st.plotly_chart(fig_map, use_container_width=True)
    
    st.markdown("---")
    
    # Colonnes pour organiser les graphiques côte à côte
    col1, col2 = st.columns(2)

    # Graphique 2 : Box Plot - Distribution des prix par type de logement
    with col1:
        fig_price = px.box(
            df_filtered, 
            x="room_type", 
            y="price", 
            color="room_type",
            labels={"room_type": "Type de logement", "price": "Prix (€)"}
        )

        fig_price.update_layout(
            title=dict(
                text="Distribution des prix par type de logement",                
                x=0.5,  # Centrer horizontalement
                xanchor="center",  # Centrer sur l'axe X
                font=dict(color="white")  # Titre en blanc
            ),
            legend=dict(
            font=dict(color="white")
            ),
            plot_bgcolor="#0E1117",  # Fond du graphique en noir
            paper_bgcolor="#0E1117",  # Fond général en noir (contours)
            margin=dict(t=40, b=40, l=0, r=0)  # Ajuste les marges pour éviter les bords blancs
        )

        st.plotly_chart(fig_price, use_container_width=True)

    # Graphique 3 : Pie Chart - Répartition des types de logement
    with col2:
        room_counts = df_filtered['room_type'].value_counts().reset_index()
        room_counts.columns = ['room_type', 'count']
        fig_room = px.pie(
            room_counts, 
            names='room_type', 
            values='count'
        )

        fig_room.update_layout(
            title=dict(
                text="Répartition des types de logement",                
                x=0.5,  # Centrer horizontalement
                xanchor="center",  # Centrer sur l'axe X
                font=dict(color="white")  # Titre en blanc
            ),
            legend=dict(
            font=dict(color="white")
            ),
            plot_bgcolor="#0E1117",  # Fond du graphique en noir
            paper_bgcolor="#0E1117",  # Fond général en noir (contours)
            margin=dict(t=40, b=40, l=0, r=0)  # Ajuste les marges pour éviter les bords blancs
        )

        st.plotly_chart(fig_room, use_container_width=True)
        
    st.markdown("---")
    # ----------------------------------------------------------------------------
    # Histogramme des prix moyens par arrondissement
    # ----------------------------------------------------------------------------
    prix_moyen_par_arr = df_filtered.groupby("neighbourhood")["price"].median().reset_index()

    prix_moyen_par_arr = prix_moyen_par_arr.sort_values(by="price", ascending=True)  # Trier en décroissant
    fig_hist = px.bar(
        prix_moyen_par_arr,
        x="neighbourhood",
        y="price",
        color="price",
        color_continuous_scale=["lightgreen", "lightblue", "#FF7F7F"],
        labels={"price": "Prix (€)", "neighbourhood": "Arrondissement"}
    )

    fig_hist.update_layout(
            title=dict(
                text="Prix médian des logements par arrondissement",                
                x=0.5,  # Centrer horizontalement
                xanchor="center",  # Centrer sur l'axe X
                font=dict(color="white")  # Titre en blanc
            ),
            legend=dict(
            font=dict(color="white")
            ),
            plot_bgcolor="#0E1117",  # Fond du graphique en noir
            paper_bgcolor="#0E1117",  # Fond général en noir (contours)
            margin=dict(t=40, b=40, l=0, r=0)  # Ajuste les marges pour éviter les bords blancs
        )
    st.plotly_chart(fig_hist, use_container_width=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Moyenne des avis par type de logement
        fig_reviews_type = px.bar(
            df_filtered.groupby("room_type")["number_of_reviews"].median().reset_index(), 
            x="room_type", 
            y="number_of_reviews", 
            color="room_type",
            labels={"room_type": "Type de logement", "number_of_reviews": "Nombre médian d'avis"},
            title="Moyenne des avis par type de logement"
        )
    
        fig_reviews_type.update_layout(
                title=dict(
                    text="Nombre médian d'avis par type de logement",                
                    x=0.5,  # Centrer horizontalement
                    xanchor="center",  # Centrer sur l'axe X
                    font=dict(color="white")  # Titre en blanc
                ),
                legend=dict(
                font=dict(color="white")
                ),
                plot_bgcolor="#0E1117",  # Fond du graphique en noir
                paper_bgcolor="#0E1117",  # Fond général en noir (contours)
                margin=dict(t=40, b=40, l=0, r=0)  # Ajuste les marges pour éviter les bords blancs
            )
        st.plotly_chart(fig_reviews_type, use_container_width=True)

    # Graphique 2 : Box Plot - Distribution des prix par type de logement
    with col2:
        # Filtrer les prix supérieurs à 20 000 et les regrouper dans une catégorie unique
        df_filtered['price_grouped'] = df_filtered['price'].apply(
            lambda x: x if x <= 20000 else 20000  # Regrouper tout ce qui est au-dessus de 20000 dans une seule catégorie
        )
        
        # Créer des plages de prix cohérentes
        bins = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1500, 20000]
        labels = ["0-100", "101-200", "201-300", "301-400", "401-500", "501-600", "601-700", "701-800", 
                  "801-900", "901-1000", "1001-1200", "1201-1500", "1501-20000"]
        
        # Ajouter une nouvelle colonne de plages de prix
        df_filtered['price_range'] = pd.cut(df_filtered['price_grouped'], bins=bins, labels=labels, right=False)
        
        # Compter le nombre d'occurrences dans chaque plage de prix
        price_counts = df_filtered['price_range'].value_counts().sort_index()
        
        # Créer l'histogramme
        fig = go.Figure(data=[go.Bar(
            x=price_counts.index,
            y=price_counts.values,
            marker=dict(color='rgb(100, 100, 255)'),
        )])
        
        # Ajouter les configurations du graphique
        fig.update_layout(
            title="Distribution des prix par nuit",
            xaxis=dict(
                title="Plage de prix",
                tickangle=45,
                tickfont=dict(color="white"),
                title_font=dict(color="white"),
                showgrid=True,
                zeroline=False,
                gridcolor="#444"
            ),
            yaxis=dict(
                title="Nombre d'appartements",
                tickfont=dict(color="white"),
                title_font=dict(color="white"),
                showgrid=True,
                zeroline=False,
                gridcolor="#444"
            ),
            plot_bgcolor="#0E1117",  # Fond du graphique en noir
            paper_bgcolor="#0E1117",  # Fond général en noir (contours)
            legend=dict(
                font=dict(color="white"),
                xanchor="center"  # Aligne la légende par rapport au centre

            ),
            margin=dict(t=40, b=80, l=40, r=40)  # Ajuste les marges
        )
        
            # Affichage du graphique avec Streamlit
        st.plotly_chart(fig, use_container_width=True)
    
    # Moyenne du prix par nombre de logements par arrondissement
    # Calculer le prix moyen et le nombre d'appartements par arrondissement
    df_price_by_neighborhood = df_filtered.groupby("neighbourhood")["price"].mean().reset_index()
    df_apartments_by_neighborhood = df_filtered.groupby("neighbourhood")["id"].nunique().reset_index()
    
    # Fusionner les deux DataFrames pour avoir à la fois le prix moyen et le nombre d'appartements
    df_neighborhoods = pd.merge(df_price_by_neighborhood, df_apartments_by_neighborhood, on="neighbourhood")
    df_neighborhoods.columns = ['neighbourhood', 'avg_price', 'num_apartments']
    
    # Créer un graphique combiné avec deux axes Y
    fig_combined = go.Figure()
    
    # Tracer le nombre d'appartements (en histogramme)
    fig_combined.add_trace(go.Bar(
        x=df_neighborhoods['neighbourhood'],
        y=df_neighborhoods['num_apartments'],
        name="Nombre d'appartements",
        marker=dict(color='rgb(100, 100, 255)'),
        yaxis="y1"
    ))
    
    # Tracer le prix moyen (en ligne)
    fig_combined.add_trace(go.Scatter(
        x=df_neighborhoods['neighbourhood'],
        y=df_neighborhoods['avg_price'],
        name="Prix moyen",
        mode='lines+markers',
        marker=dict(color='rgb(255, 100, 100)'),
        yaxis="y2"
    ))
    
    # Ajouter les configurations des axes
    
    fig_combined.update_layout(
        title="Prix moyen et Nombre d'appartements par arrondissement",
        title_x=0.5,
        xaxis=dict(
            title="Arrondissements",
            tickangle=45,
            tickmode="array",
            tickvals=df_neighborhoods['neighbourhood'],
            ticktext=df_neighborhoods['neighbourhood'],
            showgrid=True,
            zeroline=False,
            gridcolor="#444",
            tickfont=dict(color="white"),
            title_font=dict(color="white")
        ),
        yaxis=dict(
            title="Nombre d'appartements",
            showgrid=True,
            zeroline=False,
            gridcolor="#444",
            tickfont=dict(color="white"),
            title_font=dict(color="white")
        ),
        yaxis2=dict(
            title="Prix moyen",
            showgrid=True,
            zeroline=False,
            gridcolor="#444",
            tickfont=dict(color="white"),
            title_font=dict(color="white"),
            overlaying="y",
            side="right"
        ),
        plot_bgcolor="#0E1117",  # Fond du graphique en noir
        paper_bgcolor="#0E1117",  # Fond général en noir (contours)
        legend=dict(
            font=dict(color="white"),
            x=0.5,  # Centrer horizontalement
            xanchor="center",  # Centrer par rapport à l'axe X
            y=1.15,  # Positionner la légende un peu au-dessus du graphique pour éviter le chevauchement
            yanchor="bottom"  # L'ancrage doit être en bas pour ne pas recouvrir le graphique
        ),
        margin=dict(t=40, b=80, l=40, r=40)  # Ajuste les marges
    )
    st.plotly_chart(fig_combined, use_container_width=True)


elif current_page == "page2":
    st.title("Airbnb Paris Dashboard - Page 2")

    # Calcul des métriques par arrondissement
    prix_moyen_par_arr = df.groupby("neighbourhood")["price"].mean().reset_index()
    nombre_logements_par_arr = df["neighbourhood"].value_counts().reset_index()
    nombre_logements_par_arr.columns = ["neighbourhood", "count"]
    # Remplacez "Entire home/apt" par "Appartement entier"
    part_logements_entiers_par_arr = df[df["room_type"] == "Appartement entier"].groupby("neighbourhood").size().reset_index(name='count')
    # Jointure pour inclure tous les quartiers, remplissage des NaN par 0
    part_logements_entiers_par_arr = nombre_logements_par_arr.merge(
        part_logements_entiers_par_arr, on="neighbourhood", how="left"
    ).fillna(0)

    # Calcul de la part
    part_logements_entiers_par_arr["part"] = part_logements_entiers_par_arr["count_y"] / part_logements_entiers_par_arr["count_x"]

    # Renommage des colonnes
    part_logements_entiers_par_arr = part_logements_entiers_par_arr.rename(columns={"count_x": "total_logements", "count_y": "logements_entiers"})

    # Sélection de la variable à afficher
    variable = st.selectbox("Choisir la variable à afficher sur la carte", 
                            ["Prix moyen", "Nombre de logements", "Part de logements entiers"])

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

    # Carte de chaleur
    fig_heatmap = px.choropleth_mapbox(
        data,
        geojson=geojson_data,
        locations="neighbourhood",
        featureidkey="properties.neighbourhood",
        color=color,
        color_continuous_scale=["lightblue", "yellow", "red"],
        mapbox_style="carto-positron",
        zoom=10,
        center={"lat": 48.8566, "lon": 2.3522},
        opacity=0.6,
        title=title
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)
elif current_page == "page3":
    st.title("Airbnb Paris Dashboard - Page 3")
    st.markdown("### Parts de type de logement par arrondissement")

    # Calcul des parts de type de logement par arrondissement
    parts_par_arr = df.groupby(["neighbourhood", "room_type"]).size().reset_index(name='count')
    total_par_arr = df.groupby("neighbourhood").size().reset_index(name='total')
    parts_par_arr = parts_par_arr.merge(total_par_arr, on="neighbourhood")
    parts_par_arr["part"] = parts_par_arr["count"] / parts_par_arr["total"]

    # Création des small multiples
    fig_small_multiples = px.bar(
        parts_par_arr,
        x="room_type",
        y="part",
        barmode="stack",
        facet_col="neighbourhood",
        facet_col_wrap=4,
        title="Parts de type de logement par arrondissement",
        labels={"room_type": "Type de logement", "part": "Part"},
        height=1000
    )
    fig_small_multiples.update_layout(showlegend=False)
    st.plotly_chart(fig_small_multiples, use_container_width=True)
elif current_page == "page4":
    st.title("Airbnb Paris Dashboard - Page 4")
    st.markdown("### Visualisation des Airbnb autour des monuments")

    # Sélection du monument
    monument_names = monuments['name'].tolist()
    selected_monument = st.selectbox("Choisir un monument", options=monument_names)

    # Sélection de la distance
    distance = st.slider("Choisir la distance (m)", min_value=100, max_value=2000, value=200, step=100)

   # Récupération des coordonnées du monument sélectionné
    monument_coords = monuments[monuments['name'] == selected_monument][['latitude', 'longitude']].values[0]
    mon_lat, mon_lon = monument_coords

    # Approximation : 1° de latitude ≈ 111 km, la longitude dépend de la latitude
    delta_lat = distance/1000 / 111
    delta_lon = distance/1000 / (111 * abs(np.cos(np.radians(mon_lat))))

    # Définition de la bounding box
    lat_min, lat_max = mon_lat - delta_lat, mon_lat + delta_lat
    lon_min, lon_max = mon_lon - delta_lon, mon_lon + delta_lon

    # Filtrage rapide avant le calcul exact de distance
    df_filtered = df[
        (df['latitude'].between(lat_min, lat_max)) &
        (df['longitude'].between(lon_min, lon_max))
    ].copy()

    # Calcul exact de la distance sur le sous-ensemble
    df_filtered['distance'] = df_filtered.apply(
        lambda row: geodesic((row['latitude'], row['longitude']), (mon_lat, mon_lon)).km, axis=1
    )

    # Filtrage final sur la distance
    df_filtered = df_filtered[df_filtered['distance'] <= distance]

    df_filtered["Lien Airbnb"] = df_filtered["id"].apply(lambda x: f'<a href="https://www.airbnb.fr/rooms/{x}" target="_blank">Voir l’annonce</a>')

    # Affichage de la carte des Airbnb autour du monument
    fig_map = px.scatter_mapbox(
            df_filtered,
            lat="latitude",
            lon="longitude",
            hover_name="name",
            color="room_type",
            zoom= 15 - (distance / 1000),
            height=600,
            #title=f"Airbnb autour de {selected_monument} dans un rayon de {distance} m",
            custom_data=["Lien Airbnb"]
        )
    fig_map.update_layout(
            plot_bgcolor="#0E1117",
            paper_bgcolor="#0E1117",
            title=dict(
                text=f"Airbnb autour de {selected_monument} dans un rayon de {distance} m",
                x=0.5,  # Centrer horizontalement
                xanchor="center",  # Centrer sur l'axe X
                font=dict(color="white")  # Titre en blanc
            ),
            mapbox_style="carto-positron",
            margin={"r":0, "t":40, "l":0, "b":0},
            legend=dict(
            title="Type de logement",  # Titre de la légende
            title_font=dict(color="white"),  # Couleur du titre de la légende
            font=dict(color="white"),  # Couleur du texte de la légende
            bgcolor="#0E1117",         # Couleur de fond de la légende
            bordercolor="white",       # Couleur de la bordure
            borderwidth=1              # Épaisseur de la bordure
            )
            
            )
    for i, d in enumerate(fig_map.data):
        d.hovertemplate += "<br>%{customdata[0]}"

    st.plotly_chart(fig_map, use_container_width=True)
    # Initialisation d'une variable de session pour suivre le nombre de liens affichés
    if "num_links" not in st.session_state:
        st.session_state.num_links = 5  # Afficher 5 liens au départ

    # Affichage des annonces Airbnb sous forme de conteneurs
    st.markdown("### Aperçu des annonces Airbnb")
    for i, row in df_filtered.head(st.session_state.num_links).iterrows():
        st.markdown(
            f"""
            <div style="background-color: #1E222A; padding: 10px; margin-bottom: 20px; border-radius: 5px;">
                <h4 style="color: #FF5D57; margin: 0;">{row['name']}</h4>
                <p style="color: white; margin: 5px 0;">Type: {row['room_type']} | Prix: {row['price']}€</p>
                <a href="https://www.airbnb.fr/rooms/{row['id']}" target="_blank" style="color: white; text-decoration: none; background-color: #ff5d57; padding: 10px 20px; border-radius: 5px; display: inline-block; font-weight: bold;">
                    Voir l'annonce
                </a>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Ajout de styles CSS personnalisés pour le bouton
    st.markdown(
        """
        <style>
        .stButton > button {
            background-color: #FF5D57; /* Couleur de fond */
            color: white; /* Couleur du texte */
            font-weight: bold;
            padding: 10px 20px;
            border-radius: 5px;
            border: none;
            cursor: pointer;
        }
        .stButton > button:hover {
            background-color: #E04B45; /* Couleur au survol */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Bouton natif Streamlit pour afficher 5 annonces supplémentaires
    if st.button("Voir plus"):
        st.session_state.num_links += 5  # Augmenter le nombre d'annonces affichées de 5    
