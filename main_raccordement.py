# -*- coding: utf-8 -*-
"""
Created on Tue May 20 13:33:07 2025

@author: Skysun
"""

import pandas as pd
import matplotlib.pyplot as plt
import math
import contextily as ctx
import streamlit as st
import folium
from streamlit_folium import st_folium





"""             Début du code             """

# Copier coller les coordonnées ici
Coord = [47.475806275508376, -0.5601572398707153]

lat = Coord[0]
lon = Coord[1]

# Étape 1 : Lecture du fichier Excel
fichier_excel = r"G:\Drive partagés\Skysun France\03 - Commercial\01 - Projets en cours de vente\Raccordement_Capacites_DAccueil.xlsx"
df = pd.read_excel(fichier_excel)


# Étape 2 : Extraction des coordonnées X/Y (colonnes E et F), capacité immédiatement disponible sans travaux (colonne AB), capacité en travaux (colonne I)
nom_poste = df.iloc[:, 1].tolist()  # Colonne B
coord_x = df.iloc[:, 4].tolist()  # Colonne E (index 4 car on compte à partir de 0)
coord_y = df.iloc[:, 5].tolist()  # Colonne F (index 5)
dispo = df.iloc[:, 27].tolist()  # Colonne AB
travaux = df.iloc[:, 8].tolist()  # Colonne I


# On ajoute les coordonnées dans un DataFrame propre
df_coords = pd.DataFrame({
    "X": coord_x,
    "Y": coord_y
})


"""             Conversion coordonnées WGS84 ==> Lambert 93             """

def wgs84_to_lambert93(lat, lon):
    # Paramètres de l'ellipsoïde GRS80
    a = 6378137.0  # demi-grand axe
    e = 0.0818191910428  # première excentricité

    # Paramètres de la projection Lambert 93
    n = 0.7256077650
    C = 11754255.426
    xs = 700000.0
    ys = 12655612.050

    lon0 = 3 * math.pi / 180  # longitude d'origine en radians (3°E)

    # Convertir les coordonnées en radians
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)

    # Calculs intermédiaires
    l = math.log(math.tan(math.pi / 4 + lat_rad / 2) * ((1 - e * math.sin(lat_rad)) / (1 + e * math.sin(lat_rad)))**(e / 2))
    R = C * math.exp(-n * l)
    gamma = n * (lon_rad - lon0)

    # Coordonnées finales en Lambert 93
    x_lambert = xs + R * math.sin(gamma)
    y_lambert = ys - R * math.cos(gamma)

    return x_lambert, y_lambert


x_lambert, y_lambert = wgs84_to_lambert93(lat, lon)
print(f"X: {x_lambert}, Y: {y_lambert}")


# Étape 3 : Fonction de recherche des 5 postes les plus proches
def trouver_plus_proches(x_input, y_input, n=5):
    resultats = []

    for i in range(len(coord_x)):
        dx = coord_x[i] - x_input
        dy = coord_y[i] - y_input
        distance = (dx**2 + dy**2)**0.5

        if not math.isnan(dispo[i]):  # Filtrer les valeurs NaN
            temp = [
                i,
                nom_poste[i],
                coord_x[i],
                coord_y[i],
                distance,
                dispo[i],
                travaux[i]
            ]
            resultats.append(temp)

    resultats_trie = sorted(resultats, key=lambda x: x[4])
    return resultats_trie[:n]


def trouver_min(L):
    minimum=L[0]
    for i in range(len(L)):
        if L[i]<=minimum:
            minimum = L[i]
    return minimum

def trouver_max(L):
    maximum=L[0]
    for i in range(len(L)):
        if L[i]>=maximum:
            maximum = L[i]
    return maximum


résultat = trouver_plus_proches(x_lambert, y_lambert)

for site in résultat:
    print(f" {site[1]:}, \
          Distance: {site[4]:.2f} m, \
          Dispo: {site[5]:.2f} MW, \
          Travaux: {site[6]:.2f} MW ")



"""                Affichage des résultats                """
Coordonnées_X_Résultat = []
for i in range(len(résultat)):
    Coordonnées_X_Résultat.append(résultat[i][2])

Coordonnées_Y_Résultat = []
for i in range(len(résultat)):
    Coordonnées_Y_Résultat.append(résultat[i][3])

# Rogner le graphique autour de la zone d'intérêt
X_min = trouver_min(Coordonnées_X_Résultat)
X_max = trouver_max(Coordonnées_X_Résultat)
Y_min = trouver_min(Coordonnées_Y_Résultat)
Y_max = trouver_max(Coordonnées_Y_Résultat)

largeur_x = X_max - X_min
largeur_y = Y_max - Y_min

mid_X = (X_max + X_min) /2
mid_Y = (Y_max + Y_min) /2

point_bas_X = mid_X - max(largeur_x, largeur_y) / 2 - max(largeur_x, largeur_y)*1.2
point_bas_Y = mid_Y - max(largeur_x, largeur_y) / 2 - max(largeur_x, largeur_y)*1.2
point_haut_X = mid_X + max(largeur_x, largeur_y) / 2 + max(largeur_x, largeur_y)*1.2
point_haut_Y = mid_Y + max(largeur_x, largeur_y) / 2 + max(largeur_x, largeur_y)*1.2


# Créer le plot
fig, ax = plt.subplots(figsize=(10, 10))

# Scatter des points
ax.scatter(Coordonnées_X_Résultat, Coordonnées_Y_Résultat, s=1, alpha=0.5)
ax.scatter(Coordonnées_X_Résultat, Coordonnées_Y_Résultat, color='blue', label='5 plus proches')
ax.scatter(x_lambert, y_lambert, color='red', marker='x', label='Point recherché')

# Définir les limites (xlim/ylim) autour de ton point
ax.set_xlim(point_bas_X, point_haut_X)
ax.set_ylim(point_bas_Y, point_haut_Y)

# Ajouter automatiquement un fond OpenStreetMap adapté
ctx.add_basemap(ax, crs="EPSG:2154", source=ctx.providers.OpenStreetMap.Mapnik)

# Options graphiques
plt.title("Carte des postes haute tension avec fond OpenStreetMap")
plt.xlabel("Coordonnée X (m)")
plt.ylabel("Coordonnée Y (m)")
plt.grid(True)
plt.axis("equal")
plt.legend()
plt.show()




"""             Streamlit             """

import streamlit as st
import folium
from streamlit_folium import st_folium
from pyproj import Transformer

# Interface utilisateur
st.title("Calcul de Raccordement Électrique")

lat = st.number_input("Latitude du site", value=47.475806)
lon = st.number_input("Longitude du site", value=-0.560157)

# Bouton de calcul
if st.button("Calculer les 5 postes les plus proches"):
    x_lambert, y_lambert = wgs84_to_lambert93(lat, lon)
    resultat = trouver_plus_proches(x_lambert, y_lambert)
    st.session_state["resultat"] = resultat
    st.session_state["coords"] = (lat, lon)

# Affichage s’il y a des résultats en mémoire
if "resultat" in st.session_state and "coords" in st.session_state:
    resultat = st.session_state["resultat"]
    lat, lon = st.session_state["coords"]

    # Affichage texte
    for site in resultat:
        st.write(f"**{site[1]}** - Distance: {site[4]:.0f} m | Dispo: {site[5]:.2f} MW | Travaux: {site[6]:.2f} MW")

    # Carte
    m = folium.Map(location=[lat, lon], zoom_start=13)
    folium.Marker([lat, lon], tooltip="Site à raccorder", icon=folium.Icon(color="red")).add_to(m)

    transformer = Transformer.from_crs("EPSG:2154", "EPSG:4326", always_xy=True)
    for poste in resultat:
        x, y = poste[2], poste[3]
        lon_p, lat_p = transformer.transform(x, y)
        folium.Marker([lat_p, lon_p], tooltip=poste[1]).add_to(m)

    st_folium(m, width=700)

