
import pandas as pd
import folium
import os


# 1. Leer la base de datos usando ruta absoluta
csv_path = os.path.join(os.path.dirname(__file__), "municipios_zamora.csv")
df = pd.read_csv(csv_path)

# 2. Crear el mapa centrado en Zamora
mapa = folium.Map(location=[41.5, -5.75], zoom_start=8)

# 3. Añadir chinchetas desde el CSV
for idx, row in df.iterrows():
    # Texto del popup: municipio + dirección completa
    popup_text = f"{row['municipio']}<br>{row['calle']}, {row['numero']}"
    folium.Marker(
        location=[row['lat'], row['lon']],
        popup=popup_text,
        icon=folium.Icon(color="red", icon="info-sign")
    ).add_to(mapa)

# 4. Guardar el mapa en HTML
mapa.save("mapa_zamora_interactivo.html")
print("Mapa generado: mapa_zamora_interactivo.html")
