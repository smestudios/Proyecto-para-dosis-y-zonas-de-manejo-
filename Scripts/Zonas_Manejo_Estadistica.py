# Zonas_Manejo_Estadistica.py
import os
import glob
import geopandas as gpd
import pandas as pd
from rasterstats import zonal_stats

def normalizar_ruta(ruta):
    return ruta.replace("\\", "/")

def calcular_estadisticas_df(carpeta_raiz):
    """
    Calcula estadísticas zonales de los rásters sobre las zonas de manejo.
    Devuelve un DataFrame con las estadísticas.
    """
    carpeta_raster = os.path.join(carpeta_raiz, "Nutrientes")
    carpeta_zonas = os.path.join(carpeta_raiz, "Zonas")

    # Carga el shapefile ZM_
    archivos_zm = glob.glob(os.path.join(carpeta_zonas, "ZM_*.shp"))
    if not archivos_zm:
        raise FileNotFoundError("No se encontró shapefile ZM_ en la carpeta 'Zonas'")
    archivo_zm = archivos_zm[0]
    zonas_gdf = gpd.read_file(archivo_zm)

    # Orden deseado de las estadísticas
    orden_deseado = [
        "Arc","Arcilla", "Aren","Arena", "Lim","Limo", "pH", "MO", "CIC",
        "Ca", "Cu", "Fe", "K", "Mg", "Mn",
        "Na", "P", "S", "Zn", "B"
    ]

    # Calcula área en hectáreas
    if "Area" not in zonas_gdf.columns:
        zonas_gdf["Area"] = (zonas_gdf.geometry.area / 10_000).round(2)

    # Procesa cada raster
    archivos_raster = glob.glob(os.path.join(carpeta_raster, "*.tif"))
    for ruta_raster in archivos_raster:
        nombre = os.path.splitext(os.path.basename(ruta_raster))[0]  # solo el nombre del raster
        stats = zonal_stats(zonas_gdf, ruta_raster, stats=["mean"], geojson_out=False)
        columna_final = nombre.replace("_mean","")  # elimina "_mean" si existe en el nombre del raster
        zonas_gdf[columna_final] = [item["mean"] for item in stats]  

    # Reordena columnas
    columnas_estadisticas = [col for col in orden_deseado if col in zonas_gdf.columns]
    otras_columnas = [col for col in zonas_gdf.columns if col not in columnas_estadisticas]
    zonas_gdf = zonas_gdf[otras_columnas + columnas_estadisticas]

    # --- Priorizar columnas Arc/Aren/Lim ---
    for col_final, col_1, col_2 in [("Arcilla","Arc","Arcilla"), ("Arena","Aren","Arena"), ("Limo","Lim","Limo")]:
        if col_2 in zonas_gdf.columns:
            zonas_gdf[col_final] = zonas_gdf[col_2]
        elif col_1 in zonas_gdf.columns:
            zonas_gdf[col_final] = zonas_gdf[col_1]
        else:
            zonas_gdf[col_final] = pd.NA

    # Borrar columnas auxiliares antiguas
    for col in ["Arc","Aren","Lim","Arcilla","Arena","Limo"]:
        if col in zonas_gdf.columns and col not in ["Arcilla","Arena","Limo"]:
            zonas_gdf.drop(columns=col, inplace=True)

    # Sobrescribe shapefile
    zonas_gdf.to_file(archivo_zm)

    # Prepara DataFrame para Excel
    codigo = os.path.splitext(os.path.basename(archivo_zm))[0].split("_")[1]
    carpeta_dosificacion = os.path.join(carpeta_raiz, "Dosificacion")
    os.makedirs(carpeta_dosificacion, exist_ok=True)

    # 🔹 Ordenar según Zonas existentes
    zonas_gdf = zonas_gdf.sort_values(by="Zonas").reset_index(drop=True)

    # Ya no necesitamos reasignar Zonas
    columnas_final = ["Zonas", "Area"] + [col for col in orden_deseado if col in zonas_gdf.columns and col not in ["Arc","Aren","Lim"]]
    for col in columnas_final:
        if col not in zonas_gdf.columns:
            zonas_gdf[col] = pd.NA

    zonas_df = zonas_gdf[columnas_final]

    # Exporta Excel
    archivo_excel = os.path.join(carpeta_dosificacion, f"Suelos_{codigo}.xlsx")
    zonas_df.to_excel(archivo_excel, index=False)

    return zonas_df, archivo_excel
