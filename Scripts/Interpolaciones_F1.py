# Manejo de datos
import os
import pandas as pd
import numpy as np
import seaborn as sns
import unicodedata


# Geodatos vectoriales
import geopandas as gpd
from shapely.geometry import Point, Polygon
from tkinter import Tk, filedialog
from tkinter.messagebox import showwarning

# Ráster
import rasterio
from rasterio.mask import mask
from rasterio.transform import from_origin
from rasterio.features import rasterize
from rasterio.crs import CRS
from rasterstats import point_query
import xarray as xr
import rioxarray
import glob

# Visualización interactiva
import folium

# Machine learning
from shapely.geometry import Point
from shapely.geometry import box
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from libpysal.weights import DistanceBand
import skfuzzy as fuzz

# Geoestadística
from pykrige import OrdinaryKriging
from pykrige.uk import UniversalKriging
from esda import Moran

# Otras utilidades
import matplotlib.pyplot as plt

def seleccionar_carpeta():
    """Muestra un diálogo para que el usuario seleccione la carpeta de trabajo"""
    root = Tk()
    root.withdraw()  # Ocultar la ventana principal
    root.attributes('-topmost', True)  # Llevar el diálogo al frente
    
    carpeta = filedialog.askdirectory(
        title='Seleccione la carpeta de trabajo',
        initialdir='//mayaco/Agricultura_Precision$/MUESTREO DE SUELOS/SuertesAnalisis'
    )
    
    if not carpeta:
        showwarning("Advertencia", "No se seleccionó ninguna carpeta. Se usará el directorio actual.")
        return os.getcwd()
    
    return carpeta

# Uso en tu script
workspace = seleccionar_carpeta()
os.chdir(workspace)
print(f"Carpeta seleccionada: {workspace}")

codigo = input("Ingrese el código (por ejemplo, BS54): ")

# Definir sistema de coordenadas (equivalente a ms3115)
ms3115 = "+proj=tmerc +lat_0=4.596200416666666 +lon_0=-77.07750791666666 +k=1 +x_0=1000000 +y_0=1000000 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs"

# Cargar el shapefile (equivalente a shapefile())
area_cal = gpd.read_file(f"Mascara/{codigo}_M.shp")
puntos_cal = gpd.read_file(f"SHP/Muestras{codigo}.shp")

area_cal = area_cal.to_crs(ms3115)
puntos_cal = puntos_cal.to_crs(ms3115)

# Graficar
fig, ax = plt.subplots(figsize=(8, 6))
area_cal.plot(ax=ax, edgecolor='black', facecolor='none')  # contorno del polígono
puntos_cal.plot(ax=ax, color='red', markersize=30)         # puntos sobre el polígono

plt.title("Área de calibración con puntos de muestreo")
plt.xlabel("Coordenada X")
plt.ylabel("Coordenada Y")
plt.tight_layout()
plt.show()

# Definir el directorio de trabajo
outws = "Raster/Out"

# Buscar archivos .tif en el directorio
IT = glob.glob(os.path.join(outws, "*.tif"))
print(IT)

# Cargar los rásteres como lista de objetos rasterio
IT_stack = [rasterio.open(ruta) for ruta in IT]
IT_stack = [rioxarray.open_rasterio(ruta, masked=True) for ruta in IT]

# Combinar en un solo array si tienen la misma forma y sistema de referencia
stacked = xr.concat(IT_stack, dim="band")

tif_files = [os.path.join(outws, f) for f in os.listdir(outws) if f.endswith('.tif')]

# --- Paso 1: Generar una grilla regular de puntos dentro del polígono ---
def generar_grilla(area_gdf, cellsize=10):
    bounds = area_gdf.total_bounds  # xmin, ymin, xmax, ymax
    xmin, ymin, xmax, ymax = bounds
    x_coords = np.arange(xmin, xmax, cellsize)
    y_coords = np.arange(ymin, ymax, cellsize)

    puntos = [Point(x, y) for x in x_coords for y in y_coords]
    grilla = gpd.GeoDataFrame(geometry=puntos, crs=area_gdf.crs)

    # Filtrar solo los puntos que están dentro del polígono (con buffer negativo)
    area_buff = area_gdf.buffer(-11)  # buffer negativo para evitar bordes
    grilla_dentro = grilla[grilla.geometry.within(area_buff.geometry.union_all())]

    return grilla_dentro

# Asumiendo que tienes 'area_cal_ms' cargado como GeoDataFrame
grilla_dentro = generar_grilla(area_cal, cellsize=10)

# --- Paso 2: Extraer valores del stack raster para cada punto ---
coords = [(pt.x, pt.y) for pt in grilla_dentro.geometry]
valores = []

for tif in tif_files:  # 'tif_files' es la lista de tus .tif en Raster/Out
    with rasterio.open(tif) as src:
        muestras = list(src.sample(coords))
        valores.append(np.array(muestras).flatten())

# Convertir a array (n_puntos, n_bandas)
valores = np.array(valores).T

nombres_columnas = [os.path.basename(tif).replace('.tif', '')[:10] for tif in tif_files]

df_valores = pd.DataFrame(valores, columns=nombres_columnas)
df_valores['x'] = [pt.x for pt in grilla_dentro.geometry]
df_valores['y'] = [pt.y for pt in grilla_dentro.geometry]

df_valores["geometry"] = [Point(xy) for xy in zip(df_valores["x"], df_valores["y"])]
grid_mask2 = gpd.GeoDataFrame(df_valores, geometry="geometry", crs=area_cal.crs)

variables = df_valores.iloc[:, 0:10]  # Seleccionar columnas 0 a 9

# 0. Imputar NaN si existen
imputer = SimpleImputer(strategy="mean")
variables_imputed = imputer.fit_transform(variables)

# 1. Escalar los datos (centrar y escalar)
scaler = StandardScaler()
variables_scaled = scaler.fit_transform(variables_imputed)

# Paso 1: Cargar la tabla de nutrientes de suelos
suelo_df = pd.read_csv("Nutrientes/suelos.csv", sep=";", decimal=".", skipinitialspace=True)
print(suelo_df.info())

# Paso 2: Unir puntos de calibración con datos de suelos por la columna 'n'
puntos_cal = puntos_cal.merge(suelo_df, on='n')  # Asegúrate de que 'n' esté presente en ambos

# Paso 3: Guardar el shapefile combinado
puntos_cal.to_file("SHP/p_RF.shp", driver='ESRI Shapefile')
puntos_cal.to_csv("CSV/p_RFdf.csv", index=False)
# Convertir a DataFrame (si lo necesitas sin geometría)
# Diccionario estándar de columnas de índices y producción

nombres_estandar = {
    'EVI_Hist_1': 'EVI_Hist1',
    'EVI_Hist_2': 'EVI_Hist2',
    'EVI_Hist_3': 'EVI_Hist3',
    'EVI_Hist_4': 'EVI_Hist4',
    'EVI_Hist_5': 'EVI_Hist5',
    'NDVI_Pl_1': 'NDVI_Pl_1',
    'NDVI_Pl_2': 'NDVI_Pl_2',
    'NDVI_Pl_3': 'NDVI_Pl_3',
    'NDVI_Pl_4': 'NDVI_Pl_4',
    'NDVI_Pl_5': 'NDVI_Pl_5',
    'PROD_21': 'PROD21',
    'PROD_22': 'PROD22',
    'PROD_23': 'PROD23',
    'PROD_PRO': 'PRODPRO',
    'PROD_CV': 'PRODCV'
}

p_RFdf = pd.DataFrame(puntos_cal.drop(columns="geometry"))

RF = pd.read_csv("CSV/p_RFdf.csv", sep=",", decimal=".")
RF = RF.rename(columns=nombres_estandar)

# --- Selección de columnas ---
columnas_indices = [col for col in RF.columns if col.startswith(("EVI", "NDVI", "PROD"))]

columnas_suelo = [
    "Arcilla", "Limo", "Arena", "pH", "MO", "P", "K", "Na", "Ca", "Mg", "CIC", "S", "Fe", "Mn", "Zn", "Cu", "B"
]

# --- Resultados ---
importancias_resultado = {}
modelos_rf = {}
umbral_importancia = 0.05  # Puedes modificar este umbral según el análisis

for variable in columnas_suelo:
    if variable not in RF.columns:
        print(f"❌ Variable no encontrada: {variable}")
        continue

    X = RF[columnas_indices].apply(pd.to_numeric, errors='coerce')
    y = RF[variable].apply(pd.to_numeric, errors='coerce')

    datos = pd.concat([X, y], axis=1).dropna()
    X_clean = datos[columnas_indices]
    y_clean = datos[variable]

    # Entrenar el modelo usando todos los predictores para calcular la importancia
    modelo_temp = RandomForestRegressor(n_estimators=1500, max_features=3, random_state=42, n_jobs=-1)
    modelo_temp.fit(X_clean, y_clean)

    # Calcular importancias
    importancia = pd.Series(modelo_temp.feature_importances_, index=columnas_indices)
    importancia_ordenada = importancia.sort_values(ascending=False)
    importancias_resultado[variable] = importancia_ordenada

    # 🔍 Filtrar predictores relevantes según el umbral
    predictores_importantes = importancia_ordenada[importancia_ordenada > umbral_importancia].index.tolist()
    print(f"\n📊 {variable}: {len(predictores_importantes)} predictores importantes seleccionados")
    print(predictores_importantes)

    # Usar solo predictores seleccionados para entrenar el modelo final
    X_final = datos[predictores_importantes]
    modelo_final = RandomForestRegressor(n_estimators=1500, max_features=3, random_state=42, n_jobs=-1)
    modelo_final.fit(X_final, y_clean)
    modelos_rf[variable] = modelo_final

def predecir_variables(modelos_rf, grid_df, columnas_entrada):
    for variable, modelo in modelos_rf.items():
        try:
            # Forzar el orden y nombres exactos usados en entrenamiento
            X = grid_df.loc[:, modelo.feature_names_in_].apply(pd.to_numeric, errors='coerce')
            X = X.fillna(X.mean())

            pred = modelo.predict(X)
            grid_df[variable] = pred
            print(f"✅ Predicción lista para {variable}")
        except KeyError as e:
            print(f"❌ Columnas faltantes para {variable}: {e}")
        except Exception as e:
            print(f"❌ Error general al predecir {variable}: {e}")
    
    return grid_df

# Aplicar modelos a la grilla
grid_mask2 = grid_mask2.rename(columns=nombres_estandar)
grid_mask2 = predecir_variables(modelos_rf, grid_mask2, columnas_indices)
grid_mask2.to_file(f"SHP/grid_pred_{codigo}.shp", driver="ESRI Shapefile")

# Dimensiones y resolución del raster final
cellsize = 10  # mismo usado en la grilla
xmin, ymin, xmax, ymax = grid_mask2.total_bounds
ncols = int((xmax - xmin) / cellsize)
nrows = int((ymax - ymin) / cellsize)
gridx = np.linspace(xmin, xmax, ncols)
gridy = np.linspace(ymin, ymax, nrows)

# Salida
out_dir = "Nutrientes"
os.makedirs(out_dir, exist_ok=True)

for variable in columnas_suelo:
    if variable not in grid_mask2.columns:
        print(f"❌ {variable} no está en grid_mask2. Saltando...")
        continue

    data = grid_mask2[[variable, "geometry"]].dropna()
    x = data.geometry.x
    y = data.geometry.y
    z = data[variable]

    try:
        print(f"🔄 Interpolando {variable}...")
        OK = OrdinaryKriging(
            x, y, z,
            variogram_model="spherical",
            verbose=False,
            enable_plotting=False
        )
        z_interp, ss = OK.execute("grid", gridx, gridy)

        # Reemplazar NaN por nodata
        z_interp = np.where(np.isnan(z_interp), -9999, z_interp)

        transform = from_origin(xmin, ymax, cellsize, cellsize)
        meta = {
            "driver": "GTiff",
            "height": z_interp.shape[0],
            "width": z_interp.shape[1],
            "count": 1,
            "dtype": "float32",
            "crs": grid_mask2.crs,
            "transform": transform,
            "nodata": -9999
        }

        # Asegurar geometrías válidas
        area_geom = area_cal[area_cal.is_valid].geometry

        with rasterio.io.MemoryFile() as memfile:
            with memfile.open(**meta) as tmp_dst:
                tmp_dst.write(z_interp, 1)
            with memfile.open() as tmp_src:
                out_image, out_transform = mask(tmp_src, area_geom, crop=True)

                out_meta = tmp_src.meta.copy()
                out_meta.update({
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform,
                    "nodata": -9999
                })

        salida = os.path.join(out_dir, f"{variable}.tif")
        with rasterio.open(salida, "w", **out_meta) as dst:
            dst.write(out_image)

        print(f"✅ {variable} recortado correctamente en: {salida}")

    except Exception as e:
        print(f"⚠️ Error con {variable}: {e}")

# --- Selección de variables para MULTISPATI-PCA ---
# Lista de variables edáficas e índices de vegetación predichos
# Ajusta esta lista según las variables que consideres importantes

variables_multivariadas = ["Arcilla", "Limo", "Arena", "pH", "MO", "Na", "Ca", "CIC"]
X = grid_mask2[variables_multivariadas].copy()

# 1. Imputación por si quedan valores faltantes
X = X.apply(pd.to_numeric, errors='coerce')
X = X.fillna(X.mean())

# 2. Escalar
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. PCA
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

# 4. Visualizar la varianza explicada
plt.figure(figsize=(6, 4))
plt.plot(np.cumsum(pca.explained_variance_ratio_)*100, marker='o')
plt.xlabel('Número de Componentes')
plt.ylabel('% Varianza Explicada Acumulada')
plt.title('PCA - Varianza Explicada')
plt.grid(True)
plt.tight_layout()
plt.show()

# 5. Matriz de carga (contribución de cada variable a los PCs)
cargas = pd.DataFrame(
    pca.components_.T,
    columns=['PC1', 'PC2', 'PC3'],
    index=variables_multivariadas
)

print("📊 Cargas de los componentes principales:\n", cargas.round(3))

df_multiv = grid_mask2[variables_multivariadas + ["x", "y"]].dropna()

# Coordenadas en array Nx2
coords_df = df_multiv[["x", "y"]].copy()
coords_array = coords_df.to_numpy()

# Crear matriz de pesos espaciales con un radio de 50 metros
w_dist = DistanceBand(coords_array, threshold=50, binary=True, silence_warnings=True)

print(w_dist.n)  # número de observaciones
print(w_dist.histogram)  # distribución de vecinos por punto

# Calcular los scores de PCA
scores = pca.transform(X_scaled)

# Almacenar los Moran's I para cada componente
moran_results = []

for i in range(scores.shape[1]):
    comp = scores[:, i]
    moran = Moran(comp, w_dist)
    moran_results.append({
        "Componente": f"PC{i+1}",
        "Moran I": moran.I,
        "p-value": moran.p_sim
    })

# Mostrar resumen
moran_df = pd.DataFrame(moran_results)
print("📈 Autocorrelación espacial (Moran's I):")
print(moran_df)

# Transponer para que skfuzzy lo acepte (features x muestras)
data = scores[:, :3].T  # usar los primeros 3 componentes

# Definir número de clusters (provisionalmente, por ejemplo 3)
n_clusters = 3

# Fuzzy C-means
cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    data=data,
    c=n_clusters,
    m=2.0,
    error=0.005,
    maxiter=1000,
    init=None,
    seed=42
)

# Para cada punto, asignar la clase de mayor pertenencia
labels = np.argmax(u, axis=0)

# Agregar a tu dataframe
grid_mask2["Zona"] = labels

fpcs = []
range_n = range(2, 6)

for n_clusters in range_n:
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        data=scores[:, :3].T,
        c=n_clusters,
        m=2.0,
        error=0.005,
        maxiter=1000,
        init=None,
        seed=42
    )
    fpcs.append(fpc)

# Graficar FPC
plt.figure(figsize=(6, 4))
plt.plot(range_n, fpcs, marker='o')
plt.xlabel("Número de Clusters")
plt.ylabel("Fuzzy Partition Coefficient (FPC)")
plt.title("Selección del Número de Clusters")
plt.grid(True)
plt.tight_layout()
plt.show()

 #Elegir número óptimo según la gráfica
n_clusters = int(input("Ingrese el número óptimo de clusters según el FPC: "))

# Repetimos el clustering con el número de clusters seleccionado
cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    data=data,
    c=n_clusters,
    m=2.0,
    error=0.005,
    maxiter=1000,
    init=None,
    seed=42
)

print(f"FPC para {n_clusters} clusters:", round(fpc, 3))

# Procesar resultados
u_transposed = u.T
cluster_labels = np.argmax(u_transposed, axis=1)

# Agregar al dataframe
grid_mask2['Cluster'] = cluster_labels
for i in range(n_clusters):
    grid_mask2[f'Pertenencia_Cluster{i}'] = u_transposed[:, i]

# Asegúrate de que tenga geometría válida
if grid_mask2.geometry.is_empty.any():
    print("❗Algunas geometrías están vacías.")
else:
    # Mapa de clusters
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    grid_mask2.plot(column='Cluster', cmap='tab10', legend=True, ax=ax, edgecolor='black')
    ax.set_title(f"Zonificación en {n_clusters} Clusters", fontsize=14)
    plt.tight_layout()
    plt.show()

# Suponiendo que grid_mask2 tiene una columna 'Cluster' y 'geometry' con puntos
res = 10  # tamaño del píxel (ajústalo a tu malla en metros)

# Crear polígonos cuadrados alrededor de cada punto
def punto_a_celda(punto, resolucion):
    x, y = punto.x, punto.y
    return box(x - resolucion / 2, y - resolucion / 2,
               x + resolucion / 2, y + resolucion / 2)

# Generar geometría de celda
grid_mask2['geometry'] = grid_mask2.geometry.apply(lambda p: punto_a_celda(p, res))
grid_poligonos = grid_mask2.copy()

# 1. Eliminar columnas innecesarias o conflictivas
zonas_cluster = grid_poligonos.dissolve(by='Cluster', as_index=False)

# Visualizar
zonas_cluster.plot(column='Cluster', cmap='tab10', edgecolor='black', legend=True)
plt.title(f"Zonas de manejo por Cluster (n = {n_clusters})")
plt.show()

# Requiere: columnas 'x', 'y', 'Cluster'
df = grid_mask2.copy()
df = df.dropna(subset=["Cluster"])

# Asegurar que los clusters empiecen desde 1
df["Cluster"] = df["Cluster"].astype(int) + 1

# Obtener resolución (por ejemplo, 10 m)
res = 10  # puedes ajustar esto

# Calcular el bounding box
xmin, ymin, xmax, ymax = df.total_bounds

# Tamaño de la matriz ráster
n_cols = int(np.ceil((xmax - xmin) / res))
n_rows = int(np.ceil((ymax - ymin) / res))

# Crear matriz vacía con valor 0 (no data)
raster_data = np.zeros((n_rows, n_cols), dtype=np.uint8)

# Crear transformación (para georreferenciación)
transform = from_origin(xmin, ymax, res, res)

# Mapear cada punto a fila/columna en la matriz
for _, row in df.iterrows():
    col = int((row["x"] - xmin) / res)
    fila = int((ymax - row["y"]) / res)
    if 0 <= fila < n_rows and 0 <= col < n_cols:
        raster_data[fila, col] = row["Cluster"]

# Exportar a GeoTIFF
output_tif = f"Zonas/ZM_{codigo}.tif"
with rasterio.open(
    output_tif,
    "w",
    driver="GTiff",
    height=n_rows,
    width=n_cols,
    count=1,
    dtype=raster_data.dtype,
    crs=CRS.from_epsg(3115),  # Cambia al EPSG que estés usando, ejemplo: MAGNA-SIRGAS
    transform=transform,
) as dst:
    dst.write(raster_data, 1)

print(f"✅ Raster exportado a: {output_tif}")




