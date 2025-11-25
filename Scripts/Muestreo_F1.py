# Librerías básicas
import os
import geopandas as gpd         # Equivalente a sf/rgdal (vectores)
import rasterio                # Equivalente a raster (rasters)
from rasterio.plot import show # Visualización
from rasterio.warp import calculate_default_transform, reproject
from rasterio.enums import Resampling
from rasterio.mask import mask
import rasterio.errors
from shapely.geometry import Point
import numpy as np             # Operaciones numéricas
import seaborn as sns
import pandas as pd            # Manejo de datos tabulares
from pyproj import CRS
from tkinter import Tk, filedialog
from tkinter.messagebox import showwarning

# Análisis espacial
import libpysal as lps         # Equivalente a spdep
from esda.moran import Moran   # Autocorrelación espacial

# Estadística y ML
from sklearn.decomposition import PCA  # PCA (ade4/factoextra)
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from scipy.spatial import cKDTree

# Visualización
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

# Visualizar el polígono (equivalente a plot()),Transformar coordenadas (equivalente a spTransform())
area_cal.plot()
area_cal_ms = area_cal.to_crs(ms3115)
plt.title("Polígono original")
plt.show()

def process_raster(input_path, output_name, reference_raster=None, input_dir="Raster", output_dir="Raster/Out"):
    """Carga, reproyecta, recorta y guarda un raster, con opción de alinearlo a un raster de referencia."""

    # Crear directorio de salida si no existe
    os.makedirs(output_dir, exist_ok=True)

    full_input_path = os.path.join(input_dir, input_path)
    full_output_path = os.path.join(output_dir, f"{output_name}.tif")
    temp_path = os.path.join(output_dir, "temp.tif")
    reproj_path = os.path.join(output_dir, "temp_reproj.tif")

    try:
        # 1. Reproyectar a ms3115
        with rasterio.open(full_input_path) as src:
            transform, width, height = calculate_default_transform(
                src.crs, ms3115, src.width, src.height, *src.bounds)
            kwargs = src.meta.copy()
            kwargs.update({'crs': ms3115, 'transform': transform, 'width': width, 'height': height})
            
            with rasterio.open(temp_path, 'w', **kwargs) as dst:
                reproject(
                    source=rasterio.band(src, 1),
                    destination=rasterio.band(dst, 1),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=ms3115,
                    resampling=Resampling.bilinear)

        # 2. Recortar con el polígono
        with rasterio.open(temp_path) as src:
            out_image, out_transform = mask(src, area_cal_ms.geometry, crop=True, nodata=np.nan)
            out_meta = src.meta.copy()
            out_meta.update({
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform
            })

        # 3. Alinear si hay raster de referencia
        if reference_raster:
            full_reference_path = os.path.join(output_dir, f"{reference_raster}.tif")
            with rasterio.open(full_reference_path) as ref:
                out_meta.update({
                    'transform': ref.transform,
                    'width': ref.width,
                    'height': ref.height
                })
                with rasterio.open(reproj_path, 'w', **out_meta) as dst:
                    reproject(
                        source=out_image,
                        destination=rasterio.band(dst, 1),
                        src_transform=out_transform,
                        src_crs=ms3115,
                        dst_transform=ref.transform,
                        dst_crs=ref.crs,
                        resampling=Resampling.bilinear)
            
            # Copiar raster alineado a la salida final
            with rasterio.open(reproj_path) as src_aligned:
                with rasterio.open(full_output_path, 'w', **out_meta) as dst_final:
                    dst_final.write(src_aligned.read())
        else:
            # Guardar el recorte directamente
            with rasterio.open(full_output_path, 'w', **out_meta) as dst:
                dst.write(out_image)

        # Visualización rápida
        with rasterio.open(full_output_path) as vis:
            plt.figure(figsize=(10, 10))
            plt.imshow(vis.read(1), cmap='RdYlGn')
            plt.title(output_name)
            plt.colorbar()
            plt.show()

        print(f"✅ Guardado exitosamente: {full_output_path}")
        return full_output_path

    except rasterio.errors.RasterioIOError:
        print(f"⚠️ Raster no encontrado: {full_input_path}")
        return None
    except Exception as e:
        print(f"❌ Error inesperado con {full_input_path}: {e}")
        return None
    finally:
        # Eliminar archivos temporales
        for f in [temp_path, reproj_path]:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except PermissionError:
                    print(f"⚠️ No se pudo eliminar {f}. Puede estar en uso.")


variables = {
    "NDVI": {"prefix": "NDVI_median", "output_prefix": "NDVI_Pl", "use_reference": True},
    "EVI": {"prefix": "EVI_median", "output_prefix": "EVI_Hist", "use_reference": True},
    "PROD": {"prefix": "PROD_median", "output_prefix": "PROD", "use_reference": True},
}

result_paths = {}

for var, config in variables.items():
    paths = []
    for i in range(1, 6):
        input_name = f"{config['prefix']}_{codigo}_{i}.tif"
        output_name = f"{config['output_prefix']}_{i}"

        # Si es la primera, no usa raster de referencia
        if config["use_reference"] and i > 1:
            path = process_raster(input_name, output_name, reference_raster="NDVI_Pl_1")
        else:
            try:
                path = process_raster(input_name, output_name)
            except FileNotFoundError:
                print(f"❌ Archivo no encontrado: {input_name}, se omitirá.")
                continue  # O puedes usar: path = None
        
        paths.append(path)

    result_paths[var] = paths

print("Variables procesadas:", list(result_paths.keys()))

outws = f'{workspace}/Raster/Out'

tif_files = [os.path.join(outws, f) for f in os.listdir(outws) if f.endswith('.tif')]

arrays = []
meta = None

for tif in tif_files:
    with rasterio.open(tif) as src:
        data = src.read(1)  # leer banda 1
        arrays.append(data)
        if meta is None:
            meta = src.meta

# Stack: apilar en un arreglo 3D (bandas, filas, columnas)
stack = np.stack(arrays, axis=0)

print(f"Stack shape: {stack.shape}")  # (num_bandas, height, width)

# Visualizar cada banda
fig, axes = plt.subplots(1, len(tif_files), figsize=(15, 5))
if len(tif_files) == 1:
    axes = [axes]

for i, ax in enumerate(axes):
    im = ax.imshow(stack[i], cmap='RdYlGn')
    ax.set_title(f'Band {i+1}')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()

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
grilla_dentro = generar_grilla(area_cal_ms, cellsize=10)

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

variables = df_valores.iloc[:, 0:10]  # Seleccionar columnas 0 a 9

# 0. Imputar NaN si existen
imputer = SimpleImputer(strategy="mean")
variables_imputed = imputer.fit_transform(variables)

# 1. Escalar los datos (centrar y escalar)
scaler = StandardScaler()
variables_scaled = scaler.fit_transform(variables_imputed)

# 2. Aplicar PCA
pca = PCA()
pca.fit(variables_scaled)

# 3. Obtener contribuciones (cargas) de las variables para el primer componente principal
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

# 4. Visualizar variables en el espacio PCA (como el biplot de fviz_pca_var)
plt.figure(figsize=(8, 8))
for i, varname in enumerate(variables.columns):
    plt.arrow(0, 0, loadings[i, 0], loadings[i, 1], color='r', alpha=0.5)
    plt.text(loadings[i, 0]*1.15, loadings[i, 1]*1.15, varname, color='g', ha='center', va='center')

plt.xlabel("PC1 (%.1f%% varianza)" % (pca.explained_variance_ratio_[0]*100))
plt.ylabel("PC2 (%.1f%% varianza)" % (pca.explained_variance_ratio_[1]*100))
plt.title('Contribución de variables a PC1 y PC2')
plt.grid()
plt.axhline(0, color='grey', lw=1)
plt.axvline(0, color='grey', lw=1)
plt.show()

df_valores.to_csv(f"CSV/{codigo}_R.csv", index=False)

# Crear geometría a partir de x, y
df_valores['geometry'] = [Point(xy) for xy in zip(df_valores['x'], df_valores['y'])]

# Convertir a GeoDataFrame
df_valores = gpd.GeoDataFrame(df_valores, geometry='geometry', crs="EPSG:3115")

area_cal_ms["area_m2"] = area_cal_ms.geometry.area
area_cal_ms["area_ha"] = area_cal_ms["area_m2"] / 10000

print("Áreas por suerte (hectáreas):")
print(area_cal_ms[["area_ha"]])

# Solicitar divisor al usuario
while True:
    try:
        divisor = float(input("\nIngrese el número de hectáreas por punto de muestreo (por ejemplo, 2.5): "))
        if divisor <= 0:
            raise ValueError
        break
    except ValueError:
        print("⚠️ Por favor, ingrese un número válido mayor que cero.")

# Calcular puntos por suerte según divisor ingresado
area_cal_ms["n_puntos"] = (area_cal_ms["area_ha"] / divisor).round().astype(int)
total_puntos = area_cal_ms["n_puntos"].sum()

print(f"\nCon un divisor de {divisor}, se muestrearán {total_puntos} puntos en total.")

# Paso 2: Extraer coordenadas x, y para clustering
coords = np.array(list(zip(df_valores.geometry.x, df_valores.geometry.y)))

# Paso 1: Aplicar KMeans igual que antes
kmeans = KMeans(n_clusters=total_puntos, random_state=42)
kmeans.fit(coords)
df_valores["cluster"] = kmeans.labels_

# Paso 2: Obtener los puntos más cercanos al centroide, pero dentro del conjunto válido
representantes = []
for i in range(total_puntos):
    centroide = kmeans.cluster_centers_[i]
    puntos_cluster = df_valores[df_valores["cluster"] == i]
    
    if puntos_cluster.empty:
        continue
    
    # Buscar el punto más cercano al centroide
    puntos_cluster["dist"] = puntos_cluster.geometry.apply(lambda p: p.distance(Point(centroide)))
    punto_mas_cercano = puntos_cluster.sort_values("dist").iloc[0]
    representantes.append(punto_mas_cercano)

# Crear GeoDataFrame con los puntos representativos
muestra_gdf = gpd.GeoDataFrame(representantes, geometry="geometry", crs=df_valores.crs).reset_index(drop=True)
muestra_gdf["n"] = range(1, len(muestra_gdf) + 1)

# 5. Visualizar agrupamiento
fig, ax = plt.subplots(figsize=(10, 10))
df_valores.plot(ax=ax, column="cluster", cmap="tab20", markersize=5, legend=False, alpha=0.5)
muestra_gdf.plot(ax=ax, color='red', markersize=30, label="Centroides")
area_cal_ms.boundary.plot(ax=ax, color="black")
plt.title("Agrupamiento KMeans y puntos representativos")
plt.legend()
plt.show()

# Extraer valores raster en los puntos de muestra (centroides)
coords_muestra = [(pt.x, pt.y) for pt in muestra_gdf.geometry]
valores_muestra = []

for tif in tif_files:
    with rasterio.open(tif) as src:
        muestras = list(src.sample(coords_muestra))
        valores_muestra.append(np.array(muestras).flatten())

# Convertir a array y luego a DataFrame
valores_muestra = np.array(valores_muestra).T
df_muestra = pd.DataFrame(valores_muestra, columns=nombres_columnas)
df_muestra['n'] = range(1, len(df_muestra) + 1)

muestra_final = muestra_gdf

indices = ['EVI_Hist_1', 'EVI_Hist_2','EVI_Hist_3', 'EVI_Hist_4', 'EVI_Hist_5','NDVI_Pl_1','NDVI_Pl_2','NDVI_Pl_3','NDVI_Pl_4', 'NDVI_Pl_5']

sns.set(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams['figure.facecolor'] = 'white'

# Iterar sobre cada índice para comparar
for ind in indices:
    if ind in df_valores.columns and ind in df_muestra.columns:
        # Crear figura con dos subplots
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f'Distribución de valores para {ind}', y=1.02, fontsize=16, fontweight='bold')
        
        # --- HISTOGRAMA CON KDE ---
        bins = min(50, int(len(df_valores[ind].dropna())**0.5))
        
        sns.histplot(
            data=df_valores[ind], 
            kde=True, 
            color='skyblue', 
            label='Total', 
            ax=axes[0],
            bins=bins,
            alpha=0.6,
            edgecolor='white',
            linewidth=0.5,
            stat='density'
        )
        
        sns.histplot(
            data=df_muestra[ind], 
            kde=True, 
            color='salmon', 
            label='Muestra', 
            ax=axes[0],
            bins=bins,
            alpha=0.6,
            edgecolor='white',
            linewidth=0.5,
            stat='density'
        )
        
        axes[0].set_title('Histograma con KDE', pad=20)
        axes[0].legend(title='Dataset:')
        axes[0].set_xlabel('Valores')
        axes[0].set_ylabel('Densidad')
        axes[0].grid(True, linestyle='--', alpha=0.7)
        
        # --- BOXPLOT MEJORADO ---
        plot_data = pd.DataFrame({
            'Dataset': ['Total']*len(df_valores[ind]) + ['Muestra']*len(df_muestra[ind]),
            'Valores': pd.concat([df_valores[ind], df_muestra[ind]])
        })

        plot_data = plot_data.reset_index(drop=True)
        
        # SOLUCIÓN: Usar sns.boxplot con order parameter en lugar de set_xticklabels
        box = sns.boxplot(
            x='Dataset',
            hue='Dataset',
            y='Valores',
            data=plot_data,
            ax=axes[1],
            palette=['skyblue', 'salmon'],
            width=0.5,
            showmeans=True,
            meanprops={'marker':'o', 'markerfacecolor':'white', 'markeredgecolor':'black'},
            order=['Total', 'Muestra'],  # Esto establece el orden y las etiquetas
            dodge=False
        )
        
        # Añadir puntos de datos
        sns.stripplot(
            x='Dataset',
            y='Valores',
            data=plot_data,
            ax=axes[1],
            color='black',
            alpha=0.3,
            size=4,
            jitter=True,
            order=['Total', 'Muestra']  # Mantener el mismo orden
        )
        
        axes[1].set_title('Boxplot Comparativo', pad=20)
        axes[1].set_xlabel('Dataset')
        axes[1].set_ylabel('Valores')
        axes[1].grid(True, linestyle='--', alpha=0.5, axis='y')
        
        # Eliminar la línea problemática:
        # axes[1].set_xticklabels(['Total', 'Muestra'])  # YA NO ES NECESARIA
        
        plt.tight_layout()
        plt.show()
        
        # Estadísticas adicionales
        print(f"\n📊 Estadísticas para {ind}:")
        print(f"Total - Media: {df_valores[ind].mean():.4f} | Std: {df_valores[ind].std():.4f}")
        print(f"Muestra - Media: {df_muestra[ind].mean():.4f} | Std: {df_muestra[ind].std():.4f}")
        print(f"Diferencia en medias: {(df_muestra[ind].mean()-df_valores[ind].mean()):.4f}")
        
    else:
        print(f"⚠️ El índice '{ind}' no está presente en ambos DataFrames.")

# 6. Exportar puntos
muestra_final.to_file(f"SHP/Muestras{codigo}.shp")
print(f"\nExportado a 'SHP/Muestras{codigo}.shp'")


