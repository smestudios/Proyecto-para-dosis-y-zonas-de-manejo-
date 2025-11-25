# UPM.py
import os
import geopandas as gpd
import pandas as pd
from tkinter import filedialog, messagebox

# --- Variables globales ---
rutas_seleccionadas = []
nombres_suertes = []
haciendas = []
suertes_actuales = []
base_path = ""

# --- Funciones ---
def seleccionar_carpeta_base(entry_base, label_info, tree, boton_ver_suertes):
    """Selecciona la carpeta base donde se encuentran las haciendas"""
    global base_path
    carpeta = filedialog.askdirectory(title="Selecciona la carpeta base (INGENIO MAYAGÜEZ)")
    if carpeta:
        base_path = os.path.normpath(os.path.abspath(carpeta))
        entry_base.delete(0, "end")
        entry_base.insert(0, carpeta)
        mostrar_haciendas(entry_base, label_info, tree, boton_ver_suertes)

def mostrar_haciendas(entry_base, label_info, tree, boton_ver_suertes):
    """Muestra las haciendas disponibles en la carpeta base"""
    global haciendas
    tree.delete(0, "end")
    haciendas = []
    if not base_path:
        messagebox.showwarning("Aviso", "Primero selecciona la carpeta base")
        return
    for hacienda in sorted(os.listdir(base_path)):
        ruta_hacienda = os.path.join(base_path, hacienda)
        if os.path.isdir(ruta_hacienda):
            haciendas.append((hacienda, ruta_hacienda))
            tree.insert(tree.size(), hacienda)  # usar tree.size() es más seguro con CTkListbox
    label_info.configure(text="Selecciona una hacienda y presiona 'Ver suertes'")
    boton_ver_suertes.grid()

def mostrar_suertes(tree, label_info, boton_agregar, boton_atras):
    """Muestra las suertes dentro de la hacienda seleccionada"""
    global suertes_actuales
    seleccion = tree.curselection()
    print ( " prueba " + str(seleccion))
    if seleccion is None:
        messagebox.showwarning("Aviso", "Selecciona una hacienda primero")
        return

    # aseguramos que sea lista aunque sea un solo entero
    if isinstance(seleccion, int):
        seleccion = [seleccion]

    index = seleccion[0]  # tomamos el primer seleccionado


    hacienda, ruta_hacienda = haciendas[index]
    print("prueba2" + str(ruta_hacienda))

    tree.grid_remove()
    tree.delete(0, "end")
    tree.grid()
    suertes_actuales = []

    for carpeta_suerte in sorted(os.listdir(ruta_hacienda)):
        ruta_suerte = os.path.join(ruta_hacienda, carpeta_suerte, "SHP")
        ruta_suerte = os.path.normpath(ruta_suerte)
        if os.path.exists(ruta_suerte):
            for f in os.listdir(ruta_suerte):
                if f.lower().endswith(".shp") and f.startswith("Muestras"):
                    suertes_actuales.append((hacienda, carpeta_suerte, os.path.join(ruta_suerte, f)))
                    tree.insert(tree.size(), carpeta_suerte)  # ya está correcto, solo asegúrate de usar tree.size()

    label_info.configure(text="Selecciona suertes y presiona 'Agregar'")
    boton_agregar.grid()
    boton_atras.grid()

def agregar_suertes(tree, listbox, label_info):
    """Agrega las suertes seleccionadas al listado"""
    global rutas_seleccionadas, nombres_suertes
    seleccion = tree.curselection()
    if seleccion is None:
        messagebox.showwarning("Aviso", "Selecciona al menos una suerte")
        return

    # convertir a lista si es entero
    if isinstance(seleccion, int):
        seleccion = [seleccion]

    for idx in seleccion:
        h, nombre_s, ruta = suertes_actuales[idx]
        if ruta not in rutas_seleccionadas:
            rutas_seleccionadas.append(ruta)
            nombres_suertes.append(nombre_s)

    actualizar_listbox(listbox)
    label_info.configure(text=f"{len(rutas_seleccionadas)} suertes seleccionadas")

def quitar_suerte(listbox, label_info):
    """Quita las suertes seleccionadas del listado"""
    global rutas_seleccionadas, nombres_suertes
    seleccion = listbox.curselection()
    if seleccion is None:
        messagebox.showwarning("Aviso", "Selecciona una suerte para quitar")
        return

    # convertir a lista si es entero
    if isinstance(seleccion, int):
        seleccion = [seleccion]

    for i in reversed(seleccion):
        del rutas_seleccionadas[i]
        del nombres_suertes[i]


    actualizar_listbox(listbox)
    label_info.configure(text=f"{len(rutas_seleccionadas)} suertes seleccionadas")

def actualizar_listbox(listbox):
    """Actualiza el Listbox de suertes seleccionadas"""
    listbox.delete(0, "end")
    for nombre in nombres_suertes:
        listbox.insert("end", nombre)

def unir_suertes():
    """Une todos los shapefiles seleccionados en uno solo"""
    global rutas_seleccionadas
    if not rutas_seleccionadas:
        messagebox.showwarning("Aviso", "No hay suertes seleccionadas")
        return

    gdfs = []
    crs_objetivo = "EPSG:3116"
    total_puntos = 0

    for ruta_shp in rutas_seleccionadas:
        if os.path.exists(ruta_shp):
            gdf = gpd.read_file(ruta_shp)
            gdf = gdf.set_crs(crs_objetivo) if gdf.crs is None else gdf.to_crs(crs_objetivo)
            if "n" in gdf.columns:
                gdf["Muestra"] = "M" + gdf["n"].astype(str)

                # Extraemos solo el código que viene después de "Muestras" y antes del "_"
                nombre_archivo = os.path.basename(ruta_shp)  # ejemplo: MuestrasCODIGO_2024.shp
                codigo = nombre_archivo.replace("Muestras", "").split("_")[0]  # extrae solo "CODIGO"

                gdf["name"] = gdf["Muestra"] + "_" + codigo  # genera M1_CODIGO, M2_CODIGO, etc.

                total_puntos += len(gdf)
                gdfs.append(gdf)
            else:
                messagebox.showerror("Error", f"{ruta_shp} no tiene columna 'n'")
                return
        else:
            messagebox.showerror("Error", f"No se encontró el archivo SHP: {ruta_shp}")
            return

    gdf_unido = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=crs_objetivo)
    gdf_unido = gdf_unido[["n", "Muestra", "name", "geometry"]]

    save_path = filedialog.asksaveasfilename(defaultextension=".shp",
                                             filetypes=[("Shapefile", "*.shp"), ("GPX", "*.gpx")],
                                             title="Guardar capa unida")
    if save_path:
        try:
            if save_path.lower().endswith('.shp'):
                gdf_unido.to_file(save_path)
            elif save_path.lower().endswith('.gpx'):
                gdf_unido.to_file(save_path, driver="GPX")
            messagebox.showinfo("Éxito", f"Capa unida guardada con {total_puntos} puntos en:\n{save_path}")
        except Exception as e:
            messagebox.showerror("Error al guardar", str(e))
