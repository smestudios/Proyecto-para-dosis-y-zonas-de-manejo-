# dosificacion.py
import os
import glob
import pandas as pd
import geopandas as gpd
from tkinter import messagebox
import customtkinter as ctk

def crear_dosificacion(carpeta_raiz, root):
    if not carpeta_raiz:
        messagebox.showwarning("Atención", "Primero selecciona una carpeta")
        return

    popup = ctk.CTkToplevel(root)
    popup.geometry("400x200")
    popup.title("Crear Dosificación")
    
    label = ctk.CTkLabel(popup, text="Nombre del campo de dosis:", font=("Arial", 14))
    label.pack(pady=20)
    
    entry_nombre = ctk.CTkEntry(popup, placeholder_text="Ej: Dosis")
    entry_nombre.pack(pady=10)

    def guardar_dosificacion():
        nombre_campo = entry_nombre.get().strip()
        if not nombre_campo:
            messagebox.showwarning("Atención", "Debes ingresar un nombre para el campo de dosis")
            return
        
        try:
            carpeta_dosificacion = os.path.join(carpeta_raiz, "Dosificacion")
            os.makedirs(carpeta_dosificacion, exist_ok=True)

            carpeta_zonas = os.path.join(carpeta_raiz, "Zonas")
            lista_shapes = glob.glob(os.path.join(carpeta_zonas, "ZM_*.shp"))
            if not lista_shapes:
                messagebox.showwarning("Atención", f"No se encontró shapefile ZM_ en: {carpeta_zonas}")
                return
            archivo_shape = lista_shapes[0]
            gdf = gpd.read_file(archivo_shape)

            columnas_requeridas = ["Zonas", "Area"]
            for col in columnas_requeridas:
                if col not in gdf.columns:
                    gdf[col] = pd.NA

            gdf[nombre_campo] = 0.0
            gdf[nombre_campo] = gdf[nombre_campo].round(2)
            gdf = gdf.to_crs(epsg=4326)

            codigo = os.path.basename(archivo_shape).replace("ZM_","").replace(".shp","")
            archivo_guardado = os.path.join(carpeta_dosificacion, f"ZM_{codigo}_dosificacion.shp")
            gdf.to_file(archivo_guardado)

            messagebox.showinfo("Dosificación", f"Dosificación creada y shapefile guardado en:\n{archivo_guardado}")
            popup.destroy()
            
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo crear la dosificación:\n{e}")

    btn_guardar = ctk.CTkButton(popup, text="Guardar Dosificación", command=guardar_dosificacion)
    btn_guardar.pack(pady=20)
