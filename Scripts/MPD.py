# MPD.py
"""
Interfaz MPD
Mantiene la funcionalidad original: visores raster/shape, cronológico, estadísticas y dosificación.
"""

# ============================
# Librerías del sistema
# ============================
import json
import os
import glob
from functools import partial

# ============================
# Librerías científicas / análisis
# ============================
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio

# ============================
# Librerías gráficas / visualización
# ============================
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.colors import ListedColormap, BoundaryNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ============================
# Librerías GUI (CustomTkinter + complementos)
# ============================
import customtkinter as ctk
from customtkinter import CTkImage
from CTkTable import CTkTable
from CTkListbox import CTkListbox
from tkinter import filedialog, messagebox

# ============================
# Librerías de imágenes
# ============================
from PIL import Image

# ============================
# Módulos propios del proyecto
# ============================
from dosificacion import crear_dosificacion
from UPM import (
    seleccionar_carpeta_base, mostrar_suertes, mostrar_haciendas,
    agregar_suertes, quitar_suerte, unir_suertes, set_base_path
)
from Zonas_Manejo_Estadistica import calcular_estadisticas_df
from cronologico_lookup import load_cronologico, find_by_codigo, prepare_display_df


# ============================
# Normalizacion de rutas
# ============================
def normalizar_ruta(ruta):
    # Si la ruta empieza con al menos una barra invertida, es UNC
    if ruta.startswith("\\") or ruta.startswith("//"):
        # Reemplazar las barras internas, pero mantener el prefijo UNC correcto
        ruta = ruta.replace("/", "\\")  # primero asegura formato Windows
        if not ruta.startswith("\\\\"):
            ruta = "\\" + ruta  # fuerza el doble backslash inicial
        return ruta
    # Para rutas locales (C:, D:, etc.)
    return ruta.replace("/", "\\")


# ============================
# Constantes / configuración
# ============================
DEFAULT_COLORS = ["#5dade2", "#ff09ff", "#58d68d", "#f5b041"]
THEME_FILE = os.path.join(os.path.dirname(__file__), "mustard.json")
LOGO_FILE = os.path.join(os.path.dirname(__file__), "Logo_mayaguez.png")
CONFIG_FILE = os.path.join(os.path.dirname(__file__), "configuracion.json")


# ============================
# Aplicación principal
# ============================
class MPDApp:
    def __init__(self):
        # Estado
        self.carpeta_raiz_zonas = None
        self.canvas_raster = None
        self.canvas_shape = None
        self.configuracion = self._load_configuracion()

        # Inicializar UI
        self._setup_theme()
        self.root = ctk.CTk()
        self._setup_root()
        self._create_layout()
        self._aplicar_configuracion_inicial()
        self.mostrar_frame(self.frame_bienvenida)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    # ----------------------------
    # Configuración inicial
    # ----------------------------
    def _setup_theme(self):
        ctk.set_appearance_mode("")
        if os.path.exists(THEME_FILE):
            ctk.set_default_color_theme(THEME_FILE)

    def _setup_root(self):
        self.root.title("MENU DE PROCESOS AP MAYAGUEZ")
        self.root.geometry("900x600")
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)

    # ----------------------------
    # Configuración persistente
    # ----------------------------
    def _load_configuracion(self):
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        return data
            except Exception:
                pass
        return {}

    def _guardar_configuracion(self, nueva_config=None):
        if nueva_config:
            self.configuracion.update(nueva_config)
        try:
            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(self.configuracion, f, indent=2, ensure_ascii=False)
        except Exception:
            messagebox.showerror("Error", "No se pudo guardar la configuración en disco.")

    def _aplicar_configuracion_inicial(self):
        self._refrescar_entries_configuracion()
        carpeta_base_puntos = self.configuracion.get("carpeta_base_puntos")
        if carpeta_base_puntos:
            self.entry_base.delete(0, "end")
            self.entry_base.insert(0, carpeta_base_puntos)
            set_base_path(carpeta_base_puntos)
            self.label_info.configure(text="Carpeta base precargada desde configuración")

        carpeta_trabajo = self.configuracion.get("carpeta_trabajo_principal")
        if carpeta_trabajo and not self.carpeta_raiz_zonas:
            self.label_carpeta.configure(text=f"Carpeta principal: {carpeta_trabajo}")

    def _refrescar_entries_configuracion(self):
        valores = {
            "entry_config_carpeta_trabajo": self.configuracion.get("carpeta_trabajo_principal", ""),
            "entry_config_base_puntos": self.configuracion.get("carpeta_base_puntos", ""),
            "entry_config_principal_zonas": self.configuracion.get("carpeta_principal_zonas", ""),
            "entry_config_shp_hdaste": self.configuracion.get("shapefile_hdaste", ""),
            "entry_config_cronologico": self.configuracion.get("cronologico_path", ""),
        }

        for attr, valor in valores.items():
            entry = getattr(self, attr, None)
            if entry is not None:
                entry.delete(0, "end")
                entry.insert(0, valor)

    def _crear_bloque_config(self, contenedor, fila, titulo, placeholder, boton_texto, comando):
        frame = ctk.CTkFrame(contenedor, corner_radius=10)
        frame.grid(row=fila, column=0, padx=15, pady=5, sticky="ew")
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_columnconfigure(1, weight=0)

        label = ctk.CTkLabel(frame, text=titulo, anchor="w")
        label.grid(row=0, column=0, padx=10, pady=(8, 0), sticky="w")

        entry = ctk.CTkEntry(frame, placeholder_text=placeholder)
        entry.grid(row=1, column=0, padx=10, pady=8, sticky="ew")

        boton = ctk.CTkButton(frame, text=boton_texto, command=comando)
        boton.grid(row=1, column=1, padx=10, pady=8, sticky="ew")

        return entry

    def _seleccionar_directorio_en_entry(self, entry, titulo):
        initial_dir = entry.get().strip() or self.configuracion.get("carpeta_trabajo_principal") or os.getcwd()
        carpeta = filedialog.askdirectory(title=titulo, initialdir=initial_dir)
        if carpeta:
            entry.delete(0, "end")
            entry.insert(0, normalizar_ruta(carpeta))

    def _seleccionar_archivo_en_entry(self, entry, titulo, tipos):
        initial_dir = os.path.dirname(entry.get().strip()) if entry.get().strip() else (self.configuracion.get("carpeta_trabajo_principal") or os.getcwd())
        archivo = filedialog.askopenfilename(title=titulo, filetypes=tipos, initialdir=initial_dir)
        if archivo:
            entry.delete(0, "end")
            entry.insert(0, normalizar_ruta(archivo))

    def guardar_configuracion(self):
        nueva_config = {
            "carpeta_trabajo_principal": self.entry_config_carpeta_trabajo.get().strip(),
            "carpeta_base_puntos": self.entry_config_base_puntos.get().strip(),
            "carpeta_principal_zonas": self.entry_config_principal_zonas.get().strip(),
            "shapefile_hdaste": self.entry_config_shp_hdaste.get().strip(),
            "cronologico_path": self.entry_config_cronologico.get().strip(),
        }

        self._guardar_configuracion(nueva_config)
        self.label_estado_config.configure(text="Configuración guardada.", text_color="green")
        self._aplicar_configuracion_inicial()

    def crear_estructura_carpeta(self):
        carpeta_base = self.entry_config_carpeta_trabajo.get().strip() or self.entry_config_principal_zonas.get().strip()
        if not carpeta_base:
            messagebox.showwarning("Atención", "Primero selecciona la carpeta principal para crear la estructura.")
            return

        carpeta_base = normalizar_ruta(carpeta_base)
        os.makedirs(carpeta_base, exist_ok=True)
        subcarpetas = ["Zonas", "Nutrientes", "Dosificacion", "Mascara", "SHP"]
        for sub in subcarpetas:
            os.makedirs(os.path.join(carpeta_base, sub), exist_ok=True)

        self.configuracion["carpeta_trabajo_principal"] = carpeta_base
        self._guardar_configuracion()
        self._refrescar_entries_configuracion()
        self.label_estado_config.configure(
            text=f"Estructura creada/actualizada en: {carpeta_base}", text_color="#2E86C1"
        )

    # ----------------------------
    # Layout y widgets
    # ----------------------------
    def _create_layout(self):
        # Contenedor principal
        contenido_frame = ctk.CTkFrame(self.root, corner_radius=10)
        contenido_frame.grid(row=0, column=1, padx=15, pady=15, sticky="nsew")
        contenido_frame.grid_rowconfigure(0, weight=1)
        contenido_frame.grid_columnconfigure(0, weight=1)

        # Frame bienvenida
        self.frame_bienvenida = ctk.CTkFrame(contenido_frame, corner_radius=10)
        self.frame_bienvenida.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.frame_bienvenida.grid_columnconfigure(0, weight=1)
        label_bienvenida = ctk.CTkLabel(self.frame_bienvenida, text="Bienvenido al menú de procesos",
                                        font=("Arial", 20))
        label_bienvenida.grid(row=0, column=0, pady=20, padx=10)

        # Menú lateral
        menu_frame = ctk.CTkFrame(self.root, width=200, corner_radius=10)
        menu_frame.grid(row=0, column=0, padx=15, pady=15, sticky="ns")
        menu_frame.grid_propagate(False)
        menu_frame.grid_columnconfigure(0, weight=1)

        btn_puntos = ctk.CTkButton(menu_frame, text="Unificacion de Puntos",
                                   command=lambda: self.mostrar_frame(self.frame_puntos))
        btn_puntos.grid(row=1, column=0, padx=20, pady=(10, 10), sticky="ew")

        btn_zonas = ctk.CTkButton(menu_frame, text="Zonas de Manejo",
                                  command=lambda: self.mostrar_frame(self.frame_zonas))
        btn_zonas.grid(row=2, column=0, padx=20, pady=10, sticky="ew")

        btn_config = ctk.CTkButton(menu_frame, text="Configuración",
                                   command=lambda: self.mostrar_frame(self.frame_configuracion))
        btn_config.grid(row=3, column=0, padx=20, pady=10, sticky="ew")

        # Logo
        if os.path.exists(LOGO_FILE):
            try:
                imagen_logo = Image.open(LOGO_FILE)
                logo_ctk = CTkImage(light_image=imagen_logo, dark_image=imagen_logo, size=(150, 200))
                label_logo = ctk.CTkLabel(menu_frame, image=logo_ctk, text="")
                label_logo.grid(row=0, column=0, pady=(10, 0), padx=0)
            except Exception:
                # si falla la carga de imagen, se ignora y se continúa
                pass

        # ------- Frames principales: puntos y zonas -------
        self._create_frame_puntos(contenido_frame)
        self._create_frame_zonas(contenido_frame)
        self._create_frame_configuracion(contenido_frame)

    def _create_frame_puntos(self, parent):
        self.frame_puntos = ctk.CTkFrame(parent, corner_radius=10)
        self.frame_puntos.grid(row=0, column=0, sticky="nsew")
        self.frame_puntos.grid_columnconfigure(0, weight=1)
        self.frame_puntos.grid_remove()

        self.frame_puntos.grid_rowconfigure(0, weight=0)
        self.frame_puntos.grid_rowconfigure(1, weight=0)
        self.frame_puntos.grid_rowconfigure(2, weight=1)

        # Carpeta y controles
        frame_carpeta = ctk.CTkFrame(self.frame_puntos, corner_radius=10)
        frame_carpeta.grid(row=0, column=0, sticky="ew", pady=10, padx=10)
        frame_carpeta.grid_columnconfigure(0, weight=1)

        self.entry_base = ctk.CTkEntry(frame_carpeta, placeholder_text="Selecciona carpeta base")
        self.entry_base.grid(row=0, column=0, sticky="ew", padx=5, pady=5)

        btn_seleccionar = ctk.CTkButton(frame_carpeta, text="Seleccionar",
                                         command=lambda: seleccionar_carpeta_base(self.entry_base, self.label_info,
                                                                                 self.tree, self.btn_ver_suertes))
        btn_seleccionar.grid(row=0, column=1, padx=5, pady=5)

        self.label_info = ctk.CTkLabel(self.frame_puntos, text="Selecciona la carpeta base para comenzar")
        self.label_info.grid(row=1, column=0, pady=5, sticky="w", padx=10)

        panel_principal = ctk.CTkFrame(self.frame_puntos, corner_radius=10)
        panel_principal.grid(row=2, column=0, sticky="nsew", padx=10, pady=10)
        panel_principal.grid_rowconfigure(0, weight=1, minsize=200)
        panel_principal.grid_rowconfigure(1, weight=2, minsize=200)
        panel_principal.grid_columnconfigure(0, weight=1)

        self.tree = CTkListbox(panel_principal, multiple_selection=False)
        self.tree.grid(row=0, column=0, sticky="nsew", padx=(10, 5), pady=(2, 2))

        self.listbox_seleccionadas = CTkListbox(panel_principal, multiple_selection=True)
        self.listbox_seleccionadas.grid(row=1, column=0, sticky="nsew", padx=(10, 5), pady=(2, 2))

        # Botones a la derecha
        frame_botones = ctk.CTkFrame(panel_principal, corner_radius=10)
        frame_botones.grid(row=0, column=1, rowspan=2, sticky="ns", padx=5, pady=10)
        frame_botones.grid_rowconfigure((0, 1, 2, 3, 4), weight=1)

        self.btn_ver_suertes = ctk.CTkButton(frame_botones, text="Ver suertes",
                                             command=lambda: mostrar_suertes(self.tree, self.label_info,
                                                                            self.btn_agregar, self.btn_atras))
        self.btn_ver_suertes.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

        def _agregar_suertes_wrapper():
            try:
                agregar_suertes(self.tree, self.listbox_seleccionadas, self.label_info)
            except Exception as e:
                messagebox.showerror("Error", f"No se pudieron agregar las suertes:\n{e}")

        self.btn_agregar = ctk.CTkButton(frame_botones, text="Agregar suertes",
                                         command=_agregar_suertes_wrapper)
        self.btn_agregar.grid(row=1, column=0, padx=5, pady=5, sticky="ew")

        self.btn_quitar = ctk.CTkButton(frame_botones, text="Quitar suerte",
                                        command=lambda: quitar_suerte(self.listbox_seleccionadas, self.label_info))
        self.btn_quitar.grid(row=2, column=0, padx=5, pady=5, sticky="ew")

        self.btn_atras = ctk.CTkButton(frame_botones, text="Atrás",
                                       command=lambda: mostrar_haciendas(self.entry_base, self.label_info, self.tree,
                                                                        self.btn_ver_suertes))
        self.btn_atras.grid(row=3, column=0, padx=5, pady=5, sticky="ew")

        self.btn_unir = ctk.CTkButton(frame_botones, text="Unir suertes seleccionadas", command=unir_suertes)
        self.btn_unir.grid(row=4, column=0, padx=5, pady=10, sticky="ew")

    def _create_frame_zonas(self, parent):

        # Scrollable frame para zonas
        self.frame_zonas = ctk.CTkScrollableFrame(parent, corner_radius=10, orientation="vertical")
        self.frame_zonas.grid(row=0, column=0, sticky="nsew")
        # filas: 0 título, 1 seleccionar carpeta, 2 label suerte, 3 cronológico, 4 tabla estadística,
        # 5 botón estad, 6 visores, 7 dosificación
        weights = [(0, 0), (1, 0), (2, 0), (3, 1), (4, 2), (5, 0), (6, 3), (7, 0)]
        for idx, (r, w) in enumerate(weights):
            self.frame_zonas.grid_rowconfigure(idx, weight=w)
        self.frame_zonas.grid_columnconfigure(0, weight=1)

        # encabezado y título
        franja_encabezado = ctk.CTkFrame(self.frame_zonas, corner_radius=6, fg_color="#B3AAAA")
        franja_encabezado.grid(row=0, column=0, sticky="ew")
        label_zonas = ctk.CTkLabel(franja_encabezado, text="Zonas de Manejo", font=("Arial", 16),
                                   fg_color="transparent")
        label_zonas.pack(pady=10, padx=10, fill="x")

        # seleccionar carpeta
        frame_carpeta_zonas = ctk.CTkFrame(self.frame_zonas, corner_radius=10)
        frame_carpeta_zonas.grid(row=1, column=0, sticky="ew", pady=5)
        frame_carpeta_zonas.grid_columnconfigure(1, weight=1)

        btn_carpeta = ctk.CTkButton(frame_carpeta_zonas, text="Seleccionar Carpeta",
                                    command=self.seleccionar_carpeta_zonas)
        btn_carpeta.grid(row=0, column=0, padx=5, pady=5)

        self.label_carpeta = ctk.CTkLabel(frame_carpeta_zonas, text="No hay carpeta seleccionada")
        self.label_carpeta.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        # label suerte
        self.label_suerte = ctk.CTkLabel(self.frame_zonas, text="Suerte: ---", font=("Arial", 14, "bold"))
        self.label_suerte.grid(row=2, column=0, padx=20, pady=5, sticky="w")

        # tabla cronológico (scroll vertical)
        frame_tabla_cronologico = ctk.CTkFrame(self.frame_zonas, corner_radius=10)
        frame_tabla_cronologico.grid(row=3, column=0, padx=20, pady=10, sticky="nsew")
        frame_tabla_cronologico.grid_rowconfigure(0, weight=1)
        frame_tabla_cronologico.grid_columnconfigure(0, weight=1)

        frame_tabla_cronologico_scroll = ctk.CTkScrollableFrame(frame_tabla_cronologico, corner_radius=10,
                                                                orientation="vertical")
        frame_tabla_cronologico_scroll.grid(row=0, column=0, sticky="nsew")

        self.tabla_cronologico = CTkTable(
            frame_tabla_cronologico_scroll,
            dataframe=pd.DataFrame(columns=['Cod.', 'Hacienda', 'Ste', 'Edad', 'Variedad', 'NC', 'Fecha Liquidacion',
                                            'TCHC', 'Area']),
            corner_radius=8
        )
        self.tabla_cronologico.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

                # === Sección de métricas TCH ===
        frame_resumen = ctk.CTkFrame(self.frame_zonas, corner_radius=10)
        frame_resumen.grid(row=4, column=0, padx=20, pady=10, sticky="ew")
        frame_resumen.grid_columnconfigure((0, 1, 2), weight=1)

        # --- TCH Actual ---
        self.titulo_actual = ctk.CTkLabel(frame_resumen, text="TCH ACTUAL", font=("Arial", 12))
        self.titulo_actual.grid(row=0, column=0, pady=(5, 0))
        self.valor_actual = ctk.CTkButton(frame_resumen, text="--", state="disabled", fg_color="#2E86C1")
        self.valor_actual.grid(row=1, column=0, padx=5, pady=5, sticky="ew")

        # --- TCH Promedio ---
        self.titulo_promedio = ctk.CTkLabel(frame_resumen, text="TCH PROMEDIO", font=("Arial", 12))
        self.titulo_promedio.grid(row=0, column=1, pady=(5, 0))
        self.valor_promedio = ctk.CTkButton(frame_resumen, text="--", state="disabled", fg_color="#27AE60")
        self.valor_promedio.grid(row=1, column=1, padx=5, pady=5, sticky="ew")

        # --- TCH con 20% ---
        self.titulo_20 = ctk.CTkLabel(frame_resumen, text="TCH CON 20%", font=("Arial", 12))
        self.titulo_20.grid(row=0, column=2, pady=(5, 0))
        self.valor_20 = ctk.CTkButton(frame_resumen, text="--", state="disabled", fg_color="#E67E22")
        self.valor_20.grid(row=1, column=2, padx=5, pady=5, sticky="ew")


        # tabla de estadísticas (CTkScrollableFrame horizontal)
        self.frame_tabla = ctk.CTkScrollableFrame(self.frame_zonas, corner_radius=10, orientation="horizontal")
        self.frame_tabla.grid(row=4, column=0, padx=20, pady=10, sticky="nsew")
        self.frame_tabla.grid_rowconfigure(0, weight=1)
        self.frame_tabla.grid_columnconfigure(0, weight=1)

        self.tabla_ctk = CTkTable(self.frame_tabla, dataframe=pd.DataFrame(), corner_radius=8)
        self.tabla_ctk.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        # botón calcular estadísticas
        btn_calcular_estadisticas = ctk.CTkButton(self.frame_zonas, text="Calcular estadísticas",
                                                 command=self.ejecutar_estadisticas)
        btn_calcular_estadisticas.grid(row=5, column=0, padx=20, pady=10, sticky="ew")

        # visores: raster y shape
        frame_raster = ctk.CTkFrame(self.frame_zonas, corner_radius=10, fg_color="transparent")
        frame_raster.grid(row=6, column=0, padx=20, pady=10, sticky="nsew")
        frame_raster.grid_rowconfigure(0, weight=1)
        frame_raster.grid_columnconfigure(0, weight=1)   # columna izquierda expansible
        frame_raster.grid_columnconfigure(1, weight=1)   # columna derecha expansible


        # left: raster viewer
        self.frame_raster_left = ctk.CTkFrame(frame_raster, corner_radius=10)
        self.frame_raster_left.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        label_raster = ctk.CTkLabel(self.frame_raster_left, text="Visor de Zonas (Raster)",
                                    font=("Arial", 14, "bold"))
        label_raster.pack(pady=5)

        # right: shape viewer
        self.frame_raster_right = ctk.CTkFrame(frame_raster, corner_radius=10)
        self.frame_raster_right.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        label_shape = ctk.CTkLabel(self.frame_raster_right, text="Visor de Zonas (Shape)",
                                   font=("Arial", 14, "bold"))
        label_shape.pack(pady=5)

        # botón crear dosificación
        btn_crear_dosificacion = ctk.CTkButton(
            self.frame_zonas,
            text="Crear Dosificación",
            fg_color="green",
            hover_color="#32CD32",
            command=lambda: crear_dosificacion(self.carpeta_raiz_zonas, self.root)
        )
        btn_crear_dosificacion.grid(row=7, column=0, padx=20, pady=10, sticky="ew")

    def _create_frame_configuracion(self, parent):
        self.frame_configuracion = ctk.CTkScrollableFrame(parent, corner_radius=10, orientation="vertical")
        self.frame_configuracion.grid(row=0, column=0, sticky="nsew")
        self.frame_configuracion.grid_columnconfigure(0, weight=1)
        for i in range(6):
            self.frame_configuracion.grid_rowconfigure(i, weight=0)
        self.frame_configuracion.grid_remove()

        titulo = ctk.CTkLabel(self.frame_configuracion, text="Configuración general", font=("Arial", 18, "bold"))
        titulo.grid(row=0, column=0, padx=15, pady=(10, 5), sticky="w")

        descripcion = ctk.CTkLabel(
            self.frame_configuracion,
            text=(
                "Define carpetas y archivos predeterminados para no tener que seleccionarlos en cada sesión. "
                "Puedes crear la estructura base de carpetas y guardar las rutas en un solo lugar."
            ),
            wraplength=700,
            justify="left"
        )
        descripcion.grid(row=1, column=0, padx=15, pady=(0, 10), sticky="w")

        self.entry_config_carpeta_trabajo = self._crear_bloque_config(
            contenedor=self.frame_configuracion,
            fila=2,
            titulo="Carpeta de trabajo principal (HDASTE)",
            placeholder="Selecciona la carpeta principal donde se crearán las subcarpetas",
            boton_texto="Seleccionar",
            comando=lambda: self._seleccionar_directorio_en_entry(
                self.entry_config_carpeta_trabajo, "Selecciona la carpeta de trabajo"
            )
        )

        self.entry_config_base_puntos = self._crear_bloque_config(
            contenedor=self.frame_configuracion,
            fila=3,
            titulo="Carpeta base de haciendas (Unificación de puntos)",
            placeholder="Ej: \\\\SERVIDOR\\Ingenio\\Haciendas",
            boton_texto="Seleccionar",
            comando=lambda: self._seleccionar_directorio_en_entry(
                self.entry_config_base_puntos, "Selecciona la carpeta base de haciendas"
            )
        )

        self.entry_config_principal_zonas = self._crear_bloque_config(
            contenedor=self.frame_configuracion,
            fila=4,
            titulo="Carpeta principal para Zonas de Manejo",
            placeholder="Selecciona la carpeta principal donde están las suertes",
            boton_texto="Seleccionar",
            comando=lambda: self._seleccionar_directorio_en_entry(
                self.entry_config_principal_zonas, "Selecciona la carpeta principal de Zonas"
            )
        )

        self.entry_config_shp_hdaste = self._crear_bloque_config(
            contenedor=self.frame_configuracion,
            fila=5,
            titulo="Shapefile HDA-STE de trabajo (máscara)",
            placeholder="Selecciona el SHP que contiene los HDASTE",
            boton_texto="Buscar SHP",
            comando=lambda: self._seleccionar_archivo_en_entry(
                self.entry_config_shp_hdaste,
                "Selecciona el shapefile principal",
                [("Shapefile", "*.shp"), ("Todos", "*.*")]
            )
        )

        self.entry_config_cronologico = self._crear_bloque_config(
            contenedor=self.frame_configuracion,
            fila=6,
            titulo="Archivo Cronológico/Prescripción",
            placeholder="Selecciona el archivo Excel a usar por defecto",
            boton_texto="Buscar Excel",
            comando=lambda: self._seleccionar_archivo_en_entry(
                self.entry_config_cronologico,
                "Selecciona el archivo Cronologico.xlsx",
                [("Excel", "*.xlsx"), ("Excel", "*.xls"), ("Todos", "*.*")]
            )
        )

        botones_frame = ctk.CTkFrame(self.frame_configuracion, corner_radius=10)
        botones_frame.grid(row=7, column=0, padx=15, pady=10, sticky="ew")
        botones_frame.grid_columnconfigure((0, 1), weight=1)

        btn_guardar = ctk.CTkButton(botones_frame, text="Guardar configuración", command=self.guardar_configuracion)
        btn_guardar.grid(row=0, column=0, padx=10, pady=10, sticky="ew")

        btn_crear_estructura = ctk.CTkButton(
            botones_frame,
            text="Crear estructura de carpetas",
            fg_color="#2E86C1",
            hover_color="#1B4F72",
            command=self.crear_estructura_carpeta
        )
        btn_crear_estructura.grid(row=0, column=1, padx=10, pady=10, sticky="ew")

        self.label_estado_config = ctk.CTkLabel(self.frame_configuracion, text="")
        self.label_estado_config.grid(row=8, column=0, padx=15, pady=(0, 10), sticky="w")

    # ----------------------------
    # Funcionalidad de selección de carpeta / cronológico
    # ----------------------------
    def seleccionar_carpeta_zonas(self):
        initial_dir = self.configuracion.get("carpeta_principal_zonas") or self.configuracion.get("carpeta_trabajo_principal") or os.getcwd()
        carpeta = filedialog.askdirectory(title="Selecciona la carpeta de la hacienda", initialdir=initial_dir)
        if not carpeta:
            return
        self.carpeta_raiz_zonas = carpeta
        self.carpeta_raiz_zonas = normalizar_ruta(self.carpeta_raiz_zonas)
        self.label_carpeta.configure(text=carpeta)
        self.label_suerte.configure(text=f"Suerte: {os.path.basename(carpeta)}")

        self.configuracion["ultima_carpeta_zonas"] = self.carpeta_raiz_zonas
        self._guardar_configuracion()

        # Mostrar mapas (intenta raster y shape)
        for t in ("raster", "shape"):
            self.mostrar_zonas(t)

        # actualizar cronológico por código extraído
        codigo = self._obtener_codigo_de_carpeta(carpeta)
        if not codigo:
            messagebox.showwarning("Atención", "No se encontró ZM_*.tif o ZM_*.shp en la carpeta Zonas. No se puede consultar Cronologico.")
            return
        self.actualizar_tabla_cronologico(codigo)


    def _obtener_codigo_de_carpeta(self, carpeta):
        if not carpeta:
            return None
        carpeta_zonas = os.path.join(carpeta, "Zonas")
        if not os.path.isdir(carpeta_zonas):
            return None
        lista = [normalizar_ruta(p) for p in sorted(glob.glob(os.path.join(carpeta_zonas, "ZM_*.tif")) + glob.glob(os.path.join(carpeta_zonas, "ZM_*.shp")))]
        if not lista:
            return None
        archivo = os.path.basename(lista[0])
        # extraer después de ZM_ y antes de la extensión
        codigo = archivo.replace("ZM_", "")
        codigo = os.path.splitext(codigo)[0]
        return codigo

    # ----------------------------
    # Lectura datos (separadas)
    # ----------------------------
    def _leer_raster(self, ruta):
        with rasterio.open(ruta) as src:
            data = src.read(1).astype(float)
            nodata = src.nodata
        if nodata is not None:
            data = np.ma.masked_equal(data, nodata)
        else:
            data = np.ma.masked_where((data == 0) | np.isnan(data), data)
        valores = np.unique(data.compressed()).astype(int).tolist() if np.ma.is_masked(data) or data.size else []
        return data, valores

    def _leer_shape(self, ruta):
        gdf = gpd.read_file(ruta)
        # Detectar campo de zonas (orden de prioridad)
        posibles_campos = ["Zonas", "zona", "DN", "ID", "id"]
        campo_zona = next((c for c in posibles_campos if c in gdf.columns), None)
        if not campo_zona:
            campo_zona = gdf.columns[0]  # usa el primero como último recurso

        # Limpiar geometrías inválidas
        gdf = gdf[gdf.geometry.notnull() & gdf.is_valid]

        # Filtrar valores únicos de zona
        valores = sorted(gdf[campo_zona].dropna().unique())

        return gdf, campo_zona, valores


    # ----------------------------
    # Colormap y utilidades
    # ----------------------------
    def generar_colormap(self, valores_unicos, colores_fijos=None):
        if colores_fijos is None:
            colores_fijos = DEFAULT_COLORS
        # determinar máximo valor (suponiendo valores numerables)
        max_valor = int(max(valores_unicos))
        colores_finales = colores_fijos[:]
        if max_valor > len(colores_finales):
            colores_finales += ["#d3d3d3"] * (max_valor - len(colores_finales))
        colores_finales = colores_finales[:max_valor]
        cmap = ListedColormap(colores_finales)
        cmap.set_bad("white")
        norm = BoundaryNorm(np.arange(0.5, max_valor + 1.5, 1), ncolors=cmap.N, clip=True)
        return cmap, norm, max_valor

    def _clear_canvas(self, tipo):
        if tipo == "raster" and self.canvas_raster:
            try:
                self.canvas_raster.get_tk_widget().destroy()
            except Exception:
                pass
            self.canvas_raster = None
        if tipo == "shape" and self.canvas_shape:
            try:
                self.canvas_shape.get_tk_widget().destroy()
            except Exception:
                pass
            self.canvas_shape = None

    # ----------------------------
    # Mostrar zonas (raster / shape)
    # ----------------------------
    def mostrar_zonas(self, tipo="raster"):
        # comprobar carpeta
        if not self.carpeta_raiz_zonas:
            return
        carpeta_zonas = os.path.join(self.carpeta_raiz_zonas, "Zonas")
        if tipo == "raster":
            lista = sorted(glob.glob(os.path.join(carpeta_zonas, "ZM_*.tif")))
        else:
            lista = sorted(glob.glob(os.path.join(carpeta_zonas, "ZM_*.shp")))
        if not lista:
            messagebox.showwarning("Atención", f"No se encontró ningún {tipo} ZM_ en: {carpeta_zonas}")
            return

        archivo = normalizar_ruta(lista[0])
        # Compatibilidad extra para geopandas con rutas UNC
        if archivo.startswith("//"):
            archivo = archivo.replace("//", r"\\\\")  # convierte al formato \\Servidor\Carpeta

        codigo = os.path.basename(archivo).replace("ZM_", "")
        codigo = os.path.splitext(codigo)[0]

        try:
            if tipo == "raster":
                data, valores_unicos = self._leer_raster(archivo)
            else:
                gdf, campo_zona, valores_unicos = self._leer_shape(archivo)

            if not valores_unicos:
                messagebox.showwarning("Atención", f"No se encontraron valores válidos en el {tipo}")
                return

            cmap, norm, max_valor = self.generar_colormap(valores_unicos)

            fig, ax = plt.subplots(figsize=(5, 5))
            im = None
            if tipo == "raster":
                im = ax.imshow(data, cmap=cmap, norm=norm, interpolation="nearest")
            else:
                # mapear cada zona a color
                colores = {v: cmap(norm(v)) for v in valores_unicos}
                gdf.plot(ax=ax, color=gdf[campo_zona].map(colores), edgecolor="black")

            ax.set_title(f"Mapa de Zonas - {codigo}", fontsize=12)
            ax.axis("off")
            fig.tight_layout()

            if tipo == "raster" and im is not None:
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                ticks = np.arange(1, max_valor + 1)
                labels = [f"Zona {int(t)}" for t in ticks]
                cbar = fig.colorbar(im, cax=cax, ticks=ticks)
                cbar.ax.set_yticklabels(labels, fontsize=10)
                cbar.set_label("Zonas de Manejo", rotation=270, labelpad=15)

            # limpiar canvas previo y crear uno nuevo
            self._clear_canvas(tipo)
            if tipo == "raster":
                self.canvas_raster = FigureCanvasTkAgg(fig, master=self.frame_raster_left)
                self.canvas_raster.draw()
                self.canvas_raster.get_tk_widget().pack(fill="both", expand=True)
            else:
                self.canvas_shape = FigureCanvasTkAgg(fig, master=self.frame_raster_right)
                self.canvas_shape.draw()
                self.canvas_shape.get_tk_widget().pack(fill="both", expand=True)

        except Exception as e:
            messagebox.showerror("Error", f"No se pudo mostrar {tipo}:\n{e}")

    # ----------------------------
    # Estadísticas y cronológico
    # ----------------------------
    def ejecutar_estadisticas(self):
        if not self.carpeta_raiz_zonas:
            messagebox.showwarning("Atención", "Primero selecciona una carpeta")
            return
        try:
            df, archivo_excel = calcular_estadisticas_df(self.carpeta_raiz_zonas)
            df = df.round(2)
            self.tabla_ctk.update_table(df)
            messagebox.showinfo("Listo", f"Estadísticas calculadas y Excel exportado a:\n{archivo_excel}")
        except Exception as e:
            messagebox.showerror("Error", f"Ocurrió un problema:\n{e}")

    def actualizar_tabla_cronologico(self, codigo):
        df_cronologico = load_cronologico(self.configuracion.get("cronologico_path"))
        if df_cronologico is None:
            messagebox.showwarning("Atención", "No se pudo cargar el Cronologico.")
            return
        df_filtrado = find_by_codigo(df_cronologico, codigo)
        df_display = prepare_display_df(df_filtrado)
        if df_display.empty:
            try:
                self.tabla_cronologico.update_table(pd.DataFrame())
            except Exception:
                # recrear si es necesario
                self.tabla_cronologico = CTkTable(self.tabla_cronologico.master, dataframe=pd.DataFrame(), corner_radius=8)
                self.tabla_cronologico.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
            messagebox.showinfo("Aviso", f"No se encontraron registros para el código {codigo}")
            return
        # actualizar en sitio
        self.tabla_cronologico.update_table(df_display)

    # ----------------------------
    # Vista / utilidades
    # ----------------------------
    def mostrar_frame(self, frame):
        # ocultar todos y mostrar el solicitado
        for f in (self.frame_bienvenida, self.frame_puntos, self.frame_zonas, self.frame_configuracion):
            try:
                f.grid_remove()
            except Exception:
                pass
        frame.grid(row=0, column=0, sticky="nsew")

    def on_closing(self):
        try:
            for task in self.root.tk.call("after", "info"):
                try:
                    self.root.after_cancel(task)
                except Exception:
                    pass
        except Exception:
            pass
        try:
            self.root.quit()
            self.root.destroy()
        except Exception:
            pass

    # ----------------------------
    # Ejecutar app
    # ----------------------------
    def run(self):
        self.root.mainloop()


# ============================
# Entrypoint
# ============================
if __name__ == "__main__":
    app = MPDApp()
    app.run()
