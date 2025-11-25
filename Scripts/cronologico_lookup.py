# cronologico_lookup.py
"""
Utilities para cargar y preparar el 'Cronologico' usado por la GUI.
API pública:
    - load_cronologico(excel_path=None, sheet_name="REPORTE") -> pd.DataFrame | None
    - find_by_codigo(df, codigo) -> pd.DataFrame
    - prepare_display_df(df_filtered) -> pd.DataFrame
    - format_df_for_label(df_display) -> str
"""

import os
import glob
from typing import Optional
import pandas as pd
from functools import lru_cache

__all__ = ["load_cronologico", "find_by_codigo", "prepare_display_df", "format_df_for_label"]

# --------------------------
# Búsqueda/autodiscovery de archivos Excel
# --------------------------
def _find_excel_autodiscover(start_paths=None, patterns=None, ascend_levels=3):
    """
    Busca archivos Excel comunes (Prescripcion*, Cronologico*, *.xlsx) en start_paths
    y subiendo hasta `ascend_levels` niveles desde el cwd como fallback.
    Retorna la primera ruta encontrada o None.
    """
    if patterns is None:
        patterns = [
            "Cronologico*.xlsx", "Cronologico*.xls",
            "Prescripcion*.xlsx", "Prescripcion*.xls",
            "*.xlsx", "*.xls"
        ]
    if start_paths is None:
        start_paths = [os.getcwd(), os.path.dirname(__file__)]

    # Buscar en las rutas iniciales
    for base in start_paths:
        for patt in patterns:
            hits = glob.glob(os.path.join(base, patt))
            if hits:
                return os.path.abspath(hits[0])

    # Subir desde cwd
    p = os.getcwd()
    for _ in range(ascend_levels):
        p = os.path.dirname(p)
        if not p:
            break
        for patt in patterns:
            hits = glob.glob(os.path.join(p, patt))
            if hits:
                return os.path.abspath(hits[0])

    return None

# --------------------------
# Selección automática de columnas por tokens
# --------------------------
def _pick_column_by_tokens(df: pd.DataFrame, token_list):
    """
    Devuelve el nombre de la columna que mejor case con alguno de los tokens.
    - token_list: lista de strings (ej: ['cod', 'codigo'])
    - Retorna None si no encuentra coincidencias.
    """
    if df is None or df.columns.empty:
        return None

    cols = list(df.columns)
    cols_low = [str(c).lower() for c in cols]
    tokens = [t.lower() for t in token_list]

    best_col = None
    best_score = 0
    for i, cname in enumerate(cols_low):
        # score = cuántos tokens aparecen en el nombre de la columna
        score = sum(1 for t in tokens if t in cname)
        if score > best_score:
            best_score = score
            best_col = cols[i]

    return best_col if best_score > 0 else None

# --------------------------
# Cache simple para evitar lecturas repetidas
# --------------------------
_CACHED_CRONO = {"path": None, "df": None}

# --------------------------
# Cargar cronológico (API usada por MPD.py)
# --------------------------
def load_cronologico(excel_path: Optional[str] = None, sheet_name: str = "REPORTE") -> Optional[pd.DataFrame]:
    """
    Intenta leer el cronológico desde:
      1. excel_path (si se pasa)
      2. 'Cronologico.xlsx' en cwd o carpeta del script
      3. Autodiscover (buscar patrones comunes)
    Devuelve DataFrame o None si falla.
    """
    # si nos pasan ruta explícita, usarla
    if excel_path:
        excel_path = os.path.abspath(excel_path)
        if not os.path.exists(excel_path):
            # no existe la ruta explícita
            excel_path = None

    # buscar rutas conocidas si no recibimos ruta válida
    if excel_path is None:
        posibles = [
            os.path.join(os.getcwd(), "Cronologico.xlsx"),
            os.path.join(os.path.dirname(__file__), "Cronologico.xlsx")
        ]
        found = next((p for p in posibles if os.path.exists(p)), None)
        excel_path = found or _find_excel_autodiscover()

    if not excel_path or not os.path.exists(excel_path):
        # no encontrado
        return None

    # Usar cache si es la misma ruta
    if _CACHED_CRONO["path"] == excel_path and _CACHED_CRONO["df"] is not None:
        return _CACHED_CRONO["df"]

    try:
        df = pd.read_excel(excel_path, sheet_name=sheet_name, engine=None)
    except Exception:
        # intenta sin sheet (lectura por defecto)
        try:
            df = pd.read_excel(excel_path, engine=None)
        except Exception:
            return None

    # normalizar nombres
    df.columns = [str(c).strip() for c in df.columns]

    # intentar identificar columnas de hacienda y suerte
    col_hacienda = _pick_column_by_tokens(df, ["cod hacienda", "cod. hacienda", "cod hda", "cod hda", "cod"])
    col_suerte = _pick_column_by_tokens(df, ["cod suerte", "cod. suerte", "cod_suerte", "ste", "suerte"])

    # fallbackes simples
    if col_hacienda is None:
        col_hacienda = _pick_column_by_tokens(df, ["cod", "codigo", "hacienda", "id"])
    if col_suerte is None:
        col_suerte = _pick_column_by_tokens(df, ["ste", "suerte", "cod", "id"])

    # crear columna CODIGO (como string combinado)
    if col_hacienda is not None and col_suerte is not None:
        # mantener comportamiento previo: zfill(3) + zfill(4)
        try:
            df['_cod_h_str'] = df[col_hacienda].astype(str).str.strip().replace('nan', '', regex=False).str.zfill(3)
        except Exception:
            df['_cod_h_str'] = df[col_hacienda].astype(str).str.strip().replace('nan', '', regex=False)
        try:
            df['_cod_s_str'] = df[col_suerte].astype(str).str.strip().replace('nan', '', regex=False).str.zfill(4)
        except Exception:
            df['_cod_s_str'] = df[col_suerte].astype(str).str.strip().replace('nan', '', regex=False)

        df['CODIGO'] = df['_cod_h_str'].fillna('') + df['_cod_s_str'].fillna('')
    else:
        # si no fue posible combinar, crear columna con None para mantener la API
        df['CODIGO'] = None

    # guardar cache
    _CACHED_CRONO["path"] = excel_path
    _CACHED_CRONO["df"] = df

    return df

# --------------------------
# Filtrar por código
# --------------------------
def find_by_codigo(df: pd.DataFrame, codigo) -> pd.DataFrame:
    """
    Filtra DataFrame por columna 'CODIGO' comparando como strings.
    Devuelve DataFrame (copia) o un DataFrame vacío si no aplica.
    """
    if df is None or 'CODIGO' not in df.columns:
        return pd.DataFrame()
    codigo = str(codigo)
    try:
        mask = df['CODIGO'].astype(str) == codigo
        return df.loc[mask].copy()
    except Exception:
        return pd.DataFrame()

# --------------------------
# Preparar DataFrame para mostrar en la UI
# --------------------------
def prepare_display_df(df_filtered: pd.DataFrame) -> pd.DataFrame:
    """
    Recibe el DataFrame filtrado (resultado de find_by_codigo) y devuelve
    un DataFrame con columnas ordenadas y renombradas para mostrar en la UI.
    Columnas de salida:
        ['Cod.','Hacienda','Ste','Edad','Variedad','NC','Fecha Liquidacion','TCHC','Area']
    """
    cols_template = ['Cod.', 'Hacienda', 'Ste', 'Edad', 'Variedad', 'NC', 'Fecha Liquidacion', 'TCHC', 'Area']

    if df_filtered is None or df_filtered.empty:
        return pd.DataFrame(columns=cols_template)

    # mapping tokens para búsqueda
    mapping_tokens = {
        'Cod.': ['cod. hacienda', 'cod hacienda', 'cod hda', 'cod', 'codigo'],
        'Hacienda': ['nombre hacienda', 'nombre del hacienda', 'nombre', 'hacienda'],
        'Ste': ['cod. suerte', 'cod suerte', 'cod_suerte', 'ste', 'suerte'],
        'Edad': ['edad en la cosecha', 'edad'],
        'Variedad': ['variedad'],
        'NC': ['numero de cortes', 'numero cortes', 'nc', 'n.c', 'n c'],
        'Fecha Liquidacion': ['fecha liquidacion', 'fecha', 'liquidacion'],
        'TCHC': ['tchc', 'tchm'],
        'Area': ['area total', 'area', 'ha', 'area_ha']
    }

    cols_original = list(df_filtered.columns)
    cols_lower = [str(c).lower() for c in cols_original]
    used = set()
    out = {}

    def _find_col(tokens):
        # usar _pick_column_by_tokens: intenta coincidencias completas
        candidate = _pick_column_by_tokens(df_filtered, tokens)
        if candidate and candidate not in used:
            return candidate
        # fallback: buscar por substring en nombres de columnas (no exacto)
        for t in tokens:
            tl = t.lower()
            for i, cname in enumerate(cols_lower):
                if cols_original[i] in used:
                    continue
                if tl in cname:
                    return cols_original[i]
        return None

    n = len(df_filtered)
    for out_name, tokens in mapping_tokens.items():
        col = _find_col(tokens)
        if col is not None:
            used.add(col)
            # mantener tipo como string para evitar problemas de mezcla de tipos
            vals = df_filtered[col].astype(str).str.strip().replace('nan', '', regex=False).values
            out[out_name] = vals
        else:
            out[out_name] = [''] * n

    df_display = pd.DataFrame(out)
    # intentar formatear TCH a numérico redondeado
    if 'TCHC' in df_display.columns:
        df_display['TCHC'] = pd.to_numeric(df_display['TCHC'], errors='coerce').round(2)

    # preservar índices del df_filtrado para trazabilidad en UI (si es necesario)
    df_display.index = df_filtered.index

    return df_display

# --------------------------
# Formato de presentación para labels
# --------------------------
def format_df_for_label(df_display: pd.DataFrame) -> str:
    """
    Convierte el DataFrame a string tabulado o devuelve 'Sin datos'.
    Útil para mostrar en labels pequeños de la UI.
    """
    if df_display is None or df_display.empty:
        return "Sin datos"
    return df_display.to_string(index=False)
