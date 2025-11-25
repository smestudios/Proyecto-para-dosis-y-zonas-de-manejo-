#CTKTable.py
import customtkinter as ctk
import pandas as pd

class CTkTable(ctk.CTkFrame):
    def __init__(self, master, dataframe=pd.DataFrame(), corner_radius=10, **kwargs):
        super().__init__(master, corner_radius=corner_radius, **kwargs)
        self.dataframe = dataframe
        self.cells = []
        self.update_table(dataframe)

    def update_table(self, dataframe):
        # Borra lo anterior
        for widget in self.winfo_children():
            widget.destroy()
        self.cells.clear()
        self.dataframe = dataframe

        # Crear encabezados
        for i, col in enumerate(dataframe.columns):
            lbl = ctk.CTkLabel(self, text=col, font=("Arial", 12, "bold"))
            lbl.grid(row=0, column=i, padx=10, pady=5) 
            self.cells.append(lbl)

        # Llenar datos
        for r, row in enumerate(dataframe.values):
            for c, val in enumerate(row):
                lbl = ctk.CTkLabel(self, text=str(val), font=("Arial", 11))
                lbl.grid(row=r+1, column=c, padx=10, pady=5, sticky="nsew")
                self.cells.append(lbl)

