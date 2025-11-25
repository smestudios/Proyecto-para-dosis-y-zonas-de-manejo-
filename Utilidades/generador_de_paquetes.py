import pkg_resources
 
paquetes = ['numpy', 'pandas','geopandas','rasterio','seaborn','libpysal','matplotlib','scipy','sklearn','esda','tkinter','shapely','pyproj']
for paquete in paquetes:
    try:
        version = pkg_resources.get_distribution(paquete).version
        print(f"{paquete}=={version}")
    except pkg_resources.DistributionNotFound:
        print(f"{paquete} no está instalado")