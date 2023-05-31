import numpy as np 
import rasterio
from sklearn.cluster import KMeans
from osgeo import gdal
green = gdal.Open('/content/L3-NH43X11-096-051-23oct08-BAND2.tif').ReadAsArray()
red = gdal.Open('/content/L3-NH43X11-096-051-23oct08-BAND3.tif').ReadAsArray()
nir = gdal.Open('/content/L3-NH43X11-096-051-23oct08-BAND4.tif').ReadAsArray()
swir = gdal.Open('/content/L3-NH43X11-096-051-23oct08-BAND5.tif').ReadAsArray()

# Stack bands into a 3D array
data = np.dstack((green, red, nir, swir))
print(data.shape)

# Reshape for clustering
nrows, ncols, nbands = data.shape
liss3_reshape = data.reshape(nrows * ncols, nbands)

# Get profile of input image for writing output
with rasterio.open('/content/L3-NH43X11-096-051-23oct08-BAND2.tif') as src:
    profile = src.profile

# Normalize data
liss3_norm = (liss3_reshape - liss3_reshape.mean()) / liss3_reshape.std()

# Perform KMeans clustering
kmeans = KMeans(n_clusters=5, random_state=0).fit(liss3_norm)

# Reshape clustered labels back into an image
liss3_clustered = kmeans.labels_.reshape(nrows, ncols)

# Save clustered image
with rasterio.open("liss3_clustered_oct08_5.tif", "w", **profile) as dst:
    dst.write(liss3_clustered, 1)
