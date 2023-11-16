import re
import os
import requests
import numpy as np
import sys
from netCDF4 import Dataset
import xarray as xr

granules_dir = './granules' ## specify where granules are stored

# Function to process files and retain max values
def process_files(year):
    files = [filename for filename in os.listdir(granules_dir) if filename.endswith(f'WY{year}_{(year+1) % 100:02d}_SWE_SCA_POST.nc')]

    if not files:
        print(f"No files found for year {year}")

    allmaxlist = []
    for filename in files:
        dataset = xr.open_dataset(granules_dir+filename) 
        data = dataset['SWE_Post'][:,0,:,:] ## take the MEAN (5 layers of statistics)
        dataset.close()
        max_values = data.max(dim = 'Day')
        allmaxlist.append(max_values)
        
    combined_dataset = xr.concat(allmaxlist, dim='layer_name')
    combined_dataset = combined_dataset.rename({'Latitude': 'lat', 'Longitude': 'lon'})
    latitudes = combined_dataset['lat']
    longitudes = combined_dataset['lon']
    reindexed_data = combined_dataset.reindex(lat=latitudes, lon=longitudes, method='nearest')
    newdata = np.nanmax(reindexed_data, axis = 0)

    data_array = xr.DataArray(
        data= newdata,
        coords={"lon": longitudes, "lat": latitudes},
        dims=("lon", "lat"),
    )
    new_dataset = xr.Dataset({"maxSWE": data_array})
    output_filename = f'max_values_{year}_{len(files)}.nc'
    new_dataset.to_netcdf(output_filename)
        

    print(f"Processed and saved max values for year {year} to {output_filename} for {len(files)} files")
    #for filename in files:
    #    os.remove(filename)
    return



# Main
year = int(sys.argv[1])
process_files(year)
