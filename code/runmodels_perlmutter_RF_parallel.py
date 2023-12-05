## Objective
## Assemble predictors for various data combinations and train random forest models

import pandas as pd
import xarray as xr
import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from mpi4py import MPI

from dirs import prepdir

mpi_rank = MPI.COMM_WORLD.Get_rank()


meta_data = xr.open_dataset(prepdir + 'wrfinput_d02')
masks     = xr.open_dataset(prepdir + 'basin_masks_filtered.nc')
sweBC     = xr.open_dataarray(prepdir + 'snowmaxBC.nc')

snotel_extrap       = xr.open_dataarray(prepdir + 'snotel_extrapolated.nc')
cum_precip_all      = xr.open_dataarray(prepdir + 'cum_precipBC_SynthErr.nc')
cum_precip_snow_all = xr.open_dataarray(prepdir + 'cum_precip_snowBC_SynthErr.nc')
seasonal_t2_all     = xr.open_dataarray(prepdir + 'seasonal_t2BC_SynthErr.nc')
mean_fSCA_all       = xr.open_dataarray(prepdir + 'mean_fSCABC_SynthErr.nc')
pdd_sum_all         = xr.open_dataarray(prepdir + 'pdd_sumBC_SynthErr.nc')
aso_proxy_all       = xr.open_dataarray(prepdir + 'swe_apr1_SynthErr.nc')

## Read in static fields

lat_wrf    = meta_data.variables["XLAT"][0,:,:]
lon_wrf    = meta_data.variables["XLONG"][0,:,:]
z_wrf      = meta_data.variables["HGT"][0,:,:]
vgtyp_wrf  = meta_data.variables["IVGTYP"][0,:,:] ## Table 2: IGBP-Modified MODIS 20-category Land Use Categories
vegfra_wrf = meta_data.variables["VEGFRA"][0,:,:] ## Average canopy cover

lat_wrf    = xr.DataArray(lat_wrf, dims=["lat2d", "lon2d"])
lon_wrf    = xr.DataArray(lon_wrf, dims=["lat2d", "lon2d"])
z_wrf      = xr.DataArray(z_wrf, dims=["lat2d", "lon2d"])
vgtyp_wrf  = xr.DataArray(vgtyp_wrf, dims=["lat2d", "lon2d"]) 
vegfra_wrf = xr.DataArray(vegfra_wrf, dims=["lat2d", "lon2d"])

## Compute slope and aspect
myslopx, myslopy = np.gradient(z_wrf, 9000)
slope_wrf = np.degrees(np.arctan(np.sqrt(myslopx**2 + myslopy**2)))
aspect_wrf = np.degrees(np.arctan2(-myslopy,myslopx))
## Convert aspect to compass direction (clockwise from north)
aspect_q2 = (aspect_wrf > 90) & (aspect_wrf <= 180) ## [90, 180]
aspect_wrf = 90.0 - aspect_wrf
aspect_wrf[aspect_q2] = 360.0 + aspect_wrf[aspect_q2]


gcms = ['cesm2','mpi-esm1-2-lr','cnrm-esm2-1',
        'ec-earth3-veg','fgoals-g3','ukesm1-0-ll',
        'canesm5','access-cm2','ec-earth3',]


variants = ['r11i1p1f1','r7i1p1f1','r1i1p1f2',
            'r1i1p1f1','r1i1p1f1','r2i1p1f2',
            'r1i1p2f1','r5i1p1f1','r1i1p1f1',]

gcm_variants = [f'{item1}_{item2}_ssp370' for item1, item2 in zip(gcms, variants)]


def get_train_test_df (basin_id):
    basinmask = masks.basin_masks[basin_id]
    mask = basinmask.data.astype(bool)
    
    ## Location attributes
    lon_basin = lon_wrf.values[mask]
    lat_basin = lat_wrf.values[mask]

    ## Topography attributes
    z_basin = z_wrf.values[mask]
    slope_basin = slope_wrf[mask]
    aspect_basin = aspect_wrf[mask]

    ## Land-use attributes
    vgtyp_basin = vgtyp_wrf.values[mask]
    vegfra_basin = vegfra_wrf.values[mask]

    #### Create dataframes for training/testing

    Train_df = pd.DataFrame(data = {
                         'knn_snotel' : '',
                         'Longitude'  : np.repeat(lon_basin, train_length),
                         'Latitude'   : np.repeat(lat_basin, train_length),
                         'Elevation'  : np.repeat(z_basin, train_length),
                         'Slope'      : np.repeat(slope_basin, train_length),
                         'Aspect'     : np.repeat(aspect_basin, train_length),
                         'Veg-Type'   : np.repeat(vgtyp_basin, train_length),
                         'Veg-Frac'   : np.repeat(vegfra_basin, train_length),
                         'Cum-fSCA'   : '',
                         'Cum-precip' : '',
                         'Cum-snow'   : '',
                         'Mean-temp'  : '',
                         'PDD-sum'    : '',
                         'ASO-proxy'  : ''})


    Test_df = pd.DataFrame(data = {
                         'knn_snotel' : '',
                         'Longitude'  : lon_basin,
                         'Latitude'   : lat_basin,
                         'Elevation'  : z_basin,
                         'Slope'      : slope_basin,
                         'Aspect'     : aspect_basin,
                         'Veg-Type'   : vgtyp_basin,
                         'Veg-Frac'   : vegfra_basin,
                         'Cum-fSCA'   : '',
                         'Cum-precip' : '',
                         'Cum-snow'   : '',
                         'Mean-temp'  : '',
                         'PDD-sum'    : '',
                         'ASO-proxy'  : ''})
    
    return Train_df, Test_df



def get_dataarray_basin (basin_id, dataarray):
    basinmask = masks.basin_masks[basin_id]
    mask = basinmask.data.astype(bool)
    return dataarray.values[:,:,mask]

    
def get_dataarray_gcm_year (gcm_id, years, dataarray):
    return dataarray[gcm_id, years,:].flatten()

def get_KGE(labels, predictions):
    r = np.corrcoef(labels, predictions)[0][1]
    alpha = np.std(predictions)/np.std(labels)
    beta = np.mean(predictions)/np.mean(labels)
    ED = np.sqrt((r-1)**2 + (alpha-1)**2 + (beta-1)**2)
    return 1 - ED

train_length = 20
test_years = np.arange(2001, 2101, 5)
# test_years = np.arange(2001, 2101, 1) ### in case we want SWE predictions for every year

train_years = [range(item-train_length, item) for item in test_years]

Data_1 = ['knn_snotel', 'Longitude', 'Latitude']
Data_2 = Data_1 + ['Elevation', 'Slope', 'Aspect', 'Veg-Type', 'Veg-Frac']
Data_3 = Data_2 + ['Cum-fSCA']
Data_4 = Data_3 + ['Cum-precip', 'Cum-snow', 'Mean-temp', 'PDD-sum']
Data_5 = Data_4 + ['ASO-proxy']



## Specify folder path
dest_folder = 'Fig4_RF_SynthErr_preds/'

if not os.path.isdir(dest_folder):
    os.mkdir(dest_folder)
        
        
start = time.time()
### Not a very efficient parallelization, but it serves our purpose.
for basin_id in [mpi_rank]:

    basinname = masks.basin.values[basin_id]
    Train_df, Test_df = get_train_test_df (basin_id)

    knn_snotel_basin      = get_dataarray_basin (basin_id, snotel_extrap)
    cum_precip_basin      = get_dataarray_basin (basin_id, cum_precip_all)
    cum_precip_snow_basin = get_dataarray_basin (basin_id, cum_precip_snow_all)
    seasonal_t2_basin     = get_dataarray_basin (basin_id, seasonal_t2_all)
    pdd_sum_basin         = get_dataarray_basin (basin_id, pdd_sum_all)
    mean_fSCA_basin       = get_dataarray_basin (basin_id, mean_fSCA_all)
    aso_proxy_basin       = get_dataarray_basin (basin_id, aso_proxy_all)
    swe_basin             = get_dataarray_basin (basin_id, sweBC)

    for gcm_id in range(0,9):
        gcm = gcm_variants[gcm_id]
        for test_id in range(test_years.size):

            yrs = np.array(train_years[test_id])-1981
            Train_df['knn_snotel'] = get_dataarray_gcm_year (gcm_id, yrs, knn_snotel_basin)
            Train_df['Cum-precip'] = get_dataarray_gcm_year (gcm_id, yrs, cum_precip_basin)
            Train_df['Cum-snow']   = get_dataarray_gcm_year (gcm_id, yrs, cum_precip_snow_basin)
            Train_df['Mean-temp']  = get_dataarray_gcm_year (gcm_id, yrs, seasonal_t2_basin)
            Train_df['PDD-sum']    = get_dataarray_gcm_year (gcm_id, yrs, pdd_sum_basin)
            Train_df['Cum-fSCA']   = get_dataarray_gcm_year (gcm_id, yrs, mean_fSCA_basin)
            Train_df['ASO-proxy']   = get_dataarray_gcm_year (gcm_id, yrs, aso_proxy_basin)

            Test_df['knn_snotel'] = get_dataarray_gcm_year (gcm_id, test_years[test_id]-1981, knn_snotel_basin)
            Test_df['Cum-precip'] = get_dataarray_gcm_year (gcm_id, test_years[test_id]-1981, cum_precip_basin)
            Test_df['Cum-snow']   = get_dataarray_gcm_year (gcm_id, test_years[test_id]-1981, cum_precip_snow_basin)
            Test_df['Mean-temp']  = get_dataarray_gcm_year (gcm_id, test_years[test_id]-1981, seasonal_t2_basin)
            Test_df['PDD-sum']    = get_dataarray_gcm_year (gcm_id, test_years[test_id]-1981, pdd_sum_basin)
            Test_df['Cum-fSCA']   = get_dataarray_gcm_year (gcm_id, test_years[test_id]-1981, mean_fSCA_basin)
            Test_df['ASO-proxy']  = get_dataarray_gcm_year (gcm_id, test_years[test_id]-1981, aso_proxy_basin)
            
            Train_SWE = get_dataarray_gcm_year (gcm_id, yrs, swe_basin)
            Test_SWE = get_dataarray_gcm_year (gcm_id, test_years[test_id]-1981, swe_basin)
            for data_id, data_var in enumerate([Data_1, Data_2, Data_3, Data_4, Data_5]):
                
                if not os.path.isfile(dest_folder + f'{basin_id}_{basinname}_{gcm}_{test_years[test_id]}_{data_id}_RF.npy'):
                    model = RandomForestRegressor(n_estimators=500, min_samples_leaf=5, max_features=len(data_var)//3, n_jobs=12)
                    model.fit(Train_df[data_var], Train_SWE)

                    preds = model.predict(Test_df[data_var])
                    preds[preds < 0.0] = 0.0
                    
                    ## Create filepath properly. Assign a separate folder.
                    np.save(dest_folder + f'{basin_id}_{basinname}_{gcm}_{test_years[test_id]}_{data_id}_RF.npy', preds)
                
                
print (f"Time: {time.time()-start} seconds")
