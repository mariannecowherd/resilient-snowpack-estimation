## Objective
## Assemble predictors for various data combinations and train U-Net models

import pandas as pd
import xarray as xr
import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
import time
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Input, Conv2D, Conv2DTranspose
from tensorflow.keras.layers import BatchNormalization, MaxPooling2D, ReLU, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy.ma as ma
from scipy.interpolate import LinearNDInterpolator
import collections
from mpi4py import MPI

from dirs import prepdir

mpi_rank = MPI.COMM_WORLD.Get_rank()
mpi_size = MPI.COMM_WORLD.Get_size()



meta_data = xr.open_dataset(prepdir + 'wrfinput_d02')
masks     = xr.open_dataset(prepdir + 'basin_masks_filtered.nc')
sweBC     = xr.open_dataarray(prepdir + 'snowmaxBC.nc')

# snotel_extrap       = xr.open_dataarray(prepdir + 'snotel_extrapolated.nc')
snotel_extrap       = xr.open_dataarray('../snowpillow_extrapolated.nc')

cum_precip_all      = xr.open_dataarray(prepdir + 'cum_precipBC_SynthErr.nc')
cum_precip_snow_all = xr.open_dataarray(prepdir + 'cum_precip_snowBC_SynthErr.nc')
seasonal_t2_all     = xr.open_dataarray(prepdir + 'seasonal_t2BC_SynthErr.nc')
mean_fSCA_all       = xr.open_dataarray(prepdir + 'mean_fSCABC_SynthErr.nc')
pdd_sum_all         = xr.open_dataarray(prepdir + 'pdd_sumBC_SynthErr.nc')
aso_proxy_all       = xr.open_dataarray(prepdir + 'swe_apr1_SynthErr.nc')


### Normalize dynamic vars
snotel_extrap = np.log10(1+snotel_extrap)
cum_precip_all = np.log10(1+cum_precip_all)
cum_precip_snow_all = np.log10(1+cum_precip_snow_all)
pdd_sum_all = np.log10(1+pdd_sum_all)
aso_proxy_all = np.log10(1+aso_proxy_all)

t2_min, t2_max = 273.0, 283.0
seasonal_t2_all = (seasonal_t2_all - t2_min)/(t2_max - t2_min)

## Read in static fields

lat_wrf    = meta_data.variables["XLAT"][0,:,:]
lon_wrf    = meta_data.variables["XLONG"][0,:,:]
z_wrf      = meta_data.variables["HGT"][0,:,:]
vgtyp_wrf  = meta_data.variables["IVGTYP"][0,:,:] ## Table 2: IGBP-Modified MODIS 20-category Land Use Categories
vegfra_wrf = meta_data.variables["VEGFRA"][0,:,:] ## Average canopy cover

lat_wrf    = xr.DataArray(lat_wrf, dims=["lat2d", "lon2d"], name='Latitude')
lon_wrf    = xr.DataArray(lon_wrf, dims=["lat2d", "lon2d"], name='Longitude')
z_wrf      = xr.DataArray(z_wrf, dims=["lat2d", "lon2d"], name='Elevation')
vgtyp_wrf  = xr.DataArray(vgtyp_wrf, dims=["lat2d", "lon2d"], name='Veg-Type') 
vegfra_wrf = xr.DataArray(vegfra_wrf, dims=["lat2d", "lon2d"], name='Veg-Frac')

## Compute slope and aspect
myslopx, myslopy = np.gradient(z_wrf, 9000)
slope_wrf = np.degrees(np.arctan(np.sqrt(myslopx**2 + myslopy**2)))
aspect_wrf = np.degrees(np.arctan2(-myslopy,myslopx))
## Convert aspect to compass direction (clockwise from north)
aspect_q2 = (aspect_wrf > 90) & (aspect_wrf <= 180) ## [90, 180]
aspect_wrf = 90.0 - aspect_wrf
aspect_wrf[aspect_q2] = 360.0 + aspect_wrf[aspect_q2]

slope_wrf  = xr.DataArray(slope_wrf, dims=["lat2d", "lon2d"], name='Slope')
aspect_wrf = xr.DataArray(aspect_wrf, dims=["lat2d", "lon2d"], name='Aspect')



gcms = ['cesm2','mpi-esm1-2-lr','cnrm-esm2-1',
        'ec-earth3-veg','fgoals-g3','ukesm1-0-ll',
        'canesm5','access-cm2','ec-earth3',]


variants = ['r11i1p1f1','r7i1p1f1','r1i1p1f2',
            'r1i1p1f1','r1i1p1f1','r2i1p1f2',
            'r1i1p2f1','r5i1p1f1','r1i1p1f1',]

gcm_variants = [f'{item1}_{item2}_ssp370' for item1, item2 in zip(gcms, variants)]


def update_rect (basin_imin, basin_imax, basin_jmin, basin_jmax):
    """Update the bounding box around the basin such that 
    the dimensions are multiples of 8.
    """
    lat_extent = basin_imax - basin_imin
    lat_pad = int(8*(np.ceil(lat_extent/8) )) - lat_extent

    basin_imin = basin_imin - int(np.floor(lat_pad/2))
    basin_imax = basin_imax + int(np.ceil(lat_pad/2))

    lon_extent = basin_jmax - basin_jmin
    lon_pad = int(8*(np.ceil(lon_extent/8) )) - lon_extent

    basin_jmin = basin_jmin - int(np.floor(lon_pad/2))
    basin_jmax = basin_jmax + int(np.ceil(lon_pad/2))
    
    return basin_imin, basin_imax, basin_jmin, basin_jmax

def get_basin_rect(basin_id, update_shape=True):

    basinmask = masks.basin_masks[basin_id]
    mask = basinmask.data.astype(bool)

    ## Extract a rectangle encompassing the basin to maintain spatial structure
    basin_ii, basin_jj = np.nonzero(mask)
    basin_imin, basin_imax = basin_ii.min(), basin_ii.max()+1
    basin_jmin, basin_jmax = basin_jj.min(), basin_jj.max()+1
    
    if update_shape:
        return update_rect(basin_imin, basin_imax, basin_jmin, basin_jmax)
    
    return basin_imin, basin_imax, basin_jmin, basin_jmax

def get_normalized_staticvar (dataarray):
    min_value = np.min(dataarray)
    max_value = np.max(dataarray)
    
    return 2*(dataarray - min_value)/(max_value - min_value) - 1

def get_train_test_xr (basin_id, nan_mask):
    # basinmask = masks.basin_masks[basin_id]
    # mask = basinmask.data.astype(bool
    
    basin_imin, basin_imax, basin_jmin, basin_jmax = get_basin_rect(basin_id)
    
    ## Location attributes
    lon_basin = get_normalized_staticvar(lon_wrf[basin_imin:basin_imax,basin_jmin:basin_jmax])
    lat_basin = get_normalized_staticvar(lat_wrf[basin_imin:basin_imax,basin_jmin:basin_jmax])

    ## Topography attributes
    z_basin = get_normalized_staticvar(z_wrf[basin_imin:basin_imax,basin_jmin:basin_jmax])
    slope_basin = get_normalized_staticvar(slope_wrf[basin_imin:basin_imax,basin_jmin:basin_jmax])
    aspect_basin = get_normalized_staticvar(aspect_wrf[basin_imin:basin_imax,basin_jmin:basin_jmax])

    ## Land-use attributes
    vgtyp_basin = get_normalized_staticvar(vgtyp_wrf[basin_imin:basin_imax,basin_jmin:basin_jmax])
    vegfra_basin = get_normalized_staticvar(vegfra_wrf[basin_imin:basin_imax,basin_jmin:basin_jmax])
    
    ### Add in functionality to introduce gaps and impute?
    ### Use nan_mask to mask out 
    if nan_mask.sum() > 0:
        lon_basin    = impute_wrapper_2D(lon_basin, basin_id, nan_mask)
        lat_basin    = impute_wrapper_2D(lat_basin, basin_id, nan_mask)
        z_basin      = impute_wrapper_2D(z_basin, basin_id, nan_mask)
        slope_basin  = impute_wrapper_2D(slope_basin, basin_id, nan_mask)
        aspect_basin = impute_wrapper_2D(aspect_basin, basin_id, nan_mask)
        vgtyp_basin  = impute_wrapper_2D(vgtyp_basin, basin_id, nan_mask)
        vegfra_basin = impute_wrapper_2D(vegfra_basin, basin_id, nan_mask)
        
    
    
    Test_xr = xr.merge([xr.DataArray(np.full(lon_basin.shape, np.nan), dims=["lat2d", "lon2d"], name='knn_snotel'),
                        lon_basin, lat_basin, z_basin, slope_basin, aspect_basin, vgtyp_basin, vegfra_basin,
                        xr.DataArray(np.full(lon_basin.shape, np.nan), dims=["lat2d", "lon2d"], name='Cum-fSCA'),
                        xr.DataArray(np.full(lon_basin.shape, np.nan), dims=["lat2d", "lon2d"], name='Cum-precip'),
                        xr.DataArray(np.full(lon_basin.shape, np.nan), dims=["lat2d", "lon2d"], name='Cum-snow'),
                        xr.DataArray(np.full(lon_basin.shape, np.nan), dims=["lat2d", "lon2d"], name='Mean-temp'),
                        xr.DataArray(np.full(lon_basin.shape, np.nan), dims=["lat2d", "lon2d"], name='PDD-sum'),
                        xr.DataArray(np.full(lon_basin.shape, np.nan), dims=["lat2d", "lon2d"], name='ASO-proxy')
         ]).to_array(dim='channels').transpose("lat2d", "lon2d", "channels")
    
    Train_xr = Test_xr.expand_dims(dim={'time':train_length})

    ## expand_dims operation makes Train_xr read-only, so need to return a copy.
    return Train_xr.copy(), Test_xr

def get_dataarray_basin (basin_id, dataarray):
    basin_imin, basin_imax, basin_jmin, basin_jmax = get_basin_rect(basin_id)
    return dataarray.sel(lat2d=np.arange(basin_imin, basin_imax), lon2d=np.arange(basin_jmin, basin_jmax))    
    

def get_dataarray_gcm_year (gcm, years, dataarray, nan_mask):
    #### Imputation needs to happen here.
    if nan_mask.sum() > 0:
        raw_array = dataarray.sel(gcm=gcm, time=years)
        return impute_wrapper_3D (raw_array, basin_id, nan_mask)
    else:
        return dataarray.sel(gcm=gcm, time=years)
    

def extrapolate_by_edge_reflection (arr):
    
    """Extrapolate the gaps by reflecting data along the edges. 
    This will only work when array has height > width and gaps are at bottom right.
    If gaps are in a different corner or height < width, can simply flip the input array.
    Here, we assume that arr is filled with nans along the bottom right
    """

    
    arr = ma.array(arr, mask=np.isnan(arr))
    
    row_count = collections.Counter(np.argwhere(arr.mask)[:,0])
    col_count = collections.Counter(np.argwhere(arr.mask)[:,1])
    arr_rowpad = arr.copy()
    arr_colpad = arr.copy()
    nrows = arr.shape[0]
    ncols = arr.shape[1]
    arr_diagpad = arr.copy()
    
    ## Reflection along columns
    for key in col_count:
        val = col_count[key]
        arr_colpad[:, key] = np.pad(arr[:-val, key], (0, val), mode='reflect')
        
    ## Reflection along rows
    for key in row_count:
        val = row_count[key]
        arr_rowpad[key, :] = np.pad(arr[key, :-val], (0, val), mode='reflect')
        
    ## Reflection along diagonals

    diag_offsets = np.arange(-arr.shape[0]+1, arr.shape[1])
    diags = [(offset, arr.mask.diagonal(offset)) for offset in diag_offsets if np.sum(arr.mask.diagonal(offset)) > 0]
    for item in diags:
        row_ind = np.arange(-item[0], min(-item[0]+ncols, nrows))
        col_ind = np.arange(0, row_ind.size)
        npad = np.sum(item[1])
        arr_diagpad[row_ind, col_ind] = np.pad(arr[row_ind[:-npad], col_ind[:-npad]], (0, npad), mode='reflect')
    
    # Take the arithmetic mean of all three reflections and return
    return np.nanmean(np.dstack((arr_colpad,arr_rowpad, arr_diagpad)),2)

def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.
    https://stackoverflow.com/questions/6518811/interpolate-nan-values-in-a-numpy-array
    """

    return np.isnan(y), lambda z: (z.nonzero()[0], z.nonzero()[1])

def interpolate_col_refl_arr (dataarray):
    """Interpolate internal values using Linear interpolation. Thereafter, do
    column reflection along the left-most column only. This is something I have
    hard-coded for the current application only.
    """
    arr=dataarray.values
    nan_bools, nan_func = nan_helper(arr)
    interp = LinearNDInterpolator(nan_func(~nan_bools), arr[nan_func(~nan_bools)])
    arr[nan_func(nan_bools)] = interp(nan_func(nan_bools))
    
    arr = ma.array(arr, mask=np.isnan(arr))
    col_count = collections.Counter(np.argwhere(arr.mask)[:,1])
    arr_colpad = arr.copy()
    key = 0
    val = col_count[key]
    arr_colpad[:, key] = np.pad(arr[:-val, key], (0, val), mode='reflect')
    dataarray.values=arr_colpad
    ## Extrapolate the bottom-right area and return
    return extrapolate_by_edge_reflection(dataarray)


def impute_wrapper_2D (dataarray, basin_id, nan_mask):
    """This works only if the datarray is two-dimensional (lat/lon only). For three-dimensional, I have
    a separate wrapper. I have hard-coded exceptions for the current application. 
    """
    dataarray.values[nan_mask] = np.nan

    if basin_id in [13, 58, 65, 66, 67]:
        interp_arr = np.fliplr(extrapolate_by_edge_reflection(np.fliplr(dataarray)))
    elif basin_id in [64]:
        interp_arr = np.flipud(extrapolate_by_edge_reflection(np.flipud(dataarray.T))).T

    elif basin_id in [25, 61]:
        interp_arr = interpolate_col_refl_arr (dataarray)

    else:
        interp_arr = extrapolate_by_edge_reflection(dataarray)

    dataarray.values = interp_arr
    return dataarray
    
    

def impute_wrapper_3D (dataarray, basin_id, nan_mask):
    """This works only if the datarray is three-dimensional (time/lat/lon only) or 
    two-dimensional (lat/lon only).
    """
    
    if dataarray.ndim == 3:    
        for ind, dataarray_2D in enumerate(dataarray):
            dataarray_2D = impute_wrapper_2D(dataarray_2D, basin_id, nan_mask)
            dataarray[ind,:,:] = dataarray_2D
            
    else:
        dataarray = impute_wrapper_2D(dataarray, basin_id, nan_mask)
    
    return dataarray

#####################################################################################################################################
### The following three functions: EncoderMiniBlock, DecoderMiniBlock, and get_UNet_model are a modification of the code published here:
### https://github.com/VidushiBhatia/U-Net-Implementation/blob/main/U_Net_for_Image_Segmentation_From_Scratch_Using_TensorFlow_v4.ipynb
### These functions are subject to the following license:

# MIT License

# Copyright (c) 2021 Vidushi Bhatia

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

def EncoderMiniBlock(inputs, n_filters=64, max_pooling=True):
    """
    This block uses multiple convolution layers, max pool, relu activation to create an architecture for learning. 
    The block returns the activation values for next layer along with a skip connection which will be used in the decoder
    """

    conv = Conv2D(n_filters, 
                  3,   # Kernel size   
                  activation='relu',
                  padding='same')(inputs)
    conv = Conv2D(n_filters, 
                  3,   # Kernel size
                  activation='relu',
                  padding='same')(conv)
    

    if max_pooling:
        next_layer = tf.keras.layers.MaxPooling2D(pool_size = (2,2))(conv)    
    else:
        next_layer = conv

    skip_connection = conv
    
    return next_layer, skip_connection

def DecoderMiniBlock(prev_layer_input, skip_layer_input, n_filters=64, padding='same', strides=(2,2), kernel=(3,3)):
    """
    Decoder Block first uses transpose convolution to upscale the image to a bigger size and then,
    merges the result with skip layer results from encoder block
    Adding 2 convolutions with 'same' padding helps further increase the depth of the network for better predictions
    The function returns the decoded layer output
    """
    up = Conv2DTranspose(
                 n_filters,
                 kernel_size=kernel,    # Kernel size
                 strides=strides,
                 padding=padding)(prev_layer_input)

    merge = concatenate([up, skip_layer_input], axis=3)
    
    conv = Conv2D(n_filters, 
                 3,     # Kernel size
                 activation='relu',
                 padding='same')(merge)
    conv = Conv2D(n_filters,
                 3,   # Kernel size
                 activation='relu',
                 padding='same')(conv)
    return conv

def get_UNet_model(input_size):
    ## Clear session
    tf.keras.backend.clear_session()
    
    n_classes = 1
    n_filters=64
    inputs = Input(input_size)

    cblock1 = EncoderMiniBlock(inputs, n_filters, max_pooling=True)
    cblock2 = EncoderMiniBlock(cblock1[0],n_filters*2,max_pooling=True)
    cblock3 = EncoderMiniBlock(cblock2[0], n_filters*4, max_pooling=True)
    cblock4 = EncoderMiniBlock(cblock3[0], n_filters*8, max_pooling=False)

    ublock1 = DecoderMiniBlock(cblock4[0], cblock3[1],  n_filters * 4)
    ublock2 = DecoderMiniBlock(ublock1, cblock2[1],  n_filters * 2)
    ublock3 = DecoderMiniBlock(ublock2, cblock1[1],  n_filters)


    conv8 = Conv2D(n_filters,
                 3,
                 activation='relu',
                 padding='same')(ublock3)

    conv9 = Conv2D(n_classes, 1, padding='same')(conv8)

    # Define the model
    model = tf.keras.Model(inputs=inputs, outputs=conv10)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error', 
              metrics=[tf.keras.metrics.RootMeanSquaredError()])
    
    return model

#### End of the code under the above MIT license.
######################################################################################################################

early_stopping = EarlyStopping(monitor='val_loss', patience=15, verbose=1, min_delta=1e-8, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1, min_delta=1e-7, cooldown=1)


train_length = 20
test_years = np.arange(2001, 2101, 5)
train_years = [range(item-train_length, item) for item in test_years]

Data_1 = ['knn_snotel', 'Longitude', 'Latitude']
Data_2 = Data_1 + ['Elevation', 'Slope', 'Aspect', 'Veg-Type', 'Veg-Frac']
Data_3 = Data_2 + ['Cum-fSCA']
Data_4 = Data_3 + ['Cum-precip', 'Cum-snow', 'Mean-temp', 'PDD-sum']
Data_5 = Data_4 + ['ASO-proxy']


## Specify folder path

dest_folder = 'Fig4_Unet_preds/'

if not os.path.isdir(dest_folder):
    os.mkdir(dest_folder)
    
    
    
index=0
start = time.time()


for basin_id in range(masks.basin.size):

    basinname = masks.basin.values[basin_id]
    
    knn_snotel_basin      = get_dataarray_basin (basin_id, snotel_extrap)
    cum_precip_basin      = get_dataarray_basin (basin_id, cum_precip_all)
    cum_precip_snow_basin = get_dataarray_basin (basin_id, cum_precip_snow_all)
    seasonal_t2_basin     = get_dataarray_basin (basin_id, seasonal_t2_all)
    pdd_sum_basin         = get_dataarray_basin (basin_id, pdd_sum_all)
    mean_fSCA_basin       = get_dataarray_basin (basin_id, mean_fSCA_all)
    aso_proxy_basin       = get_dataarray_basin (basin_id, aso_proxy_all)
    swe_basin             = get_dataarray_basin (basin_id, sweBC)
    

    nan_mask = np.isnan(cum_precip_basin[0,0].values)
    Train_xr, Test_xr = get_train_test_xr (basin_id, nan_mask)
    
    for gcm_id in range(0,9):
        gcm = gcm_variants[gcm_id]
        print (f"Modeling {basin_id}_{basinname}_{gcm}")
        for test_id in range(test_years.size):
                    
            if index%mpi_size == mpi_rank:
        
                yrs = np.array(train_years[test_id])
                Train_xr = Train_xr.assign_coords(time=yrs)
                Train_xr.loc[dict(channels='knn_snotel')] = get_dataarray_gcm_year (gcm, yrs, knn_snotel_basin, nan_mask)
                Train_xr.loc[dict(channels='Cum-precip')] = get_dataarray_gcm_year (gcm, yrs, cum_precip_basin, nan_mask)
                Train_xr.loc[dict(channels='Cum-snow')]   = get_dataarray_gcm_year (gcm, yrs, cum_precip_snow_basin, nan_mask)
                Train_xr.loc[dict(channels='Mean-temp')]  = get_dataarray_gcm_year (gcm, yrs, seasonal_t2_basin, nan_mask)
                Train_xr.loc[dict(channels='PDD-sum')]    = get_dataarray_gcm_year (gcm, yrs, pdd_sum_basin, nan_mask)
                Train_xr.loc[dict(channels='Cum-fSCA')]   = get_dataarray_gcm_year (gcm, yrs, mean_fSCA_basin, nan_mask)
                Train_xr.loc[dict(channels='ASO-proxy')]  = get_dataarray_gcm_year (gcm, yrs, aso_proxy_basin, nan_mask)


                Test_xr.loc[dict(channels='knn_snotel')] = get_dataarray_gcm_year (gcm, test_years[test_id], knn_snotel_basin, nan_mask)
                Test_xr.loc[dict(channels='Cum-precip')] = get_dataarray_gcm_year (gcm, test_years[test_id], cum_precip_basin, nan_mask)
                Test_xr.loc[dict(channels='Cum-snow')]   = get_dataarray_gcm_year (gcm, test_years[test_id], cum_precip_snow_basin, nan_mask)
                Test_xr.loc[dict(channels='Mean-temp')]  = get_dataarray_gcm_year (gcm, test_years[test_id], seasonal_t2_basin, nan_mask)
                Test_xr.loc[dict(channels='PDD-sum')]    = get_dataarray_gcm_year (gcm, test_years[test_id], pdd_sum_basin, nan_mask)
                Test_xr.loc[dict(channels='Cum-fSCA')]   = get_dataarray_gcm_year (gcm, test_years[test_id], mean_fSCA_basin, nan_mask)
                Test_xr.loc[dict(channels='ASO-proxy')]  = get_dataarray_gcm_year (gcm, test_years[test_id], aso_proxy_basin, nan_mask)


                Train_SWE = get_dataarray_gcm_year (gcm, yrs, swe_basin, nan_mask).expand_dims(dim='channels', axis=-1)
                Train_SWE = np.log10(1+Train_SWE)
                Test_SWE = get_dataarray_gcm_year (gcm, test_years[test_id], swe_basin, nan_mask).expand_dims(dim='channels', axis=-1)

                for data_id, data_var in enumerate([Data_1, Data_2, Data_3, Data_4, Data_5]):
                    ## Sometimes U-Net weights are not initialized optimally which can affect the results. When that happens,
                    ## the model loss doesn't improve and model terminates after < 20 epochs. This while loop checks
                    ## for early termination and forces a redo if fewer than 20 epochs were used.
                    if not os.path.isfile(dest_folder + f'{basin_id}_{basinname}_{gcm}_{test_years[test_id]}_{data_id}_Unet.npy'):
                        while True:

                            model = get_UNet_model((*Test_xr.shape[:2], len(data_var)))
                            history = model.fit(Train_xr.sel(channels=data_var).values, Train_SWE.values, 
                                                epochs=500, validation_split=0.2,callbacks=[early_stopping, reduce_lr], verbose=0)
                            if len(history.epoch) < 25:
                                continue
                            else:
                                preds = model(np.expand_dims(Test_xr.sel(channels=data_var).values, axis=0))[0,...,0]
                                preds = 10**preds.numpy()-1
                                preds[preds < 0.0] = 0.0
                                np.save(dest_folder + f'{basin_id}_{basinname}_{gcm}_{test_years[test_id]}_{data_id}_Unet.npy', preds)
                                break
            index=index+1
            
print (f"Time: {time.time()-start} seconds")
