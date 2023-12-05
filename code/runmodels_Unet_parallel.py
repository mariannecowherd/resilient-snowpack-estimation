import xarray as xr
import os
import numpy as np
import numpy.ma as ma
import time
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input, Conv2D, Conv2DTranspose
from tensorflow.keras.layers import MaxPooling2D, Dropout, ReLU, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import mean_squared_error, r2_score
from mpi4py import MPI

mpi_rank = MPI.COMM_WORLD.Get_rank()
mpi_size = MPI.COMM_WORLD.Get_size()


### Specify paths to xarrays
# input_folder_path = '/pscratch/sd/u/umital/fate-of-snotels/Preprocessed_data/'
input_folder_path = '/global/cfs/cdirs/m4099/fate-of-snotel/wrfdata/umital/Preprocessed_data/'

meta_data = xr.open_dataset(input_folder_path + 'wrfinput_d02')
masks     = xr.open_dataset(input_folder_path + 'basin_masks_filtered.nc')
sweBC     = xr.open_dataarray(input_folder_path + 'snowmaxBC.nc')

snotel_extrap       = xr.open_dataarray(input_folder_path + 'snotel_extrapolated.nc')
cum_precip_all      = xr.open_dataarray(input_folder_path + 'cum_precipBC_SynthErr.nc')
cum_precip_snow_all = xr.open_dataarray(input_folder_path + 'cum_precip_snowBC_SynthErr.nc')
# cum_precip_all      = xr.open_dataarray(input_folder_path + 'cum_precipBC_SynthErrReduced.nc')
# cum_precip_snow_all = xr.open_dataarray(input_folder_path + 'cum_precip_snowBC_SynthErrReduced.nc')
seasonal_t2_all     = xr.open_dataarray(input_folder_path + 'seasonal_t2BC_SynthErr.nc')
mean_fSCA_all       = xr.open_dataarray(input_folder_path + 'mean_fSCABC_SynthErr.nc')
pdd_sum_all         = xr.open_dataarray(input_folder_path + 'pdd_sumBC_SynthErr.nc')
aso_proxy_all       = xr.open_dataarray(input_folder_path + 'swe_apr1_SynthErr.nc')

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

def get_train_test_xr (basin_id):
    
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

def get_dataarray_gcm_year (gcm_id, years, dataarray):
    return dataarray.sel(gcm=gcm, time=years)

def EncoderMiniBlock(inputs, n_filters=64, dropout_prob=0.0, max_pooling=True):
    """
    This block uses multiple convolution layers, max pool, relu activation to create an architecture for learning. 
    Dropout can be added for regularization to prevent overfitting. 
    The block returns the activation values for next layer along with a skip connection which will be used in the decoder
    """
    # Add 2 Conv Layers with relu activation 
    # 'Same' padding will pad the input to conv layer such that the output has the same height and width (hence, is not reduced in size) 
    conv = Conv2D(n_filters, 
                  3,   # Kernel size   
                  activation='relu',
                  padding='same')(inputs)
    conv = Conv2D(n_filters, 
                  3,   # Kernel size
                  activation='relu',
                  padding='same')(conv)
    

    # In case of overfitting, dropout will regularize the loss and gradient computation to shrink the influence of weights on output
    if dropout_prob > 0:     
        conv = tf.keras.layers.Dropout(dropout_prob)(conv)

    # Pooling reduces the size of the image while keeping the number of channels same
    # Pooling has been kept as optional as the last encoder layer does not use pooling (hence, makes the encoder block flexible to use)
    # Below, Max pooling considers the maximum of the input slice for output computation and uses stride of 2 to traverse across input image
    if max_pooling:
        next_layer = tf.keras.layers.MaxPooling2D(pool_size = (2,2))(conv)    
    else:
        next_layer = conv

    # skip connection (without max pooling) will be input to the decoder layer to prevent information loss during transpose convolutions      
    skip_connection = conv
    
    return next_layer, skip_connection

def DecoderMiniBlock(prev_layer_input, skip_layer_input, n_filters=64, padding='same', strides=(2,2), kernel=(3,3)):
    """
    Decoder Block first uses transpose convolution to upscale the image to a bigger size and then,
    merges the result with skip layer results from encoder block
    Adding 2 convolutions with 'same' padding helps further increase the depth of the network for better predictions
    The function returns the decoded layer output
    """
    # Start with a transpose convolution layer to first increase the size of the image
    up = Conv2DTranspose(
                 n_filters,
                 kernel_size=kernel,    # Kernel size
                 strides=strides,
                 padding=padding)(prev_layer_input)

    # Merge the skip connection from previous block to prevent information loss
    merge = concatenate([up, skip_layer_input], axis=3)
    
    # Add 2 Conv Layers with relu activation
    # The parameters for the function are similar to encoder
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
    
    # input_size = (24,24,14) ## This is an example shape
    n_classes = 1
    n_filters=64
    inputs = Input(input_size)

    cblock1 = EncoderMiniBlock(inputs, n_filters,dropout_prob=0, max_pooling=True)
    cblock2 = EncoderMiniBlock(cblock1[0],n_filters*2,dropout_prob=0, max_pooling=True)
    cblock3 = EncoderMiniBlock(cblock2[0], n_filters*4,dropout_prob=0, max_pooling=True)
    cblock4 = EncoderMiniBlock(cblock3[0], n_filters*8,dropout_prob=0, max_pooling=False)


    # Decoder includes multiple mini blocks with decreasing number of filters
    # Observe the skip connections from the encoder are given as input to the decoder
    # Recall the 2nd output of encoder block was skip connection, hence cblockn[1] is used
    ublock1 = DecoderMiniBlock(cblock4[0], cblock3[1],  n_filters * 4)
    ublock2 = DecoderMiniBlock(ublock1, cblock2[1],  n_filters * 2)
    ublock3 = DecoderMiniBlock(ublock2, cblock1[1],  n_filters)


    # Complete the model with 1 3x3 convolution layer (Same as the prev Conv Layers)
    # Followed by a 1x1 Conv layer to get the image to the desired size. 
    # Observe the number of channels will be equal to number of output classes
    conv9 = Conv2D(n_filters,
                 3,
                 activation='relu',
                 padding='same')(ublock3)

    conv10 = Conv2D(n_classes, 1, padding='same')(conv9)

    # Define the model
    model = tf.keras.Model(inputs=inputs, outputs=conv10)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error', 
              metrics=[tf.keras.metrics.RootMeanSquaredError()])
    
    return model



early_stopping = EarlyStopping(monitor='val_loss', patience=15, verbose=1, min_delta=1e-8, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1, min_delta=1e-7, cooldown=1)


train_length = 20
test_years = np.arange(2001, 2101, 5)
# test_years = np.arange(2001, 2101)
train_years = [range(item-train_length, item) for item in test_years]

Data_1 = ['knn_snotel', 'Longitude', 'Latitude']
Data_2 = Data_1 + ['Elevation', 'Slope', 'Aspect', 'Veg-Type', 'Veg-Frac']
Data_3 = Data_2 + ['Cum-fSCA']
Data_4 = Data_3 + ['Cum-precip', 'Cum-snow', 'Mean-temp', 'PDD-sum']
Data_5 = Data_4 + ['ASO-proxy']


## Specify folder path

# dest_folder = '/pscratch/sd/u/umital/fate-of-snotels_output/Fig4_Unet_preds_test/'
dest_folder = '/global/cfs/cdirs/m4099/fate-of-snotel/wrfdata/umital/Fig4_results/Fig4_Unet_preds_test/'

if not os.path.isdir(dest_folder):
    os.mkdir(dest_folder)
    
index=0
start = time.time()


for basin_id in range(0,1):
# for basin_id in range(masks.basin.size):

### Will loop over basin_id

    basinname = masks.basin.values[basin_id]
    Train_xr, Test_xr = get_train_test_xr (basin_id)

    ### Take subset of met and swe data here

    knn_snotel_basin      = get_dataarray_basin (basin_id, snotel_extrap)
    cum_precip_basin      = get_dataarray_basin (basin_id, cum_precip_all)
    cum_precip_snow_basin = get_dataarray_basin (basin_id, cum_precip_snow_all)
    seasonal_t2_basin     = get_dataarray_basin (basin_id, seasonal_t2_all)
    pdd_sum_basin         = get_dataarray_basin (basin_id, pdd_sum_all)
    mean_fSCA_basin       = get_dataarray_basin (basin_id, mean_fSCA_all)
    aso_proxy_basin       = get_dataarray_basin (basin_id, aso_proxy_all)
    swe_basin             = get_dataarray_basin (basin_id, sweBC)

    for gcm_id in range(0,9):
    ### Will loop over gcm_id, this will be nested inside basin_id loop
        gcm = gcm_variants[gcm_id]
        print (f"Modeling {basin_id}_{basinname}_{gcm}")
        for test_id in range(test_years.size):
        # for test_id in range(1,2):
        ### Will loop over test_id, this will be nested inside gcm_id loop
            
            yrs = np.array(train_years[test_id])
            Train_xr = Train_xr.assign_coords(time=yrs)
            Train_xr.loc[dict(channels='knn_snotel')] = get_dataarray_gcm_year (gcm_id, yrs, knn_snotel_basin)
            Train_xr.loc[dict(channels='Cum-precip')] = get_dataarray_gcm_year (gcm_id, yrs, cum_precip_basin)
            Train_xr.loc[dict(channels='Cum-snow')]   = get_dataarray_gcm_year (gcm_id, yrs, cum_precip_snow_basin)
            Train_xr.loc[dict(channels='Mean-temp')]  = get_dataarray_gcm_year (gcm_id, yrs, seasonal_t2_basin)
            Train_xr.loc[dict(channels='PDD-sum')]    = get_dataarray_gcm_year (gcm_id, yrs, pdd_sum_basin)
            Train_xr.loc[dict(channels='Cum-fSCA')]   = get_dataarray_gcm_year (gcm_id, yrs, mean_fSCA_basin)
            Train_xr.loc[dict(channels='ASO-proxy')]  = get_dataarray_gcm_year (gcm_id, yrs, aso_proxy_basin)
            
            
            Test_xr.loc[dict(channels='knn_snotel')] = get_dataarray_gcm_year (gcm_id, test_years[test_id], knn_snotel_basin)
            Test_xr.loc[dict(channels='Cum-precip')] = get_dataarray_gcm_year (gcm_id, test_years[test_id], cum_precip_basin)
            Test_xr.loc[dict(channels='Cum-snow')]   = get_dataarray_gcm_year (gcm_id, test_years[test_id], cum_precip_snow_basin)
            Test_xr.loc[dict(channels='Mean-temp')]  = get_dataarray_gcm_year (gcm_id, test_years[test_id], seasonal_t2_basin)
            Test_xr.loc[dict(channels='PDD-sum')]    = get_dataarray_gcm_year (gcm_id, test_years[test_id], pdd_sum_basin)
            Test_xr.loc[dict(channels='Cum-fSCA')]   = get_dataarray_gcm_year (gcm_id, test_years[test_id], mean_fSCA_basin)
            Test_xr.loc[dict(channels='ASO-proxy')]  = get_dataarray_gcm_year (gcm_id, test_years[test_id], aso_proxy_basin)
            
            Train_SWE = get_dataarray_gcm_year (gcm_id, yrs, swe_basin).expand_dims(dim='channels', axis=-1)
            Train_SWE = np.log10(1+Train_SWE)
            Test_SWE = get_dataarray_gcm_year (gcm_id, test_years[test_id], swe_basin).expand_dims(dim='channels', axis=-1)
            
            # for data_id, data_var in enumerate([Data_2]):
            for data_id, data_var in enumerate([Data_1, Data_2, Data_3, Data_4, Data_5]):
                ## Sometimes U-Net weights are not initialized optimally which can affect the results. When that happens,
                ## the model loss doesn't improve and model terminates after < 20 epochs. This while loop checks
                ## for early termination and forces a redo if fewer than 20 epochs were used.
                if not os.path.isfile(dest_folder + f'{basin_id}_{basinname}_{gcm}_{test_years[test_id]}_{data_id}_Unet.npy'):
                    while index%mpi_size == mpi_rank:
                        
                        model = get_UNet_model((*Test_xr.shape[:2], len(data_var)))
                        history = model.fit(Train_xr.sel(channels=data_var).values, Train_SWE.values, 
                                            epochs=500, validation_split=0.2,callbacks=[early_stopping, reduce_lr], verbose=0)
                        if len(history.epoch) < 25:
                            continue
                        else:
                            preds = model.predict(np.expand_dims(Test_xr.sel(channels=data_var).values, axis=0))[0,...,0]
                            preds = 10**preds-1
                            preds[preds < 0.0] = 0.0
                            print (f'{basin_id}_{basinname}_{gcm}_{test_years[test_id]}_{data_id} r2: {r2_score(Test_SWE[:,:,0], preds)}')
                            np.save(dest_folder + f'{basin_id}_{basinname}_{gcm}_{test_years[test_id]}_{data_id}_Unet.npy', preds)
                            break
                index=index+1

print (f"Time: {time.time()-start} seconds")