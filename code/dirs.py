#!/usr/bin/env python3
import os

## define directories
## home directory
basedir = '/pscratch/sd/c/cowherd/resilient-snowpack-estimation/'
## where the snotel data are stored
snoteldir = '/global/cfs/cdirs/m4099/fate-of-snotel/snoteldata/'
## downscaled GCM data
wrfdir = '/global/cfs/cdirs/m4099/fate-of-snotel/wrfdata/'
## downscaled GCM metadata
coorddir = wrfdir + 'meta/meta_new/'
## estimation model outputs
outputsdir = '/global/cfs/cdirs/m4099/fate-of-snotel/wrfdata/umital/Fig4_results/'
## where you have the ucla reanalysis data (Fang et al., 2022) downloaded
ucladir = '/global/cfs/cdirs/m4099/fate-of-snotel/fos-marianne/resilient-snowpack-estimation/data/uclaSR/'
## where you save preprocessed data
prepdir =  '/global/cfs/cdirs/m4099/fate-of-snotel/wrfdata/umital/Preprocessed_data/'

