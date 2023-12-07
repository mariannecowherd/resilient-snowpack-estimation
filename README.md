# resilient-snowpack-estimation code

## overview
1. Fork, clone, or download this github respository.
2. Create a conda environment `conda env create -f environment.yml`
3. Download the downscaled GCM solutions from Rahimi et al., 2023. (required)
4. If you want to reproduce the comparisons between the GCM solutions and the UCLA SR product, you will also need to download granules from Fang et al., 2022. (optional)
5. Download computed distributed peak SWE predictions from Dryad [link] OR you may run the predictions code on your machine [see code below]

## code
The code folder contains python scripts and jupyter notebooks required to reproduce the analysis and figures. Main figure visualizations are labeled with the figure number and extended data figures are labeled as "ext." Other codes are labeled sequentially and with a descriptive phrase.
To run your own distributed SWE predictions, use the files labeled with the relevant method -- linear regression, random forest, or U-Net -- and save the ouputs locally

### Notebooks
`generate_tmpfiles.ipynb`: requires WUS-D3 and WUS-SR data \\ 
`swe_Apr1_synthetic_error.ipynb`: requires WUS-D3 data; produces preprocessed synthetic error SWE observations for data group 3
`Compare_GCM_PRISM.ipynb`: requires PRISM and WUS-D3 data; produces data group 2 and 3 input data
`Combine_preds_Unet.ipynb`: requires outputs of `runmodels*`; produces 
`Accuum_fSCa_synthetic_error.ipynb` requires WUS-D3 data; prodcues data group 2 and 3 fSCA forcing data
`runmodels_perlmutter_LR.ipynb`: requires pre-processed data; runs linear regression models  
`fig1_fos-map.ipynb`: requires WUS-D3 data and tmp files; produces figure 1
`fig2_CDF_patterns_disappearance.ipynb`: requires WUS-D3 data and tmp files; produces figure 2
`fig3_error_map.ipynb`: requires outputs of all `runmodels*` and `Combine_preds_Unet.ipynb` ; produces figure 3
`fig4_compare_model_outputs.ipynb`: requires outputs of all `runmodels*` and `Combine_preds_Unet.ipynb`; produces figure 4
`ext_prism.ipynb`: requires the outputs of `Accuum_fSCa_synthetic_error.ipynb` and `swe_Apr1_synthetic_error.ipynb`; produces extended data figure 2
`ext_pattern.ipynb`: requires tmp files, creates nondimensionalized snow maps; produces extended data figure 3
`ext_climatology.ipynb`: requires SNOTEL, WUS-SR, and WUS-D3 data downloaded and pre-processed with `generate_tmpfiles.ipynb`; produces extended data figure 1

### Python scripts
`dirs.py`: list of paths used in the project. Must be updated by the user to reflect your workspace.
`myutils.py`: holds functions used in several other scripts or notebooks
`runmodels_Unet_parallel.py`: requires preprocessed data; runs U-Nets
`runmodels_Unet_parallel_impute.py`: requires preprocessed data; runs U-Nets with imputation
`runmodels_perlmutter_RF_parallel.py`: requires preprocessed data, runs random forests
`max_UCLA.py`: requires the WUS-SR dataset downloaded and produces annual pixel-wise maxima

## data
The data folder contains some of the data required to reproduce the analysis and figures, including geospatial products which are available online. The actual downscaled GCM solutions are archived and available for download at [link to Rahimi paper] with full explanation of their generation and structure in Rahimi et al., 202X. Intermediate reduced data products are generated from scripts in the `code` folder to streamline plotting and analysis and are referenced in `code/tmp/[filename]` in this repository. 

## figures
Saved main and extended data figures.
