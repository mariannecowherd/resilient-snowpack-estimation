# resilient-snowpack-estimation code

## overview
1. Fork, clone, or download this github respository.
2. Create a conda environment `conda env create -f environment.yml`
3. Download the downscaled GCM solutions from Rahimi et al., 2023. (required)
4. If you want to reproduce the comparisons between the GCM solutions and the UCLA SR product, you will also need to download granules from Fang et al., 2022. (optional)
5. Download computed distributed peak SWE predictions from Zenodo [link] OR you may run the predictions code on your machine [see code below]

## code
The code folder contains python scripts and jupyter notebooks required to reproduce the analysis and figures. Main figure visualizations are labeled with the figure number and extended data figures are labeled as "ext." Other codes are labeled sequentially and with a descriptive phrase.
To run your own distributed SWE predictions, use the files labeled with the relevant method -- linreg, rf, or unet -- and save the ouputs locally

## data
The data folder contains some of the data required to reproduce the analysis and figures, including geospatial products which are available online. The actual downscaled GCM solutions are archived and available for download at [link to Rahim paper] with full explanation of their generation and structure in Rahimi et al., 202X. Intermediate reduced data products are generated from scripts in the `code` folder to streamline plotting and analysis and are referenced in `code/tmp/[filename]` in this repository. 

## figures
Saved main and extended data figures.
