# ACCELERATED HYDRATION SITE LOCALIZATION AND THERMODYNAMIC PROFILING

This is the code supporting our work ''ACCELERATED HYDRATION SITE LOCALIZATION AND THERMODYNAMIC PROFILING''. We predict location, entropy and enthalpy of high occupancy hydration sites of proteins.

<div style="text-align: right;">
    <img src="documentation/images/predictions/prediction_cropped.gif" alt="Animation GIF"  />
</div>


## Data sets

The data for training and test can be downloaded [here](https://zenodo.org/records/14182834).



## Installation

The conda environment can be created by running

```
conda env create -f environment.yml
```

For faster installation we recommend using mamba instead:

```
mamba env create -f environment.yml
```


## Training

### Location model

To train the model for hydration site location prediction, execute the command

```
python -m training.train
```

Training can be performed on multiple GPUs by changing the line ```cuda_ids: [0]``` in the config file ```config/location_model.yaml```.

### Thermodynamics model

A separate model was trained for predicting thermodynamic properties (i.e. enthalpy and entropy). To train this model, switch the ```config_name``` in

```
@hydra.main(config_path="../config/", config_name="location_model", version_base="1.1")
```

to ```thermo_model``` in the training script ```training/train.py```.
The execute

```
python -m training.train
```
## Evaluation and visualization

### Location model

After training, the model performance can be evaluated:

```
python -m inference.evaluation.evaluate_after_clustering
```

Loading our pretrained model checkpoint, we obtain the following results:


<div align="center">

| Cutoff (Angstrom) | Ground Truth Revory Rate | Prediction Hit Rate |
| :---: |:--------:|:---------:|
| 0.5 |    59.0\%   |   48.3\%    |
| 1.0 |   80.2\%   |   65.9\%    |
</div>

The ground truth recovery rate can be further investigated by differentiating based on occupancy:

<div align="center">

| Cutoff (Angstrom) | [0.5,0.6]|[0.6,0.7] | [0.7,0.8] |[0.8,0.9] |[0.9,1.0] |
| :---: |:--------:|:---------:|:---------:|:---------:|:---------:|
| 0.5 | 42.0\% |57.2\%|65.1\%|70.0\%|69.7\% |
| 1.0 |   62.3\%   | 79.4\%|89.0\%|91.0\%|90.8\%|

</div>

If we restrict ourselves to the first layer of hydration sites (distance from non-hydrogen atoms no further than $3.5$ &Aring;), then we obtain the following resuluts:

<div align="center">

| Cutoff (Angstrom) | Ground Truth Revory Rate |
| :---: |:--------:|
| 0.5 |    62.5\%  |
| 1.0 |   84.5\%   |
</div>


<div align="center">

| Cutoff (Angstrom) | [0.5,0.6]  | [0.6,0.7]  | [0.7,0.8]  | [0.8,0.9]  | [0.9,1.0]  |
| :---:             | :--------: | :--------: | :--------: | :--------: | :--------: |
| 0.5               | 48.5\%     | 60.6\%     | 67.0\%     | 70.9\%     | 69.9\%     |
| 1.0               | 71.0\%     | 83.6\%     | 88.9\%     | 91.9\%     | 90.9\%     |

</div>

The second layer water hydration sites (distance from protein non-hydrogen atoms at least $3.5$ &Aring;) are more challenging to predict, accordingly the results are substantially worse:

<div align="center">

| Cutoff (Angstrom) | Ground Truth Revory Rate |
| :---: |:--------:|
| 0.5 |    15.2\%  |
| 1.0 |   26.9\%   |
</div>


<div align="center">

| Cutoff (Angstrom) | [0.5,0.6]  | [0.6,0.7]  | [0.7,0.8]  | [0.8,0.9]  | [0.9,1.0]  |
| :---:             | :--------: | :--------: | :--------: | :--------: | :--------: |
| 0.5               | 11.4\%     | 17.6\%     | 22.8\%     | 29.6\%     | 38.7\%     |
| 1.0               | 20.8\%     | 30.6\%     | 39.8\%     | 50.7\%     | 63.5\%     |

</div>


### Thermodynamics model

We evaluate the prediction peformance for enthalpy and entropy by investigating the correlation of the predictions with the simulated ground truth. Running the following command prints Pearson correlation and creates density plots:

```
python -m inference.visualization.thermodynamics.create_correlation_plots
```

<div style="display: flex; justify-content: center; align-items: center;">
    <img src="documentation/images/enthalpy_valid.svg" alt="Correlation plot of enthalpy" style="width: 49%; margin-right: 10px;" />
    <img src="documentation/images/entropy_valid.svg" alt="Correlation plot of entropy" style="width: 49%;" />
</div>

The correlation between predictions and ground truth is given the following table:
<div align="center">

| **Predicted Variable** | **Pearson r Correlation with Ground Truth** |
|:----------------------:|:----------------------------------------:|
|        Enthalpy        |                 0.8388                   |
|        Entropy         |                 0.8643                   |


</div>

## Desolvation free energies vs experimental binding affinities
For a protein-ligand binding, the ligand displaces the water molecules in the binding pocket. The desolvation free energy can be calculated as

$$
 \Delta G = \sum_{i=1}^{n} [  \Delta H_{i} - T  \Delta S_{i}],  \quad \text{(1)}
$$

for $i \in \{1,...,n\}$ displaced waters with corresponding entthalpy $H_{i}$ and entropy $S_{i}$.

We apply our model to predict the hydration sites within a protein binding pocket and calculate the Gibbs free energy $G$ by displacement (see (1)) for a range of different ligands. The results are plotted against experimental binding affinities. If our model is correct we should observe an affine linear relationship. To execute the corresponding code, first change in both config files ```config/location_model.yaml``` and ```config/thermo_model.yaml``` the ```data``` parameter to ```data: case_study```.
Then execute the script

```
python -m inference.visualization.calculate_displaced_waters
```
The highest Pearson correlation of $0.931$ is found for a water dispacement tolerance distance of $2.4$ &Aring;. Note that we ignored hydrogens (both for the hydration sites and and the protein) in our model, which justifies this cutoff. 


<div style="display: flex; justify-content: center; align-items: center;"> <img src="documentation/images/prediction_vs_experiment_correlation_displacement_2.4000000953674316A.svg" alt="Correlation plot of enthalpy" style="width: 100%; margin-right: 10px;" /> </div>


## Case studies

In order to predict water molecules with associated enthalpy and entropy, set both in ```config/location_model.yaml``` and ```config/thermo_model.yaml``` the data entry to the data set of interest:

 ```- data: case_study```.

 Then run
```python -m inference.evaluation.predict_waters```.

The predicted location, entropy and enthalpy e.g. for the protein ```1I06``` will be saved at

```
images/case_study/1I06/enthalpy.pt
images/case_study/1I06/entropy.pt
images/case_study/1I06/location_prediction.pt
```

A pymol visualization is provided at

```images/case_study/1I06/ protein_with_predictions.pse```.

