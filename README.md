# Supposedly Fair Classification Systems and Their Impacts - Loan Repayment Use Case Implementation


"Implementation of the Dataset Generatio for the Loan Repayment use case, descirbed in..."

We owe a great deal to Liu et al.'s work, [*Delayed Impact of Fair Machine Learning*](https://arxiv.org/abs/1803.04383). We extended their [code](https://github.com/lydiatliu/delayedimpact) here generate synthetic datasets from it. 


**Data**:
Our simulated,synthetic datasets are based on Hardt et al.'s 2016 dataset. 
- Download the data folder from the Github repository for [fairmlbook](https://github.com/fairmlbook/fairmlbook.github.io/tree/master/code/creditscore) (Barocas, Hardt and Narayanan 2018)
- Save it to the root directory of this repository (csvs should be in the folder 'data')
- Then run: ```delayedimpact/Liu_paper_code/FICO-figures.ipynb```

# Repo Structure
 - Files:
    - requirements.txt -> contains the required python packages for the project
    - data_creation.yaml -> configurations/arguments for data collection from cmd line
    - create_data.py -> run from cmd line
 - Folder:
    - Liu_paper_code: contains the forged code from https://github.com/lydiatliu/delayedimpact (indirectly used for data collection)

# Dataset Generation

Geration of simulated and synthetic datasets.

**Parameters**:
- Need to be set:
  - directory of the raw data from Hardt et al. (2016)
  - path (incl. filname) for the created synthetic dataset
  - set_size: absolute size of the final dataset 
   - order_of_magnitude: number of samples with are drawn from the FICO-distribution in one step
  - group_size_ratio: ratio of Race in the dataset (black to white samples)
  - black_label_ratio: ratio of Black samples with true and false labels.
  - round_num_scores: indicator of how the feature "Scores" is rounded
  - shuffle_seed: controls the shuffle of the dataset
 
  
**Key details**:
- The original dataset according to Hardt et al. (2016) has the group_size_ratio: [0.12;0.88] and black_label_ratio: [0.66;0.34] (white_label_ratio: [0.24,0.76]). 
  By changing those params we interfere tith the score distributions and create a synthetic dataset.
- The ```data_creation_utils.py``` is the pyfile that includes all of the helpful functions for the data collection
- How to run:
  - Way 2: Set params in ```data_creation.yaml``` or create your own .yaml file and run ```python create_data.py -config data_creation``` from any cmd line (substitude the -config parameter with your own yaml-file name).



<!-- CONTACT -->
## Contact
* Mackenzie Jorgensen - mackenzie.jorgensen@kcl.ac.uk
* Hannah Richert - hrichert@ous.de

<!-- ACKNOWLEDGEMENTS -->
## Acknowledgments
Thank you to Lydia for helping me get started using her code!

<!-- License -->
## License
Lydia's [repository](https://github.com/lydiatliu/delayedimpact) is licensed under the BSD 3-Clause "New" or "Revised" [License](https://github.com/lydiatliu/delayedimpact/blob/master/LICENSE).
