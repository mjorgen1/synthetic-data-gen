# Synthetic Dataset Generation for Analyzing Fair Classification Systems


This repository contains code for synthetic dataset generation for a loan repayment use case.

We take some inspiration from Liu et al.'s work, [*Delayed Impact of Fair Machine Learning*](https://arxiv.org/abs/1803.04383). We extended their [code](https://github.com/lydiatliu/delayedimpact) here and generated synthetic datasets from their dataset. 


**Data**:
Our simulated, synthetic datasets are based on Hardt et al.'s 2016 dataset. 
- Download the data folder from the Github repository for [fairmlbook](https://github.com/fairmlbook/fairmlbook.github.io/tree/master/code/creditscore) (Barocas, Hardt and Narayanan 2018)
- Save it to the root directory of this repository (the csv files should be in the folder 'data')
- Run: ```delayedimpact/Liu_paper_code/FICO-figures.ipynb```

# Repository Structure
 - Files:
    - ```requirements.txt``` -> contains the required python packages for the project
    - ```data_creation.yaml``` -> configurations/arguments for data collection from cmd line
    - ```create_data.py``` -> run from cmd line
 - Folder:
    - ```Liu_paper_code```: contains the code from https://github.com/lydiatliu/delayedimpact (indirectly used for data collection)

# Synthetic Dataset Generation

**Parameters**:
- Need to be set:
  - Directory of the raw data from Hardt et al. (2016)
  - Path (incl. filname) for the created synthetic dataset
  - ```set_size```: absolute size of the final dataset 
  - ```order_of_magnitude```: number of samples with are drawn from the FICO-distribution in one step
  - ```group_size_ratio```: ratio of Race in the dataset (black to white samples)
  - ```black_label_ratio```: ratio of Black samples with true and false labels.
  - ```round_num_scores```: indicator of how the feature "Scores" is rounded
  - ```shuffle_seed```: controls the shuffle of the dataset
 
  
**Key details**:
- The original dataset according to Hardt et al. (2016) has a ```group_size_ratio```: [0.12;0.88], ```black_label_ratio```: [0.66;0.34], and ```white_label_ratio```: [0.24,0.76]. By changing those parameters we interfere with the demographic and label distributions and create a synthetic dataset.
- The ```data_creation_utils.py``` is the pyfile that includes all of the helpful functions for the data collection
- How to run:
  - Way 2: Set params in ```data_creation.yaml``` or create your own ```~.yaml``` file and run ```python create_data.py -config data_creation``` from any cmd line (substitute the ```-config``` parameter with your ```~.yaml``` file).


<!-- CONTACT -->
## Contact
* Mackenzie Jorgensen - mackenzie.jorgensen@kcl.ac.uk
* Hannah Richert - hrichert@ous.de

<!-- License -->
## License
Lydia's [repository](https://github.com/lydiatliu/delayedimpact) is licensed under the BSD 3-Clause "New" or "Revised" [License](https://github.com/lydiatliu/delayedimpact/blob/master/LICENSE).

<!-- ACKNOWLEDGEMENTS -->
## Acknowledgments
Thank you to Lydia for helping me get started using her code!

<!-- License -->
## License
Lydia's [repository](https://github.com/lydiatliu/delayedimpact) is licensed under the BSD 3-Clause "New" or "Revised" [License](https://github.com/lydiatliu/delayedimpact/blob/master/LICENSE).
