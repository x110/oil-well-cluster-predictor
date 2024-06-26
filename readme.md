# Oil Well Cluster Predictor
## Overview
This project aims to predict the cluster affiliation of oil wells based on time series production metrics using machine learning and deep learning  techniques. The dataset consists of two main files: `timeseries_data.csv` containing the time series production metrics of various wells, and `well_data.csv` containing the cluster grouping of each well.
## Project Structure
* `dataset/`: 
    * `raw/`: Contains the raw input data files, including timeseries_data.csv and well_data.csv.
    * `interm/`: Holds intermediate data files, including the indices of the train and test samples used in the project.
* `models/`: Directory containing the input data files.
* `notebooks/`: Jupyter notebooks for project demo, exploratory data analysis (EDA), and error analysis. 
* `scripts/`: Scripts for various project tasks, such as train/test split, model training, evaluation, and prediction.
* `src/`:  Source code files are organized in this directory, including modules for data preprocessing, feature engineering, model training, and evaluation.

## Setup Instructions

1. **Download and Unzip:**
* Download the project zip file
* Extract the contents to your local machine
```bash
unzip well-cluster-prediction.zip
cd well-cluster-prediction

```
2. **Install Dependencies**
* Install the required Python dependencies:
```bash
    pip install -r requirements.txt
```

This section outlines the process of training and evaluating a machine learning model for the oil well cluster predictor project.

### Training Data Preparation

Firstly, the training and test datasets are prepared using the `train_test_split.py` script with the configuration file `config.json`. After execution, the datasets are saved to the following locations:
- Train dataset: `./dataset/interm/train.csv`
- Test dataset: `./dataset/interm/test.csv`

```bash
!python scripts/train_test_split.py config.json
```

### Model Training

Next, the model training is performed using the `run_experiment.py` script with the same configuration file. The best model is identified along with its parameters and is saved for future use. Here are the details of the best model obtained:

- Preprocessing steps: StandardScaler
- Classifier: RandomForestClassifier with balanced class weights and a maximum depth of 10

```bash
!python scripts/run_experiment.py config.json
```

Best parameters:
```python
Pipeline(steps=[('preprocessor', StandardScaler()),
                ('clf',
                 RandomForestClassifier(class_weight='balanced',
                                        max_depth=10))])
```

Best score: 0.2758683098711358

Best model saved at: `./models/model_20240229152028.pkl`

### Model Evaluation

The trained model is evaluated using the `evaluation.py` script with the configuration file `config.json`. The classification report and confusion matrix are generated to assess the model's performance.

```bash
!python scripts/evaluation.py config.json
```

#### Classification Report
```
              precision    recall  f1-score   support
           0       0.44      0.59      0.51        32
           1       0.30      0.19      0.23        16
           2       0.33      0.37      0.35        30
           3       0.67      0.18      0.29        11

    accuracy                           0.39        89
   macro avg       0.44      0.33      0.34        89
weighted avg       0.41      0.39      0.38        89
```

#### Confusion Matrix
```
[[19  1 12  0]
 [ 6  3  6  1]
 [15  4 11  0]
 [ 3  2  4  2]]
```

### Prediction

Finally, predictions are made using the trained model on new data (`new_predict_data.csv`) using the `predict.py` script with the configuration file `config.json`. The predictions for each well are provided in the output dictionary.

```bash
!python scripts/predict.py config.json new_predict_data.csv
```

Prediction results:
```python
{'well_14': 'constant', 'well_9': 'multi'}
```


## Tasks
- [x] split the notebook into .py modules
    - [x] make_dataset.py
    - [x] train.py
        -[x] restructure the end points of train.py
    - [x] evaluation.py
    - [x] write predict.py including the preprocessing pipeline
- [x] test the complete code
- [x] write script to run whole thing:
    -[x] Fix imports, collect all relevant functions in one place
    -[x] write it as a script
    - [x] split script into scripts
    - [x] write evaluation script
    - [x] write predict script
- [x] fix warning: cuments/ml-projects/oil-well-cluster-predictor/src/transformations.py:112: FutureWarning: DataFrame.fillna with 'method' is deprecated ...
- [x] save model in a unique file path
- [ ] add error analysis:
    - [x] confusion matrix
    - [x] find samples that are class a and classified as a, class a and classified as b, class b and classified as a and class b and classified as b
    - inspect their plots:
        -[x] create a 5x5 subplot
- [x] encapsulate all trans in one pipeline: ||||||
    - [x] bug fix: add the index as a colum so you can align X and y after processing X
- [x] pass model path in evaluation script
- [ ] update readme
- [ ] modify gridsearchcv to iterate over pipelines
- [ ] plot learning curves pg.133
- [ ] add wandb
- [ ] Data audit
- [ ] EDA: Well card
- [ ] add samples removal cleanup methods
- [ ] add error analysis
- [ ] Solve using DL
- [ ] Solve using ensemble/blending/stacking
- [ ] solve it as a clustering problem

- [ ] add APIs
- [ ] ad pytests
- [ ] merge scripts with src?
- [ ] install as pip package to get rid of path append
- [ ] link the play/debug button to the main file
- [ ] repeat on a different dataset
- create a run_all.py script


Bonus:
- [ ] Develop a gui and host it on github.io
