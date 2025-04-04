# Data Science Classification Pipeline

A framework for training, evaluating, and comparing machine learning models on classification tasks.

## Features

- **Modular Design**: 
  - Separate components for data loading, preprocessing, model definition, and experiment running
- **Built-in Datasets**:
  - Taiwanese Credit Default dataset
  - US Permanent Visa application dataset
- **Preprocessing Pipeline**:
  - Automated datetime feature extraction
  - Cyclical feature encoding

## Getting Started
- click through `start.ipynb` for a quick start

## Todos

### data.loaders
- Port cleaner and preprocessing code for BAM data from Dan's project

### models.utils
- Check: Validity of existing pipeline templates
- split up logic of `get_pipeline` into two parts: imputation and encoding. Encoding should change only with the model but not dataset. Imputation changes with dataset, may or may not depend on model? 
- Learn: should use `LabelEncoder` in preprocessing to encode string features numerically? For ordinal categorical features with a meaningful order
- Build CatBoost pipeline

### experiments.runners
- print info / number of features in `X_train` (just to check what came through the pipeline)
- add parameters for models to pass through, e.g. `RandomForestClassifier(n_estimators=200)`
- parameterize the subsampler for training data (for increasing subsample experiments)
- Build experiment setup using [MLXP](https://github.com/inria-thoth/mlxp)
- chunk-sample over test data (to get results on all test data even for TabPFN)
- build saving run results as CSV or so
- place artifacts and results in sensible output path in outputs/