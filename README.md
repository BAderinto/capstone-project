# Capstone Project - Fetal Health Classification

### Overview
The dataset for this project was obtained from kaggle, [Fetal Health Classification](https://www.kaggle.com/andrewmvd/fetal-health-classification). 

The dataset is based on the aim to reduce child mortality which is reflected in several of the United Nations' Sustainable Development Goals and is a key indicator of human progress. The UN expects that by 2030, countries end preventable deaths of newborns and children under 5 years of age, with all countries aiming to reduce under‑5 mortality to at least as low as 25 per 1,000 live births.

Parallel to notion of child mortality is of course maternal mortality, which accounts for 295 000 deaths during and following pregnancy and childbirth (as of 2017). The vast majority of these deaths (94%) occurred in low-resource settings, and most could have been prevented.

The dataset was collected using Cardiotocograms (CTGs) are a simple and cost accessible option to assess fetal health, allowing healthcare professionals to take action in order to prevent child and maternal mortality. The equipment itself works by sending ultrasound pulses and reading its response, thus shedding light on fetal heart rate (FHR), fetal movements, uterine contractions and more.


## Experiment Overview
This capstone project was aimed at training machine learning models for classifying fetal health risks that could prevent child and maternal mortality based on Cardiotocograms(CTG) exams data. The project takes the dataset and the `AutoML` and `HyperDrive` training capabilities of the Microsoft Azure Machine Learning SDK to achieve the classification experiment. The two experiments were successfully completed, and the best performing model based on their accuracy metric was the AutoML experiment which was then deployed as a ACI (Azure Container Instance) web service.

The `HyperDrive` experiment was aimed at optimizing the parameters of a pre-selected machine learning algorithm in this case `RandomForestClassifier`, based on its returned percentage accuracy to achieve a high performance from machine learning model. The Azure ML experiment environment was defined in a `conda environment YAML` file with the required training script dependencies. A `ScriptRunConfig` object was used to specify the configuration details of your training job, the training script, environment to use, and the compute target used. The random sampling was used to try different configuration sets of hyperparameters – `n_estimators` and `min_samples_split`, to maximize the primary metric, Accuracy. The experiment was submitted through the specification of the `HyperDriveConfig` which contains information about hyperparameter space sampling, termination policy, primary metric, estimator, and the compute target. The experiment run resulted in 93.57% accuracy.

![hyperdrive_config experiment run accuracy](Images/hyperconfig_accuracy.png)


The AutoML experiment gives the opportunity to automatically explore a variety of machine learning algorithms to get improved model performance based on the primary metric, accuracy. The result of the the AutoML experiment showed that the `Voting Ensemble` model was the best performing algorithm with accuracy > 95.51%. 

![automl experiment run accuracy](Images/voting_ensemble.png)

The best performing model from the AutoML experiment was deployed as a webservice using Azure Container Instance (ACI) and a REST URI was produced. HTTP post requests were successfully sent to the URI for Inferencing using the test data in json format. The service was successfuly deleted afterwards.


## Dataset
It contains 2126 records of features extracted from Cardiotocogram exams, which were then classified by three expert obstetritians into 3 classes - Normal,Suspect, Pathological.


### Task
*TODO*: Explain the task you are going to be solving with this dataset and the features you will be using for it.

baseline value - Baseline Fetal Heart Rate (FHR)

accelerations - Number of accelerations per second

fetal_movement - Number of fetal movements per second

uterine_contractions - Number of uterine contractions per second

light_decelerations - Number of LDs per second

severe_decelerations - Number of SDs per second

prolongued_decelerations - Number of PDs per second

abnormal_short_term_variability - Percentage of time with abnormal short term variability

mean_value_of_short_term_variability - Mean value of short term variability

percentage_of_time_with_abnormal_long_term_variability - Percentage of time with abnormal long term variability

mean_value_of_long_term_variability - Mean value of long term variability

histogram_width - Width of the histogram made using all values from a record

histogram_min - Histogram minimum value

histogram_max - Histogram maximum value

histogram_number_of_peaks - Number of peaks in the exam histogram

histogram_number_of_zeroes - Number of zeroes in the exam histogram

histogram_mode - Hist mode

histogram_mean - Hist mean

histogram_median - Hist Median

histogram_variance - Hist variance

histogram_tendency - Histogram trend

fetal_health - Fetal health: 1 - Normal 2 - Suspect 3 - Pathological


### Access
The data set has been downloaded from kaggle and can be accessed via [my github](https://raw.githubusercontent.com/BAderinto/capstone-project/main/fetal_health.csv). The data is read into an Azure Machine Learning TabularDataset by using the following code

```
from azureml.data.dataset_factory import TabularDatasetFactory
ds = TabularDatasetFactory.from_delimited_files("https://raw.githubusercontent.com/BAderinto/capstone-project/main/fetal_health.csv")
```

## Automated ML

THe Automated ML experiment by instantiating an `AutoMLConfig` object as follows:
```
automl_settings = {
    "experiment_timeout_minutes": 20,
    "max_concurrent_iterations": 5,
    "primary_metric" : 'accuracy'
}

automl_config = AutoMLConfig(
        task='classification',
        compute_target=compute_target,
        training_data=train_dataset,
        label_column_name='fetal_health',
        n_cross_validations=5,
        **automl_settings
)
```


## Automated ML

Configuration and settings used for this Capstone Automated ML experiment are further tabulated in the table below:

Configuration | Description | Value
------------- | ----------- | -----
experiment_timeout_minutes | This is used as an exit criteria, it defines how long, in minutes, your experiment should continue to run | 20
max_concurrent_iterations | Represents the maximum number of iterations that would be executed in parallel | 5
primary_metric | The metric that Automated Machine Learning will optimize for model selection | accuracy
task | The type of task to run. Values can be 'classification', 'regression', or 'forecasting' depending on the type of automated ML problem | classification
compute_target | The compute target to run the experiment on | compute_target
training_data | Training data, contains both features and label columns | train_dataset
label_column_name | The name of the label column | fetal_health
n_cross_validations | No. of cross validations to perform | 5 


### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

### AutoML Screenshots

**Run Details Widget**
This is screenshot of the `RunDetails` widget for the automl experiment after completion.
![autoML_runDetails](Images/automl_result0.png)

![autoML_runDetails_accuracy](Images/automl_result2.png)

**Best Model**
This is screenshot of the best model trained with it's parameters.
![autoML_bestModel](Images/best_automl_run.png)

**Others**
Below are diagnostic curves that help in the interpretation of probabilistic forecast for classification predictive modeling problems.

**Precsion-Recall curve plot showing the accuracy, AUC_macro, AUC_micro, AUC_weighted values for the best run. **
![autoML_bestModel](Images/automl1.png)
![autoML_bestModel](Images/automl2.png)

**ROC curve plot**
![autoML_bestModel](Images/automl4.png)

**Calibrartion curve**
![autoML_bestModel](Images/automl5.png)
![autoML_bestModel](Images/automl6.png)

**Lift curve**
![autoML_bestModel](Images/automl7.png)

**Cummlative gains curve**
![autoML_bestModel](Images/automl8.png)
![autoML_bestModel](Images/automl9.png)

## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search
For the hyperparameter tuning experiment, `RandomForestClassifier` algorithm from the `sklearn.ensemble` framework in conjuction with hyperDrive was used. The hyperparameters used for this experiment are `n_estimators`, the number of trees in the forest, with default value is 20 and `min_samples_split` the minimum number of samples required to split an internal node, default value is 2.

The conda_dependencies.yml was added to the created environment that contains the scikit-learn library as shown below.

![autoML_bestModel](Images/scikit-env.png)

HyperDriveConfig was created using the `ScriptRunConfig` which was created by specifying the training script, compute target and environment, the termination policy, `BanditPolicy` as well as the hyperparameteras shown below. 

![autoML_bestModel](Images/hyperconfig1.png)
![autoML_bestModel](Images/hyperconfig20.png)

### Results
![best hyperparameter tuning](Images/best_hyp_model.png)
![best hyperparameter tuning](Images/hyp_result1.png)
![best hyperparameter tuning](Images/hyp_result2.png)


## Model Deployment

To deploy a Model using Azure Machine Learning Service, we need following:
1. A trained Model
1. Inference configuration; includes scoring script and environment
1. Deploy configuration; includes choice of deployment (ACI, AKS or local) and cpu/gpu/memory allocation

Scoring script, `score.py`, which describes the input data that the model expects and passes it to the model for prediction and returns the results as well as the environment 
can be downloaded and saved from the model generated from the best automl experiment run.  

The deployment was achieved using Azure Container Instances with `cpu_cores = 1` and `memory_gb = 1`, while the Inference configuration is created using the downloaded scoring script.
![aci_webservice_model](Images/automl_aci_model.png)

The test data passed to the model endpoint is converted to JSON format as shwon below.
![aci_webservice_model](Images/test_data_json.png)

Below is the script to pass the test data to the model as an HTTP POST request and return the response; 

![web service](Images/result.png)

Screenshots below show a demonstration of sample data response from the deployed model.

## Screen Recording
[screen recording](https://youtu.be/b-mrf_2GbXo)


## Future Improvements
To improve this project in future:
•	compare "Accuracy" to "AUC Weighted" as primary metric 
•	vary values for the n_estimators and the fractional(float) values of min_samples_split hyperparameters
•	explore max_depth parameter for the algorithm.
•	convertion of the model to ONNX format would be considered



## Reference
[sklearn.ensemble.RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
