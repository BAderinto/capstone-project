from sklearn.ensemble import RandomForestClassifier
import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory
from azureml.core import Dataset
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler


# TODO: Create TabularDataset using TabularDatasetFactory

ds = TabularDatasetFactory.from_delimited_files(path='https://raw.githubusercontent.com/BAderinto/capstone-project/main/fetal_health.csv')


run = Run.get_context()
  
x = ds.to_pandas_dataframe().dropna()

y = x.pop("fetal_health")


# TODO: Split data into train and test sets.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state = 342, stratify = y)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test) 

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_estimators', type=int, default=20, help="The number of trees in the forest")
    parser.add_argument('--min_samples_split', type=int, default=2, help="The minimum number of samples required to split an internal node")
    
    args = parser.parse_args(args=[])
    
    run.log("The number of trees in the forest:", np.int(args.n_estimators))
    run.log("The minimum number of samples required to split an internal node:", np.int(args.min_samples_split))
    
    model = RandomForestClassifier(n_estimators=args.n_estimators, random_state=0, min_samples_split=args.min_samples_split).fit(x_train, y_train)
    #predict
    fh_preds = model.predict(x_test)

    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", np.float(accuracy))
    
    os.makedirs('outputs', exist_ok=True)
    joblib.dump(value=model, filename='outputs/model.pkl')

    run.complete()
    
if __name__ == '__main__':
    main()
