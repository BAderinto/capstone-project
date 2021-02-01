# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
import json
import logging
import os
import pickle
import numpy as np
import pandas as pd
import joblib

import azureml.automl.core
from azureml.automl.core.shared import logging_utilities, log_server
from azureml.telemetry import INSTRUMENTATION_KEY

from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType
from inference_schema.parameter_types.pandas_parameter_type import PandasParameterType


input_sample = pd.DataFrame({"baseline value": pd.Series([0.0], dtype="float64"), "accelerations": pd.Series([0.0], dtype="float64"), "fetal_movement": pd.Series([0.0], dtype="float64"), "uterine_contractions": pd.Series([0.0], dtype="float64"), "light_decelerations": pd.Series([0.0], dtype="float64"), "severe_decelerations": pd.Series([0.0], dtype="float64"), "prolongued_decelerations": pd.Series([0.0], dtype="float64"), "abnormal_short_term_variability": pd.Series([0.0], dtype="float64"), "mean_value_of_short_term_variability": pd.Series([0.0], dtype="float64"), "percentage_of_time_with_abnormal_long_term_variability": pd.Series([0.0], dtype="float64"), "mean_value_of_long_term_variability": pd.Series([0.0], dtype="float64"), "histogram_width": pd.Series([0.0], dtype="float64"), "histogram_min": pd.Series([0.0], dtype="float64"), "histogram_max": pd.Series([0.0], dtype="float64"), "histogram_number_of_peaks": pd.Series([0.0], dtype="float64"), "histogram_number_of_zeroes": pd.Series([0.0], dtype="float64"), "histogram_mode": pd.Series([0.0], dtype="float64"), "histogram_mean": pd.Series([0.0], dtype="float64"), "histogram_median": pd.Series([0.0], dtype="float64"), "histogram_variance": pd.Series([0.0], dtype="float64"), "histogram_tendency": pd.Series([0.0], dtype="float64")})
output_sample = np.array([0])
try:
    log_server.enable_telemetry(INSTRUMENTATION_KEY)
    log_server.set_verbosity('INFO')
    logger = logging.getLogger('azureml.automl.core.scoring_script')
except:
    pass


def init():
    global model
    # This name is model.id of model that we want to deploy deserialize the model file back
    # into a sklearn model
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'model.pkl')
    path = os.path.normpath(model_path)
    path_split = path.split(os.sep)
    log_server.update_custom_dimensions({'model_name': path_split[1], 'model_version': path_split[2]})
    try:
        logger.info("Loading model from path.")
        model = joblib.load(model_path)
        logger.info("Loading successful.")
    except Exception as e:
        logging_utilities.log_traceback(e, logger)
        raise


@input_schema('data', PandasParameterType(input_sample))
@output_schema(NumpyParameterType(output_sample))
def run(data):
    try:
        result = model.predict(data)
        return json.dumps({"result": result.tolist()})
    except Exception as e:
        result = str(e)
        return json.dumps({"error": result})
