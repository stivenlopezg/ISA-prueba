import sys
import logging
import numpy as np

# Logger configuration -----------------------------------------------------------------------------------------------

logger_app_name = 'ISA-analytic-model'
logger = logging.getLogger(logger_app_name)
logger.setLevel(logging.INFO)
consoleHandle = logging.StreamHandler(sys.stdout)
consoleHandle.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
consoleHandle.setFormatter(formatter)
logger.addHandler(consoleHandle)

# Project ------------------------------------------------------------------------------------------------------------

INPUT_FILE = 'baterias-datos.csv'

NUMERICAL_FEATURES = ['vcd', 'amp', 'tmp', 'imp']

LABEL = 'estado_id'

TO_DROP = 'estado'

SEED = 42

FEATURES_DTYPES = {
    'vcd': np.float64,
    'amp': np.float64,
    'tmp': np.float64,
    'imp': np.float64
}

LABEL_DTYPE = {
    'estado_id': np.int64
}

DATA_FOLDER = 'data'
MODELS_FOLDER = 'models'
