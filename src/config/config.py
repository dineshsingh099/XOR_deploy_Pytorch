import os
import pathlib
import src

LOSS_FUNCTION = "bce_loss"
mb_size = 2
epochs = 100

X_train = None
Y_train = None
training_data = None

PACKAGE_ROOT = pathlib.Path(src.__file__).resolve().parent
DATAPATH = os.path.join(PACKAGE_ROOT,"datasets")

SAVED_MODEL_PATH = os.path.join(PACKAGE_ROOT,"trained_models")