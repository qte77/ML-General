NAME = ""
COLLABORATORS = ""
# Python ≥3.5 is required
import sys
#assert sys.version_info >= (3, 5)

#h5py error: "We will have people working on making TF work with
#h5py >= 3 in the future, but this will only land in TF 2.5 or later."
#https://github.com/keras-team/keras/issues/14265
#https://github.com/tensorflow/tensorflow/issues/44467
#versions: 'h5py < 3.0.0', 'h5py==2.10.0'
pip install 'h5py < 3.0.0'
import h5py
#h5py.__version__


# Scikit-Learn ≥0.20 is required
import sklearn
#assert sklearn.__version__ >= "0.20"

try:
    # %tensorflow_version only exists in Colab.
    %tensorflow_version 2.x
except Exception:
    pass

# TensorFlow ≥2.0 is required
import tensorflow as tf
#assert tf.__version__ >= "2.0"

from tensorflow import keras

# Common imports
import numpy as np
import os

# to make this notebook's output stable across runs
np.random.seed(42)

# To plot pretty figures
%matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "ann"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

# Ignore useless warnings (see SciPy issue #5998)
import warnings
warnings.filterwarnings(action="ignore", message="^internal gelsd")
sv = sys.version_info
print("sys:\t %s.%s.%s" % (sv.major, sv.minor, sv.micro))
print("skl:\t", sklearn.__version__)
print("tf:\t", tf.__version__)
print("k:\t", keras.__version__)
print("np:\t", np.__version__)
print("h5:\t", h5py.__version__)
# YOUR CODE HERE
#raise NotImplementedError()
#https://en.wikipedia.org/wiki/Early_stopping#Early_stopping_based_on_cross-validation
#https://www.javacodemonk.com/difference-between-loss-accuracy-validation-loss-validation-accuracy-in-keras-ff358faa
#https://docs.paperspace.com/machine-learning/wiki/accuracy-and-loss
#https://keras.io/guides/training_with_built_in_methods/
#https://www.tensorflow.org/guide/keras/train_and_evaluate
#https://keras.io/api/metrics/
from scipy.stats import reciprocal
from sklearn.model_selection import RandomizedSearchCV
np.random.seed(42)
tf.random.set_random_seed(42)
#Training
#Validation on a holdout set generated from the original training data
#Evaluation on the test data
#https://keras.io/guides/training_with_built_in_methods/
(X_train_o, y_train_o), (X_test_o, y_test_o) = keras.datasets.mnist.load_data()
# Preprocess the data (these are NumPy arrays)
X_train = X_train_o.reshape(-1, 784).astype("float32") / 255.0
X_test = X_test_o.reshape(-1, 784).astype("float32") / 255.0

y_train = y_train_o.astype("float32")
y_test = y_test_o.astype("float32")

# Reserve 10,000 samples for validation
X_valid = X_train[-10000:]
y_valid = y_train[-10000:]
X_train = X_train[:-10000]
y_train = y_train[:-10000]
def build_model(n_hidden=2, n_neurons=30,learning_rate=3e-3,input_dim=784):
    model = keras.models.Sequential()
    for layer in range(n_hidden):
        model.add(keras.layers.Dense(n_neurons, input_dim=input_dim, activation="relu"))
    model.add(keras.layers.Dense(n_neurons, input_dim=input_dim)), #activation="softmax"))
    optimizer = keras.optimizers.SGD(lr=learning_rate)
    model.compile(loss=["sparse_categorical_crossentropy"],
                  optimizer="sgd",
                  metrics=["acc"]) #categorical_accuracy
    return model
keras_reg = keras.wrappers.scikit_learn.KerasRegressor(build_model)
#https://keras.io/guides/writing_your_own_callbacks/
#https://github.com/keras-team/keras/issues/2548
#TODO stop RandomizedSearchCV not just epoch
class EarlyStoppingAtMode(keras.callbacks.Callback):
    """Stop training when the mode (loss, acc) is at its min/max.

Arguments:
    test_data
    mode
    threshold
    patience
  """

    def __init__(self, test_data, mode="loss", threshold=0.99, patience=0):
        super(EarlyStoppingAtMode, self).__init__()
        self.test_data = test_data
        self.mode = mode
        self.threshold = threshold
        self.patience = patience
        
        self.wait = 0
        self.best_weights = None
        if self.mode == "loss":
            self.best = np.Inf
        elif self.mode == "acc":
            self.best = -np.Inf
        else:
            self.mode = 0
        # The epoch the training stops at.
        self.stopped_epoch = 0

    #not suitable for RandomizedSearchCV or Grid, resets after every training cycle
    def on_train_begin(self, logs=None):
        self.model.set_weights(self.best_weights)

    def on_epoch_end(self, epoch, logs=None):

        def stop_train(epoch):
            print("\t > Restoring model weights from the end of the best epoch.")
            self.stopped_epoch = epoch
            self.model.stop_training = True
            self.model.set_weights(self.best_weights) 

        current = logs.get(self.mode)
        x, y = self.test_data
        test_loss, test_acc = self.model.evaluate(x, y, verbose=0)
        
        print()
        print("\t > Validation best %s: %s" % (self.mode, self.best))
        print("\t > Testing loss: %s, acc: %s" % (test_loss, test_acc))
        print("\t > Epochs since last improvement: %s" % self.wait)
                
        if (self.mode == "loss" and np.less(test_loss, self.threshold)) or \
        (self.mode == "acc" and np.greater(test_acc, self.threshold)):
            print("\n\t > Threshold reached.")
            stop_train(epoch)
        
        if (self.mode == "loss" and np.less(current, self.best)) or \
        (self.mode == "acc" and np.greater(current, self.best)):
            print("\t > Setting new best values.")
            self.best = current
            self.wait = 0
            # Record the best weights if current results is better (more).
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                print("\t > No patience.")
                stop_train(epoch)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print("\n\t > Epoch %05d: early stopping" % (self.stopped_epoch + 1))

callbacks = [
    keras.callbacks.EarlyStopping(patience=10),
    EarlyStoppingAtMode((X_test, y_test), "acc", 0.95, 50)
]
#https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html
#TODO: InvalidArgumentError: Received a label value of 9
#which is outside the valid range of [0, 9).  Label values: [...]
param_distribs = {
    "n_hidden": [0, 1, 2, 3],
    "n_neurons": np.arange(1, 100),
    "learning_rate": reciprocal(3e-4, 3e-2),
}
rnd_search_cv = RandomizedSearchCV(
    keras_reg, param_distribs, n_iter=10, cv=3, verbose=2
)
history = rnd_search_cv.fit(X_train, y_train,
                  validation_data=(X_valid, y_valid),
                  epochs=10, steps_per_epoch=5,
                  batch_size=64, shuffle=True,
                  callbacks=callbacks)
rnd_search_cv.best_params_
rnd_search_cv.best_score_
rnd_search_cv.best_estimator_
rnd_search_cv.score(X_test, y_test)
model = rnd_search_cv.best_estimator_.model
model
model.evaluate(X_test, y_test)
