import sys
import os
import tensorflow_datasets as tfds
import tensorflow as tf
from vn_sign_dataset_builder import VnSign
import logging
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
# Add the parent directory to the sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Suppress TensorFlow logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Suppress specific TensorFlow warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

class VnSignTest(tfds.testing.DatasetBuilderTestCase):
    """Tests for vn_sign dataset."""
    DATASET_CLASS = VnSign
    SPLITS = {
        'train': 3,  # Number of fake train examples
        'test': 1,  # Number of fake test examples
    }

    DL_EXTRACT_RESULT = {
        'videos': 'vn_sign',  # Relative to vn_sign/dummy_data dir
    }

if __name__ == '__main__':
    tfds.testing.test_main()
