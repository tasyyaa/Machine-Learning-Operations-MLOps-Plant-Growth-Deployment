"""
Transform module
"""
import tensorflow as tf
import tensorflow_transform as tft

LABEL_KEY = "Growth_Milestone"
FEATURE_KEYS = ['Sunlight_Hours', 'Temperature', 'Humidity']

def transformed_name(key):
    """Renaming transformed features"""
    return key + "_xf"

def preprocessing_fn(inputs):
    """
    Preprocess input features into transformed features

    Args:
        inputs: map from feature keys to raw features.

    Return:
        outputs: map from feature keys to transformed features.    
    """

    outputs = {}

    # Transform numerical features
    for key in FEATURE_KEYS:
        outputs[transformed_name(key)] = tft.scale_to_z_score(inputs[key])

    # Transform label
    outputs[transformed_name(LABEL_KEY)] = tf.cast(inputs[LABEL_KEY], tf.int64)

    return outputs
