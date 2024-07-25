from typing import NamedTuple, Dict, Any, Text
from tfx.components.trainer.fn_args_utils import FnArgs
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers
import keras_tuner as kt
from keras_tuner.engine.base_tuner import BaseTuner
import tensorflow_transform as tft
import tensorflow as tf

TunerFnResult = NamedTuple('TunerFnResult', [('tuner', BaseTuner), ('fit_kwargs', Dict[Text, Any])])

LABEL_KEY = "Growth_Milestone"
FEATURE_KEYS = ['Sunlight_Hours', 'Temperature', 'Humidity']


def transformed_name(key):
    """Renaming transformed features"""
    return key + "_xf"

def gzip_reader_fn(filenames):
    """Loads compressed data"""
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')

def input_fn(file_pattern, tf_transform_output, num_epochs=None, batch_size=64):
    """Get post_transform feature & create batches of data"""
    
    # Get post_transform feature spec
    transform_feature_spec = (
        tf_transform_output.transformed_feature_spec().copy())
    
    # create batches of data
    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transform_feature_spec,
        reader=gzip_reader_fn,
        num_epochs=num_epochs,
        label_key=transformed_name(LABEL_KEY))
    return dataset

def model_builder(hp):
    """Build machine learning model with hyperparameters."""
    inputs = {transformed_name(key): tf.keras.Input(shape=(1,), name=transformed_name(key), dtype=tf.float32) for key in FEATURE_KEYS}
    x = layers.Concatenate()(list(inputs.values()))
    x = layers.Dense(units=hp.Int('units_1', min_value=32, max_value=128, step=32), activation='relu')(x)
    x = layers.Dropout(rate=hp.Float('dropout_1', min_value=0.0, max_value=0.5, step=0.1))(x)
    x = layers.Dense(units=hp.Int('units_2', min_value=16, max_value=64, step=16), activation='relu')(x)
    x = layers.Dropout(rate=hp.Float('dropout_2', min_value=0.0, max_value=0.5, step=0.1))(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
        metrics=[tf.keras.metrics.BinaryAccuracy()]
    )
    
    return model

def tuner_fn(fn_args: FnArgs) -> TunerFnResult:
    """Build the tuner using the KerasTuner API."""
    
    # Load the transformed data
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)
    train_set = input_fn(fn_args.train_files, tf_transform_output, num_epochs=10)
    val_set = input_fn(fn_args.eval_files, tf_transform_output, num_epochs=10)

    # Define the hyperband tuner
    tuner = kt.Hyperband(
        model_builder,
        objective='val_binary_accuracy',
        max_epochs=10,
        factor=3,
        directory=fn_args.working_dir,
        project_name='kt_hyperband'
    )

    # Set fit arguments for the tuner
    early_stopping = EarlyStopping(monitor='val_binary_accuracy',  mode='max', min_delta=0.001, patience=5, verbose=1)

    fit_kwargs = {
        "callbacks": [early_stopping],
        'x': train_set,
        'validation_data': val_set,
        'steps_per_epoch': fn_args.train_steps,
        'validation_steps': fn_args.eval_steps
    }

    return TunerFnResult(tuner=tuner, fit_kwargs=fit_kwargs)

