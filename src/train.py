import os
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import mlflow
import mlflow.tensorflow
from mlflow.tracking import MlflowClient
print(mlflow.__version__)

tf.get_logger().setLevel('INFO')


# Set MLflow tracking remote server using Dagshub Mlflow server URI
mlflow.set_tracking_uri(f'https://dagshub.com/{DAGSHUB_REPO_OWNER}/{DAGSHUB_REPO_NAME}.mlflow')



strategy = tf.distribute.get_strategy()
print("Number of replicas:", strategy.num_replicas_in_sync)


ARTIFACTS_PATH = "./data_store/artifacts"
EXPERIMENT_NAME = "Hand_Signs_2"
SET_ARTIFACTS_LOCATION ="/content/MLflow_TF-serving/data_store/artifacts/"

# Load the data

AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 32

FILENAMES_PATH = "/content/MLflow_TF-serving/data_store/data/American Sign Language Letters.v1-v1.tfrecord/"

TRAINING_FILENAMES =  FILENAMES_PATH + "train/Letters.tfrecords"
VALID_FILENAMES = FILENAMES_PATH + "valid/Letters.tfrecords"
TEST_FILENAMES = FILENAMES_PATH + "test/Letters.tfrecords"

print("Train TFRecord Files:", len(TRAINING_FILENAMES))
print("Validation TFRecord Files:", len(VALID_FILENAMES))
print("Test TFRecord Files:", len(TEST_FILENAMES))



# Create the dataset object for tfrecord file(s)

def load_dataset(tf_filenames):
  ignore_order = tf.data.Options()
  ignore_order.experimental_deterministic = False  # disable order, increase speed
  dataset = tf.data.TFRecordDataset(tf_filenames)

  dataset = dataset.with_options(ignore_order)

  return dataset

# Decoding function
def parse_record(record):

  tfrecord_feat_format = (
              {
                  "image/encoded": tf.io.FixedLenFeature([], tf.string),
                  "image/filename": tf.io.FixedLenFeature([], tf.string),
                  "image/format": tf.io.FixedLenFeature([], tf.string),
                  "image/height": tf.io.FixedLenFeature([], tf.int64),
                  "image/object/bbox/xmax": tf.io.FixedLenFeature([], tf.float32),
                  "image/object/bbox/xmin": tf.io.FixedLenFeature([], tf.float32),
                  "image/object/bbox/ymax": tf.io.FixedLenFeature([], tf.float32),
                  "image/object/bbox/ymin": tf.io.FixedLenFeature([], tf.float32),
                  "image/object/class/label": tf.io.FixedLenFeature([], tf.int64),
                  "image/object/class/text": tf.io.FixedLenFeature([], tf.string),
                  "image/width": tf.io.FixedLenFeature([], tf.int64),
              }
          )



  example = tf.io.parse_single_example(record, tfrecord_feat_format)



  IMAGE_SIZE = [400, 400]

  image =  tf.io.decode_jpeg(example["image/encoded"], channels=3)
  image = tf.cast(image, tf.float32)

  xmax = tf.cast(example["image/object/bbox/xmax"], tf.int32)
  xmin = tf.cast(example["image/object/bbox/xmin"], tf.int32)
  ymax = tf.cast(example["image/object/bbox/ymax"], tf.int32)
  ymin = tf.cast(example["image/object/bbox/ymin"], tf.int32)

  box_width = xmax - xmin
  box_height = ymax - ymin
  image = tf.image.crop_to_bounding_box(image, ymin, xmin, box_height, box_width)
  image = tf.image.convert_image_dtype(image, tf.float32)
  image = tf.image.resize(image, IMAGE_SIZE)

# more feature preprocessing
  image = tf.image.random_flip_left_right(image)
  # image = tfa.image.rotate(image, 40, interpolation='NEAREST')


  # image = tf.cast(image, "uint8")
  # image = tf.image.encode_jpeg(image, format='rgb', quality=100)


  label = example["image/object/class/label"]
  label = tf.cast(label, tf.int32)
  # label = tf.one_hot(label, depth=26)


  return (image, label)

def get_dataset(filenames):
  ignore_order = tf.data.Options()
  ignore_order.experimental_deterministic = False  # disable order, increase speed
  dataset = tf.data.TFRecordDataset(filenames)

  dataset = dataset.with_options(ignore_order)

  dataset = dataset.map(parse_record, num_parallel_calls=AUTOTUNE)
  dataset = dataset.cache()

  dataset = dataset.shuffle(buffer_size=10 * BATCH_SIZE, reshuffle_each_iteration=True)
  dataset = dataset.batch(BATCH_SIZE)
  dataset = dataset.prefetch(buffer_size=AUTOTUNE)
  # dataset = dataset.repeat()

  return dataset


def get_cnn():
  model = tf.keras.Sequential([

  tf.keras.layers.Conv2D(kernel_size=3, filters=32, padding='same', activation='relu', input_shape=[400, 400, 3]),
  tf.keras.layers.MaxPooling2D(pool_size=2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(26,'softmax')
  ])

  # opt = SGD(learning_rate=learning_rate)
  optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
  model.compile(loss= tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                optimizer=optimizer,
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
                )
  # model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

  return model


tfr_dataset = get_dataset(TRAINING_FILENAMES)
print(tfr_dataset)

tfr_testdata = get_dataset(VALID_FILENAMES)
print(tfr_testdata)


model = get_cnn()
model.summary()




def main():

  print(mlflow.tracking.get_tracking_uri())
  print(mlflow.get_artifact_uri())

  client = MlflowClient()
  experiment = client.get_experiment_by_name("Hand_Signs_x")


  # if experiment.name == EXPERIMENT_NAME:
  # Set experiment
  # mlflow.set_experiment(experiment_name="Hand_Signs_2")
  experiment = client.get_experiment_by_name("Hand_Signs_2")
  print("Name: {}".format(experiment.name))
  print("Experiment_id: {}".format(experiment.experiment_id))
  print("Artifact Location: {}".format(experiment.artifact_location))
  print("Tags: {}".format(experiment.tags))
  print("Lifecycle_stage: {}".format(experiment.lifecycle_stage))


  # else:
  # create and set experiment
  # experiment_id = mlflow.create_experiment(EXPERIMENT_NAME,
  #                                          artifact_location=SET_ARTIFACTS_LOCATION)
  # print(experiment_id)


  # client.set_experiment_tag(experiment_id, "CV.framework", "Tensorflow_CV")
  # experiment = client.get_experiment(experiment_id)
  # print("Name: {}".format(experiment.name))
  # print("Experiment_id: {}".format(experiment.experiment_id))
  # print("Artifact Location: {}".format(experiment.artifact_location))
  # print("Tags: {}".format(experiment.tags))
  # print("Lifecycle_stage: {}".format(experiment.lifecycle_stage))


  # start experiment tracking runs
  with mlflow.start_run(experiment_id=experiment.experiment_id, run_name="test_artifact store"):
    run = mlflow.active_run()
    print("run_id: {}; status: {}".format(run.info.run_id, run.info.status))

    mlflow.tensorflow.autolog()


    start_training = time.time()

    history = model.fit(tfr_dataset,
              # steps_per_epoch=1513/BATCH_SIZE,
              epochs=30, verbose=1)
    end_training = time.time()

    tf.keras.models.save_model(model, "./model")

    training_time = end_training - start_training

    mlflow.log_param("learning_rate", 0.0001)
    mlflow.log_param("optimizer", "Adam")
    mlflow.log_metric('batchsize', BATCH_SIZE)
    mlflow.log_metric('training_accuracy', history.history['sparse_categorical_accuracy'][-1])
    mlflow.log_metric('training_loss', history.history['loss'][-1])
    # mlflow.log_metric('precision', history.history['precision'])
    # mlflow.log_metric('recall', history.history['recall'])

    mlflow.log_metric('training_time', training_time)
    # mlflow.log_artifact("./model", artifact_path=ARTIFACTS_PATH)



    tfr_testdata = get_dataset(VALID_FILENAMES)
    steps_per_epoch = 145/BATCH_SIZE

    start_evaluating = time.time()
    val_loss, val_accuracy = model.evaluate(tfr_testdata,
      # steps_per_epoch=steps_per_epoch,
      )

    end_evaluating = time.time()
    evaluating_time = end_evaluating - start_evaluating


    mlflow.log_metric('validation_accuracy', val_accuracy)
    mlflow.log_metric('validation_loss', val_loss)
    mlflow.log_metric('evaluating_time', evaluating_time)


    run = mlflow.get_run(run.info.run_id)
    print("run_id: {}; status: {}".format(run.info.run_id, run.info.status))
    print("--")

  # Check for any active runs
  print("Active run: {}".format(mlflow.active_run()))

  return "TRAINING COMPLETED!!!"

# !dvc add data_store/

# !git add data_store.dvc

# !git commit -m "Added artifacts to data_store"

# !dvc push -r origin

# !git push

if __name__ ==  '__main__':
    # pass
    main()