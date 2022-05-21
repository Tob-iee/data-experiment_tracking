# import the necessary packages
import os
import argparse
# import csv, json
import pandas as pd
# import string
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
tf.get_logger().setLevel('INFO')
# from functools import partial


# FILENAMES_PATH = "./data_store/data/American Sign Language Letters.v1-v1.tensorflow/"

# Initialize the parser
parser = argparse.ArgumentParser(description="Data Preprocessor")

# Add the parameters positional/optional
parser.add_argument("-dp", dest= "datapath", help="path directory to the raw images and csv", default="./data_store/data/American Sign Language Letters.v1-v1.tensorflow/")
parser.add_argument("-s", dest= "datasplit", required=True, help="the particular data split to be preprocessed (choose between train/, valid/, or test/ but if all enter all")
parser.add_argument("-n", dest= "tfrec", help="the name for the tfrecord files", default="Letters")

# parse the arguments
args = parser.parse_args()

# TRAINING_FILENAMES =  FILENAMES_PATH + "train/"
# VALID_FILENAMES = FILENAMES_PATH + "valid/"
# TEST_FILENAMES = FILENAMES_PATH + "test/"


# TRAINING_FILENAMES =  args.datapath + "train/"
# VALID_FILENAMES = args.datapath + "valid/"
# TEST_FILENAMES = args.datapath + "test/"

# print("Train TFRecord Files:", len(TRAINING_FILENAMES))
# print("Validation TFRecord Files:", len(VALID_FILENAMES))
# print("Test TFRecord Files:", len(TEST_FILENAMES))
datasplits = []
FILE_PATH = args.datapath + args.datasplit

def image_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[tf.io.encode_jpeg(value).numpy()])
    )


def bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode()]))


def float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def float_feature_list(value):
    """Returns a list of float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def create_example(image, path, example):
    feature = {
        "image/encoded": image_feature(image),
        "image/filename": bytes_feature(example["filename"]),
        "image/format": bytes_feature("jpg"),
        "image/height":  int64_feature(example["height"]),
        "image/object/bbox/xmin": float_feature(example["xmin"]),
        "image/object/bbox/ymin": float_feature(example["ymin"]),
        "image/object/bbox/xmax": float_feature(example["xmax"]),
        "image/object/bbox/ymax": float_feature(example["ymax"]),
        "image/object/class/label": int64_feature(example["label"]),
        "image/object/class/text": bytes_feature(example["class"]),
        "image/width": int64_feature(example["width"])
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

def tf_rec_writer(dataset_, path, data_split):
    record_file = f"{data_split}.tfrecords"
    with tf.io.TFRecordWriter(record_file) as writer:
        for data in dataset_:

            image = os.path.join(path, data['filename'])

            image = tf.io.decode_jpeg(tf.io.read_file(image))
    #         print(image)
            feature = create_example(image, path, data)
            writer.write(feature.SerializeToString())


def file_loader(FILE_PATH):
    count=0
    images = []
    annotations = []
    for files in os.listdir(FILE_PATH):
    #     count+= 1
    # print(count)

        if "csv" in files:
            annotations.append(files)

        else:
            images.append(files)

    # print(annotations)
#     print(images)
    return annotations, images

def convert_to_dict(FILE_PATH, annotations):
    dataset = pd.read_csv(os.path.join(FILE_PATH, annotations[0]))

    lb_make = LabelEncoder()
    dataset['label'] = lb_make.fit_transform(dataset[["class"]])

    dataset = dataset.to_dict(orient='records')
    return dataset



if __name__ ==  '__main__':
    # pass
    print("Train TFRecord Files:", len(FILE_PATH))

    # for splits in datasplits:
    # annotations_test, images = file_loader(TEST_FILENAMES)
    # dataset_test = convert_to_dict(TEST_FILENAMES, annotations_test)
    # print(len(dataset_test))

    # test_tfrec = tf_rec_writer(dataset_test, TEST_FILENAMES, "Letters")