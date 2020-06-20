import json
import os

from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import tensorflow as tf
from tqdm import tqdm

flags.DEFINE_string('data_dir', '..', 'path to raw TIL dataset')
flags.DEFINE_enum('split', 'train', ['train', 'val'], 'specify train or val spit')
flags.DEFINE_string('output_file', './data/til.tfrecord', 'output dataset')
flags.DEFINE_string('classes', './data/til.names', 'classes file')

def transform_annotations(annotations):
  tf_annotations = {}

  for annotation in annotations:
    obj = {
      'category_id': annotation['category_id'],
      'bbox': annotation['bbox']
    }

    if annotation['image_id'] not in tf_annotations:
      tf_annotations[annotation['image_id']] = [obj]
    else:
      tf_annotations[annotation['image_id']].append(obj)

  return tf_annotations

def build_example(image_id, annotation, classes):
  img_path = os.path.join(FLAGS.data_dir, FLAGS.split, FLAGS.split, '{}.jpg'.format(image_id))

  with open(img_path, 'rb') as f:
    img_raw = f.read()
    im = cv2.imread(img_path)

    height, width, _ = im.shape

    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes_text = []

    for obj in annotation:
      xmin.append(float(obj['bbox'][0]) / width)
      ymin.append(float(obj['bbox'][1]) / height)
      xmax.append(float(obj['bbox'][0] + obj['bbox'][2]) / width)
      ymax.append(float(obj['bbox'][1] + obj['bbox'][3]) / height)
      classes_text.append(classes[obj['category_id'] - 1].encode('utf8'))

    example = tf.train.Example(features=tf.train.Features(feature={
      'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
      'image/object/bbox/xmin': tf.train.Feature(float_list=tf.train.FloatList(value=xmin)),
      'image/object/bbox/xmax': tf.train.Feature(float_list=tf.train.FloatList(value=xmax)),
      'image/object/bbox/ymin': tf.train.Feature(float_list=tf.train.FloatList(value=ymin)),
      'image/object/bbox/ymax': tf.train.Feature(float_list=tf.train.FloatList(value=ymax)),
      'image/object/class/text': tf.train.Feature(bytes_list=tf.train.BytesList(value=classes_text))
    }))

    return example

def main(_argv):
  with open(os.path.join(FLAGS.data_dir, '{}.json'.format(FLAGS.split))) as f:
    data = json.load(f)
    annotations = data['annotations']

    with open(FLAGS.classes) as g:
      classes = g.read().splitlines()

      writer = tf.io.TFRecordWriter(FLAGS.output_file)
      tf_annotations = transform_annotations(annotations)

      for image_id, annotation in tqdm(tf_annotations.items()):
        tf_example = build_example(image_id, annotation, classes)
        writer.write(tf_example.SerializeToString())

      writer.close()
      logging.info('Done')

if __name__ == '__main__':
    app.run(main)
