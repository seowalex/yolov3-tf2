import json
import os

from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from yolov3_tf2.models import YoloV3
from yolov3_tf2.dataset import transform_images

flags.DEFINE_string('data_dir', '..', 'path to raw TIL dataset')
flags.DEFINE_string('classes', './data/til.names', 'path to classes file')
flags.DEFINE_string('weights', './checkpoints/yolov3_train_3.tf',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_integer('num_classes', 5, 'number of classes in the model')

def main(_argv):
  physical_devices = tf.config.experimental.list_physical_devices('GPU')

  for physical_device in physical_devices:
    tf.config.experimental.set_memory_growth(physical_device, True)

  yolo = YoloV3(classes=FLAGS.num_classes)
  yolo.load_weights(FLAGS.weights).expect_partial()
  logging.info('weights loaded')

  class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
  logging.info('classes loaded')

  output = {'annotations': []}

  for i, filename in tqdm(enumerate(os.listdir(FLAGS.data_dir))):
    with open(os.path.join(FLAGS.data_dir, filename), 'rb') as f:
      img_raw = tf.io.decode_image(f.read(), channels=3)

      img = tf.expand_dims(img_raw, 0)
      img = transform_images(img, FLAGS.size)

      boxes, scores, classes, nums = yolo(img)
      boxes, scores, classes, nums = boxes[0], scores[0], classes[0], nums[0]
      height, width, _ = tf.shape(img_raw)
      height, width = int(height), int(width)

      for j in range(nums):
        image_id = int(filename.split('.')[0])
        category_id = int(classes[j] + 1)
        bbox = [
          int(boxes[j][0] * width),
          int(boxes[j][1] * height),
          int(boxes[j][2] * width - boxes[j][0] * width),
          int(boxes[j][3] * height - boxes[j][1] * height)
        ]
        score = float(scores[j])

        output['annotations'].append({
          'id': 1 + i + j,
          'image_id': image_id,
          'category_id': category_id,
          'bbox': bbox,
          'score': score
        })
    
  with open('annotations.json', 'w') as outfile:
    json.dump(output, outfile)

if __name__ == '__main__':
  try:
    app.run(main)
  except SystemExit:
    pass
