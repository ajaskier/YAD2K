import colorsys
import random

import numpy as np
from PIL import Image
from keras import backend as K
from keras.models import load_model

from .yad2k.models.keras_yolo import yolo_head, yolo_eval


class YoloDetector:

    def __init__(self, model_path, anchors_path, classes_path):

        with open(classes_path) as f:
            class_names = f.readlines()
        self.class_names = [c.strip() for c in class_names]

        with open(anchors_path) as f:
            anchors = f.readline()
            anchors = [float(x) for x in anchors.split(',')]
            self.anchors = np.array(anchors).reshape(-1, 2)

        #self.sess = K.get_session()
        self.model = load_model(model_path)

        num_classes = len(self.class_names)
        num_anchors = len(self.anchors)

        # TODO: Assumes dim ordering is channel last
        model_output_channels = self.model.layers[-1].output_shape[-1]
        assert model_output_channels == num_anchors * (num_classes + 5), \
            'Mismatch between model and given anchor and class sizes. ' \
            'Specify matching anchors and classes with --anchors_path and ' \
            '--classes_path flags.'

        # Check if model is fully convolutional, assuming channel last order.
        self.model_image_size = self.model.layers[0].input_shape[1:3]
        self.is_fixed_size = self.model_image_size != (None, None)

        self._prepare_color_palette()
        self._prepare_tensors()

    def _prepare_color_palette(self):
        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                colors))
        random.seed(10101)  # Fixed seed for consistent colors across runs.
        random.shuffle(
            colors)  # Shuffle colors to decorrelate adjacent classes.
        random.seed(None)  # Reset seed to default.

    def _prepare_tensors(self):
        # Generate output tensor targets for filtered bounding boxes.
        # TODO: Wrap these backend operations with Keras layers.
        yolo_outputs = yolo_head(self.model.output, self.anchors,
                                 len(self.class_names))
        self.input_image_shape = K.placeholder(shape=(2,))
        self.boxes, self.scores, self.classes = yolo_eval(
            yolo_outputs,
            self.input_image_shape,
            score_threshold=0.3,
            iou_threshold=0.5,
            max_boxes=20
        )

    def detect(self, image):

        if self.is_fixed_size:
            resized_image = image.resize(
                tuple(reversed(self.model_image_size)), Image.BICUBIC)
            image_data = np.array(resized_image, dtype='float32')
        else:
            # Due to skip connection + max pooling in YOLO_v2, inputs must have
            # width and height as multiples of 32.
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            resized_image = image.resize(new_image_size, Image.BICUBIC)
            image_data = np.array(resized_image, dtype='float32')

        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        with K.get_session().as_default() as sess:

            out_boxes, out_scores, out_classes = sess.run(
                [self.boxes, self.scores, self.classes],
                feed_dict={
                    self.model.input: image_data,
                    self.input_image_shape: [image.size[1], image.size[0]],
                    K.learning_phase(): 0
                })

        return out_boxes, out_scores, out_classes
