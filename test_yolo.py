import imghdr

import fire
import os

from PIL import Image

from yolo_detection import YoloDetector
from bbox_painter import BboxPainter


def main(model_path, anchors_path, test_path='images/',
         output_path='images/out', classes_path='model_data/coco_classes.txt'):

    detector = YoloDetector(model_path, anchors_path, classes_path)
    painter = BboxPainter(detector.class_names)

    for image_file in os.listdir(test_path):

        try:
            image_type = imghdr.what(os.path.join(test_path, image_file))
            if not image_type:
                continue
        except IsADirectoryError:
            continue

        image = Image.open(os.path.join(test_path, image_file))
        out_boxes, out_scores, out_classes = detector.detect(image)
        annotated_image = painter.apply_bboxes(image, out_boxes, out_scores,
                                               out_classes)
        annotated_image.save(os.path.join(output_path, image_file), quality=90)


if __name__ == '__main__':
    fire.Fire(main)
