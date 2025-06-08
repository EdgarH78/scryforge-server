import os
import cv2
from xml.etree.ElementTree import Element, SubElement, ElementTree
from scryforge.lens import Pipeline, RotateProcessor, ArucoProcessor, DetectionProcessor, ScaleProcessor, ProcessedFrame
from scryforge.detector import Base, Color

class TrainingDataExporter:
    def __init__(self, detector, settings, detection_settings, output_dir="labeled_data"):
        self.pipeline = Pipeline()
        self.pipeline.add_processor(RotateProcessor(settings.rotate_degrees))
        if detection_settings.enabled:
            self.pipeline.add_processor(ArucoProcessor())
            self.pipeline.add_processor(DetectionProcessor(detector))
        self.pipeline.add_processor(ScaleProcessor(settings.scale))

        self.output_dir = output_dir
        os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "annotations"), exist_ok=True)

    def _write_voc_annotation(self, filename, image_shape, bases, annotation_path):
        height, width, depth = image_shape
        annotation = Element('annotation')

        SubElement(annotation, 'filename').text = filename

        size = SubElement(annotation, 'size')
        SubElement(size, 'width').text = str(width)
        SubElement(size, 'height').text = str(height)
        SubElement(size, 'depth').text = str(depth)

        for base in bases:
            obj = SubElement(annotation, 'object')
            SubElement(obj, 'name').text = base.color.value.lower()
            bndbox = SubElement(obj, 'bndbox')
            x, y, w, h = base.bounding_rect
            SubElement(bndbox, 'xmin').text = str(x)
            SubElement(bndbox, 'ymin').text = str(y)
            SubElement(bndbox, 'xmax').text = str(x + w)
            SubElement(bndbox, 'ymax').text = str(y + h)

        tree = ElementTree(annotation)
        tree.write(annotation_path)

    def export_frame(self, frame: np.ndarray, frame_id: int):
        processed: ProcessedFrame = self.pipeline.process(frame)
        filename = f"frame_{frame_id:04d}.jpg"
        img_path = os.path.join(self.output_dir, "images", filename)
        xml_path = os.path.join(self.output_dir, "annotations", filename.replace(".jpg", ".xml"))

        # Save image
        cv2.imwrite(img_path, processed.image)

        # Write annotation
        self._write_voc_annotation(filename, processed.image.shape, processed.bases, xml_path)
        print(f"✅ Exported {filename} with {len(processed.bases)} bases.")

