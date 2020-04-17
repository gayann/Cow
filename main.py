from imageai.Detection import ObjectDetection
import os

exec_path = os.getcwd()
detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath(os.path.join(
	exec_path, "resnet50_coco_best_v2.0.1.h5")
)
detector.loadModel()
custom =  detector.CustomObjects( cow= True,  dog= False, sheep=False)
list = detector.detectObjectsFromImage(
	input_image=os.path.join(exec_path, "objects.jpg"),
	output_image_path=os.path.join(exec_path, "new_objects.jpg"),
	minimum_percentage_probability=40,
	display_percentage_probability=True,
	display_object_name=True,
    extract_detected_objects=True
)
