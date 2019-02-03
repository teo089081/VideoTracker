from imageai.Detection import VideoObjectDetection
import os
import time

execution_path = os.getcwd()
s = time.time()
detector = VideoObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath( os.path.join(execution_path , "yolo.h5"))
detector.loadModel(detection_speed = "faster")

custom_objects = detector.CustomObjects(person=True)

video_path = detector.detectCustomObjectsFromVideo(custom_objects=custom_objects, input_file_path=os.path.join(execution_path, "1080p_test.mp4"),
                                output_file_path=os.path.join(execution_path, "custom_detected1080")
                                , frames_per_second=30, log_progress=True)
print(video_path)
e = time.time()
timetaken = e-s
print(timetaken)
