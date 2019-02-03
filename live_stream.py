from imageai.Detection import VideoObjectDetection
import os
import cv2

execution_path = os.getcwd()


camera = cv2.VideoCapture(0)

def forFrame(frame_number, output_array, output_count):
    print("FOR FRAME " , frame_number)
    print("Output for each object : ", output_array)
    print("Output count for unique objects : ", output_count)
    print("------------END OF A FRAME --------------")

def forSeconds(second_number, output_arrays, count_arrays, average_output_count):
    print("SECOND : ", second_number)
    print("Array for the outputs of each frame ", output_arrays)
    print("Array for output count for unique objects in each frame : ", count_arrays)
    print("Output average count for unique objects in the last second: ", average_output_count)
    print("------------END OF A SECOND --------------")

def forMinute(minute_number, output_arrays, count_arrays, average_output_count):
    print("MINUTE : ", minute_number)
    print("Array for the outputs of each frame ", output_arrays)
    print("Array for output count for unique objects in each frame : ", count_arrays)
    print("Output average count for unique objects in the last minute: ", average_output_count)
    print("------------END OF A MINUTE --------------")

detector = VideoObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath(os.path.join(execution_path , "yolo.h5"))
detector.loadModel(detection_speed="flash")


video_path = detector.detectObjectsFromVideo(camera_input=camera,
                                output_file_path=os.path.join(execution_path, "live_stream_super.mp4")
                                , frames_per_second=15, log_progress=True, per_second_function = forSeconds, per_frame_function = forFrame, per_minute_function = forMinute, minimum_percentage_probability = 30)


