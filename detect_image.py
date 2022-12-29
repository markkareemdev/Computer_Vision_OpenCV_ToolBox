import cv2
import numpy as np

# initialize buttons
# from gui_buttons import Buttons

# button = Buttons()
# button.add_button("")

# openCV DNN
# download zip file here
# http://pysource.com/download/crash_course_ods.zip

net = cv2.dnn.readNet("dnn_model/yolov4-tiny.weights", "dnn_model/yolov4-tiny.cfg")
model = cv2.dnn_DetectionModel(net)

# note
# size tells the deep learning to scale the image to that size 
# The smaller the image the faster the deetection but lower the accuracy and vice versal
# scale: opencv peak is from 1 to 255, while od deep learnig neural network the value goes from 0 to 1
# scale of 1/25, scales opencv to neural network image processing limits

model.setInputParams(
    size=(320,320),
    # size=(416,416),
    scale=1/255 
       )

# loaded classes
classes = []
with open("dnn_model/classes.txt", "r") as file_object:
    for class_name in file_object.readlines():
        classes.append(class_name.strip())





# initialize camera
cap = cv2.VideoCapture(0)

#  adjust camera height and width for better solution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# event windows

button_person = False

def click_button(event, x, y, flags, params):

    global button_person

    if event == cv2.EVENT_LBUTTONDOWN:
        polygon = np.array([[(20,20), (220, 20), (220, 70), (20, 70)]])
        is_inside = cv2.pointPolygonTest(polygon, (x,y), False)
        if is_inside > 0:
            print('we clicking inside ', (x,y))

            if button_person is False:
                button_person = True
            else:
                button_person = False

            print(button_person)


# create window
cv2.namedWindow("frame")
cv2.setMouseCallback("frame", click_button )



while True:

    # Get frames
    ret, frame = cap.read()

    # detect objects
    # when we detect an image we detect the classids of the image, score n bboxes
    class_ids, scores, bboxes = model.detect(frame)

    # draw a rectangle, put text on frame
    for class_ids, score, bbox in zip(class_ids, scores, bboxes):
        
        x,y,w,h = bbox
        class_name = classes[class_ids]

        if button_person:
            cv2.putText(
                frame,
                class_name,
                (x, y-10),
                cv2.FONT_HERSHEY_PLAIN,
                1,
                (200,0, 50),
                2
            )
            cv2.rectangle(
                frame, 
                (x,y),
                (x+w, y+h),
                (200, 0, 50),
                3
            )


        # create a button

        # cv2.rectangle(frame, (20,20), (150, 70), (0,0,200), -1)

        polygon = np.array([[(20,20), (220, 20), (220, 70), (20, 70)]])
        cv2.fillPoly(frame, polygon, (0,0,200))

        cv2.putText(frame, class_name, (30, 60), cv2.FONT_HERSHEY_PLAIN, 3, (255,255,255), 3)
         
 


    cv2.imshow('frame', frame)
    cv2.waitKey(1)

