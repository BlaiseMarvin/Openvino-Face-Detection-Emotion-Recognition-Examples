import cv2
import numpy as np
from openvino.runtime import Core

ie=Core()

f_model="/home/blaisemarvinrusoke/Desktop/internship/openvino/intel/face-detection-adas-0001/FP32/face-detection-adas-0001.xml"
e_model="/home/blaisemarvinrusoke/Desktop/internship/openvino/intel/emotions-recognition-retail-0003/FP32/emotions-recognition-retail-0003.xml"
f_model=ie.read_model(model=f_model)
e_model=ie.read_model(model=e_model)

f_compiled_model=ie.compile_model(model=f_model,device_name="CPU")
e_compiled_model=ie.compile_model(model=e_model,device_name="CPU")


# print(f_model.input)

FN, FC, FH, FW = f_compiled_model.input().shape
EN, EC, EH, EW = e_compiled_model.input().shape

cap=cv2.VideoCapture(0)

def emotion_recognition(face):
    resized_face=cv2.resize(src=face, dsize=(EW, EH))
    new_face=np.expand_dims(np.transpose(resized_face,(2,0,1)),0)
    result = e_compiled_model(new_face)['prob_emotion']

    output=np.argmax(result)
    return output
     

def draw_bounding_boxes(frame,result,width,height):
    try:
        for box in result[0][0]:
            if box[2]>0.5:
                    xmin=int(box[3] * width)
                    ymin=int(box[4] * height)
                    xmax=int(box[5] * width)
                    ymax=int(box[6] * height)

                    face=frame[ymin:ymax,xmin:xmax]

                    output=emotion_recognition(face)
                    if output==0:
                        text='neutral'
                    elif output==1:
                        text='happy'
                    elif output==2:
                        text='sad'
                    elif output==3:
                        text='surprise'
                    elif output==4:
                        text='anger'

                    cv2.putText(frame,text,(xmin,ymin-10),cv2.FONT_HERSHEY_SIMPLEX,0.9,(36,255,12),2)



                    cv2.rectangle(frame,(xmin,ymin),(xmax,ymax),(255,0,255),2)
        
        return frame
    except:
        pass
            


while(cap.isOpened()):
    flag,frame=cap.read()

    width=cap.get(3)
    height=cap.get(4)

    if not flag:
        break

    resized_frame=cv2.resize(src=frame, dsize=(FW, FH))
    new_frame=np.expand_dims(np.transpose(resized_frame,(2,0,1)),0)
    result = f_compiled_model(new_frame)['detection_out']
    
    frame=draw_bounding_boxes(frame,result,width,height)

    try:
        cv2.imshow('capture',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except:
        pass
# After the loop release the cap object
cap.release()
# Destroy all the windows
cv2.destroyAllWindows()

