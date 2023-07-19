from openvino.runtime import Core
import cv2
import numpy as np

ie=Core()

face_detection_model="/home/blaisemarvinrusoke/Desktop/internship/openvino/intel/face-detection-adas-0001/FP32/face-detection-adas-0001.xml"
emotions_recognition_model="/home/blaisemarvinrusoke/Desktop/internship/openvino/intel/emotions-recognition-retail-0003/FP32/emotions-recognition-retail-0003.xml"

f_model=ie.read_model(face_detection_model)
e_model=ie.read_model(emotions_recognition_model)

detection_model=ie.compile_model(model=f_model,device_name="CPU")
recognition_model=ie.compile_model(model=e_model,device_name="CPU")

f_input_layer=detection_model.input()
f_output_layer=detection_model.output()

e_input_layer=recognition_model.input()
e_output_layer=recognition_model.output()



FB,FC,FH,FW=f_input_layer.shape
EB,EC,EH,EW=e_input_layer.shape

# B,C,H,W = input_layer.shape

cap=cv2.VideoCapture(0)


def emotion_algorithm(image,W,H):
    resized_image=cv2.resize(src=image,dsize=(W,H))

    input_data = np.expand_dims(np.transpose(resized_image,(2,0,1)),0).astype(np.float32)
    
    result=recognition_model(input_data)['prob_emotion']

    # print(np.argmax(result))
    return np.argmax(result)

def extract_faces(image,result,width,height):
    for box in result[0][0]:
        if box[2]>0.5:
            xmin=int(box[3] * width)
            ymin=int(box[4] * height)
            xmax=int(box[5] * width)
            ymax=int(box[6] * height)                
            face=image[ymin:ymax,xmin:xmax]
                        
            emotion = emotion_algorithm(face,EW,EH)
            if emotion==0:
                text='neutral'
            elif emotion==1:
                text='happy'
            elif emotion==2:
                text='sad'
            elif emotion==3:
                text='surprise'
            elif emotion==4:
                text='anger'
            cv2.putText(image,text,(xmin,ymin-10),cv2.FONT_HERSHEY_SIMPLEX,0.9,(36,255,12),2)
            cv2.rectangle(image,(xmin,ymin),(xmax,ymax),(0,0,255),1)

    return image

while(cap.isOpened()):
    flag,frame=cap.read()

    width=int(cap.get(3))
    height=int(cap.get(4))

    if not flag:
        break
    
    resized_frame=cv2.resize(src=frame,dsize=(FW,FH))

    input_data = np.expand_dims(np.transpose(resized_frame,(2,0,1)),0).astype(np.float32)
    
    result=detection_model(input_data)['detection_out']

    f_image=extract_faces(frame,result,width,height)


    cv2.imshow('frame',f_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# After the loop release the cap object
cap.release()
# Destroy all the windows
cv2.destroyAllWindows()