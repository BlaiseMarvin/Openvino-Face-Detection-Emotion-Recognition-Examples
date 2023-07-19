from openvino.runtime import Core
import cv2
import numpy as np

ie=Core()
face_detection_model="/home/blaisemarvinrusoke/Desktop/internship/openvino/intel/face-detection-adas-0001/FP32/face-detection-adas-0001.xml"
model=ie.read_model(face_detection_model)
compiled_model=ie.compile_model(model=model,device_name="CPU")

input_layer=compiled_model.input()
output_layer=compiled_model.output()

B,C,H,W = input_layer.shape

cap=cv2.VideoCapture(0)

def extract_faces(image,result,width,height):
	for box in result[0][0]:
		if box[2]>0.5:
			
			xmin=int(box[3] * width)
			ymin=int(box[4] * height)
			xmax=int(box[5] * width)
			ymax=int(box[6] * height)
                        
			face=image[ymin:ymax,xmin:xmax]
			
			text='face'

			cv2.putText(image,text,(xmin,ymin-10),cv2.FONT_HERSHEY_SIMPLEX,0.9,(36,255,12),2)
			cv2.rectangle(image,(xmin,ymin),(xmax,ymax),(0,0,255),1)

	return image

while(cap.isOpened()):
    flag,frame=cap.read()

    width=int(cap.get(3))
    height=int(cap.get(4))

    if not flag:
        break

    resized_frame=cv2.resize(src=frame,dsize=(W,H))

    input_data = np.expand_dims(np.transpose(resized_frame,(2,0,1)),0).astype(np.float32)
    
    result=compiled_model(input_data)['detection_out']

    f_image=extract_faces(frame,result,width,height)


    cv2.imshow('frame',f_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# After the loop release the cap object
cap.release()
# Destroy all the windows
cv2.destroyAllWindows()