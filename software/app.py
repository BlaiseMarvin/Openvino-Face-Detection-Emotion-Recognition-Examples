import cv2
from openvino.runtime import Core
import numpy as np

cap=cv2.VideoCapture(0)

ie=Core()

model=ie.read_model(model="/home/blaisemarvinrusoke/Desktop/internship/openvino/intel/face-detection-adas-0001/FP32/face-detection-adas-0001.xml")
input_layerx=model.input(0)
compiled_model=ie.compile_model(model=model,device_name="CPU")

input_layer=compiled_model.input(0)
output_layer=compiled_model.output(0)

N,C,H,W=input_layer.shape



while(cap.isOpened()):
    flag,frame = cap.read()

    if not flag:
        break

    # Resize to the model shape.
    input_image = cv2.resize(src=frame, dsize=(W, H))

    # Reshape to model input shape.
    input_image = np.expand_dims(np.transpose(input_image,(2,0,1)),0).astype(np.float32)

    result=compiled_model(input_image)[output_layer]
    print(model.output(index))

    cv2.imshow('frame',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# After the loop release the cap object
cap.release()
# Destroy all the windows
cv2.destroyAllWindows()