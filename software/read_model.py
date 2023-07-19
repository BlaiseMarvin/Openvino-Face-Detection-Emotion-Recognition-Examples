from openvino.runtime import Core

ie=Core()
face_detection_model="/home/blaisemarvinrusoke/Desktop/internship/openvino/intel/face-detection-adas-0001/FP32/face-detection-adas-0001.xml"

model=ie.read_model(model=face_detection_model)
compiled_model=ie.compile_model(model=model,device_name="CPU")

# print(f'Model Inputs: {model.inputs}, {compiled_model.inputs}')
print(model.input)

# output_layer=model.output()
# print(output_layer.any_name)