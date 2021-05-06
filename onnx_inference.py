import onnxruntime
from PIL import Image
import numpy as np
import cv2

def image_to_input(img):
    img = img.resize((512,512), Image.CUBIC)
    image = np.array(img)

    tmpImg = np.zeros((image.shape[0],image.shape[1],3))
    image = image/np.max(image)

    tmpImg = np.zeros((image.shape[0],image.shape[1],3))
    image = image/np.max(image)            
    tmpImg[:,:,0] = (image[:,:,0]-0.485)/0.229
    tmpImg[:,:,1] = (image[:,:,1]-0.456)/0.224
    tmpImg[:,:,2] = (image[:,:,2]-0.406)/0.225
    tmpImg = tmpImg.transpose((2, 0, 1))
    
    return tmpImg

ort_session = onnxruntime.InferenceSession("u2net_portrait.onnx")

img = Image.open('1.jpg').convert('RGB')

onnx_input = image_to_input(img)
onnx_input = np.expand_dims(onnx_input, axis=0)
onnx_input = np.array(onnx_input, dtype=np.float32)
print(onnx_input.shape)
print(onnx_input.dtype)

ort_inputs = {ort_session.get_inputs()[0].name: onnx_input}

ort_outs = ort_session.run(None, ort_inputs)