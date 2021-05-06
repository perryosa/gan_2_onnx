import os
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms#, utils
# import torch.optim as optim

import numpy as np
from PIL import Image
import glob

from data_loader import RescaleT
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from model import U2NET # full size version 173.6 MB
from model import U2NETP # small version u2net 4.7 MB

# normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn

def save_output(image_name,pred,d_dir):

    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()

    im = Image.fromarray(predict_np*255).convert('RGB')
    img_name = image_name.split(os.sep)[-1]
    image = io.imread(image_name)
    imo = im.resize((image.shape[1],image.shape[0]),resample=Image.BILINEAR)

    pb_np = np.array(imo)

    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1,len(bbb)):
        imidx = imidx + "." + bbb[i]

    imo.save(d_dir+'/'+imidx+'.png')

# --------- 1. get image path and name ---------
model_name='u2net_portrait'#u2netp


image_dir = './custom_inputs'
prediction_dir = './custom_inputs_results'
if(not os.path.exists(prediction_dir)):
    os.mkdir(prediction_dir)

model_dir = './saved_models/u2net_portrait/u2net_portrait.pth'

img_name_list = glob.glob(image_dir+'/*')
print("Number of images: ", len(img_name_list))

# --------- 2. dataloader ---------
#1. dataloader
test_salobj_dataset = SalObjDataset(img_name_list = img_name_list,
                                    lbl_name_list = [],
                                    transform=transforms.Compose([RescaleT(512),
                                                                    ToTensorLab(flag=0)])
                                    )
test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                    batch_size=1,
                                    shuffle=False,
                                    num_workers=1)

# --------- 3. model define ---------

print("...load U2NET---173.6 MB")
net = U2NET(3,1)

net.load_state_dict(torch.load(model_dir, map_location=torch.device('cpu')))
if torch.cuda.is_available():
    net.cuda()
net.eval()

# --------- 4. inference for each image ---------
for i_test, data_test in enumerate(test_salobj_dataloader):

    print("inferencing:",img_name_list[i_test].split(os.sep)[-1])

    inputs_test = data_test['image']
    print(inputs_test.shape)
    inputs_test = inputs_test.type(torch.FloatTensor)

    if torch.cuda.is_available():
        inputs_test = Variable(inputs_test.cuda())
    else:
        inputs_test = Variable(inputs_test)


    d1 = net(inputs_test)

    print(d1.shape)
    # normalization
    pred = 1.0 - d1[:,0,:,:]
    pred = normPRED(pred)
    print(pred.shape)
    print(type(img_name_list[i_test]))

    # save results to test_results folder
    save_output(img_name_list[i_test],pred,prediction_dir)

    del d1


# Input to the model
batch_size = 1
x = torch.randn(batch_size, 3, 512, 512, requires_grad=False)
torch_out = net(x)

print(next(net.parameters())[0][0][0][0].dtype)

# Export the model
torch.onnx.export(net,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  "u2net_portrait.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=11,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                                'output' : {0 : 'batch_size'}})

