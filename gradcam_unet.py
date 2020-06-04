import argparse
import cv2
import numpy as np
import torch
from torch.autograd import Function
from torchvision import models
from DSS import DSS
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
import os


ckpt_path = 'DSS/105_checkpoint.pth.tar'


model = DSS()
model = torch.nn.DataParallel(model, device_ids=list(range(1))).cuda()
cudnn.benchmark = True

if os.path.isfile(ckpt_path):
    print("=> Loading checkpoint '{}'".format(ckpt_path))
    checkpoint = torch.load(ckpt_path)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)
    print("=> Loaded checkpoint (epoch {})".format(checkpoint['epoch']))
else:
    raise Exception("=> No checkpoint found at '{}'".format(ckpt_path))

model=model.module
# print(model)
class FeatureExtractor():
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):  # 在对x求导时，将梯度保存下来
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x


class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """

    def __init__(self, model, feature_module, target_layers):
        self.model = model
        self.feature_module = feature_module
        self.feature_extractor = FeatureExtractor(self.feature_module, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations = []
        # for name, module in self.model._modules.items():
        #     print(module)
        #     if module == self.feature_module:
        #         target_activations, x = self.feature_extractor(x)
        #     elif "avgpool" in name.lower():
        #         x = module(x)
        #         x = x.view(x.size(0), -1)
        #     else:
        #         print(x.shape)
        #         x = module(x)
        x=model(x)
        #print(x.shape)
        target_activations, x = self.feature_extractor(x)



        return target_activations, x


def preprocess_image(img):
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    preprocessed_img = img.copy()[:, :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = \
        np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    preprocessed_img = torch.from_numpy(preprocessed_img)
    preprocessed_img.unsqueeze_(0)
    input = preprocessed_img.requires_grad_(True)
    return input


def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    cv2.imwrite("cam.jpg", np.uint8(255 * cam))


class GradCam:
    def __init__(self, model, feature_module, target_layer_names, use_cuda):
        self.model = model
        self.feature_module = feature_module
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs(self.model, self.feature_module, target_layer_names)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        if self.cuda:
            features, output = self.extractor(input.cuda())  # [1, 2048, 7, 7], [1, 1000]
        else:
            features, output = self.extractor(input)

        # if index == None:
        #     index = np.argmax(output.cpu().data.numpy())
        # print(index)
        # print(output.size())
        # one_hot = np.zeros((1, (256,256)), dtype=np.float32)  # 1,1000
        # one_hot[0][index] = 1
        # one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        output=torch.where(output>0.5,output,torch.full_like(output, 0))

        if self.cuda:
            one_hot = torch.sum(output)
            #print(one_hot)
        else:
            one_hot = torch.sum(output)

        self.feature_module.zero_grad()
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()  # 1, 2048, 7, 7
        target = features[-1]  # 1,2048,7,7
        target = target.cpu().data.numpy()[0, :]  # 2048, 7, 7

        weights = np.mean(grads_val, axis=(2, 3))[0, :]  # 2048
        cam = np.zeros(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):  # w:weight,target:feature
            cam += w * target[i, :, :]

        cam = np.maximum(cam, 0)  # 7,7
        cam = cv2.resize(cam, input.shape[2:])  # 224,224
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=True,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image-path', type=str, default='./examples/1_10.jpg',
                        help='Input image path')
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print("Using GPU for acceleration")
    else:
        print("Using CPU for computation")

    return args


def deprocess_image(img):
    """ see https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py#L65 """
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)
    return np.uint8(img * 255)


if __name__ == '__main__':
    """ python grad_cam.py <path_to_image>
    1. Loads an image with opencv.
    2. Preprocesses it for VGG19 and converts to a pytorch variable.
    3. Makes a forward pass to find the category index with the highest score,
    and computes intermediate activations.
    Makes the visualization. """

    args = get_args()

    # Can work with any model, but it assumes that the model has a
    # feature method, and a classifier method,
    # as in the VGG models in torchvision.
    # model = models.resnet50(pretrained=True)

    grad_cam = GradCam(model=model, feature_module=model.pre_fina, \
                       target_layer_names=["0"], use_cuda=args.use_cuda)
    # grad_cam = GradCam(model=model, feature_module=model.vgg1, \
    #                    target_layer_names=["29"], use_cuda=args.use_cuda)

    img = cv2.imread(args.image_path, 1)
    img = np.float32(cv2.resize(img, (256, 256))) / 255
    input = preprocess_image(img)

    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested index.
    target_index = None
    mask = grad_cam(input, target_index)
    #print(mask.shape)

    show_cam_on_image(img, mask)
