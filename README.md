# SEG-GRAD-CAM
pytorch implementation of SEG-GRAD-CAM,which based on grad-cam.It also has 3D SEG-GRAD-CAM which is used for video.
see the paper [Towards Interpretable Semantic Segmentation via Gradient-weighted Class Activation Mapping](https://arxiv.org/abs/2002.11434)
## How to use

### Dependencies
This tutorial depends on the following libraries:

* pytorch
* opencv-python

### usage
[gradcam.py](gradcam.py):which is the formal implementation [GRAD-CAM](https://github.com/jacobgil/pytorch-grad-cam).My work is based on this work.

[gradcam_unet.py](gradcam_unet.py):which is the implementation SEG-GRAD-CAM.We use the model [deep smoke Segmentation](https://arxiv.org/abs/1809.00774)(like Unet).

you can run gradcam_unet.py using model [BaiDuyun](https://pan.baidu.com/s/16IolEoXFZChlTKNo2t5jnA) with password "3d7c" or [google drive](https://drive.google.com/file/d/1MXPr6WDdlj6ZcqjJClmLUPkE6kCGfuXL/view?usp=sharing)

![result/pic_1.jpg](result/pic_1.jpg)
![result/cam_1.jpg](result/cam_1.jpg)

[gradcam_3d.py](gradcam_3d.py):which is the implementation SEG-GRAD-CAM based on 3dunet.It's used for video Activation Mapping.
because the paper haven't been public.So the model will be release soon.But it is not important if you want to vis activation map based on your own model.

[generate_gif.py](generate_gif.py): the input and output data of gradcam_3d.py is pic.If you want to compose a series of pic to gif,you can run generate_gif.py.

![result/video_2.gif](result/video_2.gif)
![result/vis2.gif](result/vis2.gif)

##Question
if you have any question about the code. Please email me 359684740@qq.com

