import torch
import torch.nn as nn
import torchvision.models as models
from tensorboardX import SummaryWriter
#from torch_deform_conv.layers import ConvOffset2D
#from torch.utils.tensorboard import SummaryWriter


vgg_model_1 = models.vgg16(pretrained=True).features[:-1]
# vgg_model_1[-2] = ConvOffset2D(filters=512)
# vgg_model_1[-4] = ConvOffset2D(filters=512)
# vgg_model_1[-6] = ConvOffset2D(filters=512)

vgg_model_2 = models.vgg16(pretrained=True).features[:16]
# vgg_model_2[-2] = ConvOffset2D(filters=256)

def conv_relu(in_dim, out_dim):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),nn.ReLU(inplace=True),)

class DSS(nn.Module):

    # pretrained_model = \
    #     osp.expanduser('~/data/models/pytorch/fcn8s_from_caffe.pth')
    #
    # @classmethod
    # def download(cls):
    #     return fcn.data.cached_download(
    #         url='http://drive.google.com/uc?id=0B9P1L--7Wd2vT0FtdThWREhjNkU',
    #         path=cls.pretrained_model,
    #         md5='dbd9bbb3829a3184913bccc74373afbb',
    #     )

    def __init__(self):
        super(DSS, self).__init__()
        ####path1
        self.vgg1 = vgg_model_1
        self.upscore1_6 = nn.ConvTranspose2d(512, 512, 2, stride=2, bias=False)
        self.conv1_6 = conv_relu(512,512)  ###########concat

        self.upscore1_7 = nn.ConvTranspose2d(1024, 512, 2, stride=2, bias=False)
        self.conv1_7 = conv_relu(512,512)#######concat

        self.conv1_8_1 = conv_relu(768, 256)
        self.conv1_8_2 = conv_relu(256, 256)
        self.conv1_8_3 = conv_relu(256, 256)

        self.upscore1_9 = nn.ConvTranspose2d(256, 256, 2, stride=2, bias=False)
        self.conv1_9_1 = conv_relu(256, 128)
        self.conv1_9_2 = conv_relu(128, 128)
        self.upscore1_10 = nn.ConvTranspose2d(128, 128, 2, stride=2, bias=False)
        self.conv1_10_1 = conv_relu(128, 64)
        self.conv1_10_2 = conv_relu(64, 64)
        self.pre_phase_1=nn.Sequential(nn.Conv2d(64, 1, kernel_size=1, padding=0),nn.Sigmoid(),)

        ####path2
        self.vgg2 = vgg_model_2
        self.upscore2_4 = nn.ConvTranspose2d(256, 256, 2, stride=2, bias=False)
        self.conv2_4 = conv_relu(256, 256)
        self.upscore2_5 = nn.ConvTranspose2d(384, 256, 2, stride=2, bias=False)
        self.conv2_5 = conv_relu(256, 256)
        self.conv2_6_1 = conv_relu(320, 64)
        self.conv2_6_2 = conv_relu(64, 64)
        self.pre_phase_2 = nn.Sequential(nn.Conv2d(64, 1, kernel_size=1, padding=0), nn.Sigmoid(), )
        ####Final prediction phase
        self.pre_fina = nn.Sequential(nn.Conv2d(1, 1, kernel_size=1, padding=0), nn.Sigmoid(), )



    def forward(self, x):
        ####path_1
        feature_1 = self.vgg1(x)
        feature_1_16 = self.vgg1[:16](x)
        feature_1_23 = self.vgg1[:23](x)
        feature_1=self.upscore1_6(feature_1)
        feature_1=self.conv1_6(feature_1)
        feature_1=torch.cat((feature_1,feature_1_23),1)
        feature_1=self.upscore1_7(feature_1)
        feature_1=self.conv1_7(feature_1)
        feature_1=torch.cat((feature_1,feature_1_16),1)
        feature_1=self.conv1_8_1(feature_1)
        feature_1 = self.conv1_8_2(feature_1)
        feature_1 = self.conv1_8_3(feature_1)
        feature_1 = self.upscore1_9(feature_1)
        feature_1 = self.conv1_9_1(feature_1)
        feature_1 = self.conv1_9_2(feature_1)
        feature_1 = self.upscore1_10(feature_1)
        feature_1 = self.conv1_10_1(feature_1)
        feature_1 = self.conv1_10_2(feature_1)
        out_1=self.pre_phase_1(feature_1)


        ####path_2
        feature_2 = self.vgg2(x)
        feature_2_4=self.vgg2[:4](x)
        feature_2_9=self.vgg2[:9](x)
        feature_2=self.upscore2_4(feature_2)
        feature_2 = self.conv2_4(feature_2)
        feature_2 = torch.cat((feature_2, feature_2_9), 1)
        feature_2 = self.upscore2_5(feature_2)
        feature_2 = self.conv2_5(feature_2)
        feature_2 = torch.cat((feature_2, feature_2_4), 1)
        feature_2 = self.conv2_6_1(feature_2)
        feature_2 = self.conv2_6_2(feature_2)
        out_2=self.pre_phase_2(feature_2)

        out=out_1+out_2
        #out=self.pre_fina(out)



        return out


if __name__ == "__main__":
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #image_size = 64
    # x = torch.rand(1, 3, 256,256)
    # # x.to(device)
    # print("x size: {}".format(x.size()))
    #
    model =DSS()
    # with SummaryWriter(comment='DSSNet') as w:
    #      w.add_graph(model, (x,))
    # out = model(x)
    print(model.pre_fina)
    print("out size: {}".format(out.size()))