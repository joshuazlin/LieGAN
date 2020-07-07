import torch
import torch.nn as nn
import torch.nn.parallel
import dcgan
from torch.nn import functional as F

#import torch.nn.module


class Conv2d_S1(nn.Conv2d):
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride = 1,
        padding = 0,
        dilation = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',  # TODO: refine this type
        clip = None
    ):
        super(Conv2d_S1,self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, 
            groups, bias, padding_mode)
        self.clip = clip
    
    def _conv_forward(self,input,weight):
        #print("reaching _conv_forward")
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            self.clip*torch.sin(weight), self.bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, self.clip*torch.sin(weight), self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
    
    def forward(self, input):
        return self._conv_forward(input, self.weight)
    
    
class BatchNorm2d_S1(nn.BatchNorm2d):
    def __init__(self, 
                 num_features, 
                 eps=1e-5, 
                 momentum=0.1, 
                 affine=True,
                 track_running_stats=True,
                 clip=None
    ):
        super(BatchNorm2d_S1,self).__init__(
            num_features,eps,momentum,affine,track_running_stats)
        self.clip = clip
    
    def forward(self,input):
        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        """ Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        """Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        """
        TESTING CLIPPING THE BIAS TOO; no idea why that would help
        """
        return F.batch_norm(
            input,
            # If buffers are not to be tracked, ensure that they won't be updated
            self.running_mean if not self.training or self.track_running_stats else None,
            self.running_var if not self.training or self.track_running_stats else None,
            self.clip*torch.sin(self.weight), self.clip*torch.sin(self.bias), bn_training, exponential_average_factor, self.eps)

        


class DCGAN_D_LIE(nn.Module):
    def __init__(self, isize, nz, nc, ndf, ngpu, n_extra_layers=0,clip=None):
        #print("what")
        super(DCGAN_D_LIE, self).__init__()
        #print("how")
        self.ngpu = ngpu
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        main = nn.Sequential()
        # input is nc x isize x isize
        #print("whats going on")
        main.add_module('initial:{0}-{1}:conv'.format(nc, ndf),
                        Conv2d_S1(nc, ndf, 4, 2, 1, bias=False,clip=clip))
        #print("how is this happening")
        main.add_module('initial:{0}:relu'.format(ndf),
                        nn.LeakyReLU(0.2, inplace=True))
        csize, cndf = isize / 2, ndf

        # Extra layers
        for t in range(n_extra_layers):
            main.add_module('extra-layers-{0}:{1}:conv'.format(t, cndf),
                            Conv2d_S1(cndf, cndf, 3, 1, 1, bias=False,clip=clip))
            main.add_module('extra-layers-{0}:{1}:batchnorm'.format(t, cndf),
                            BatchNorm2d_S1(cndf,clip=clip))
            main.add_module('extra-layers-{0}:{1}:relu'.format(t, cndf),
                            nn.LeakyReLU(0.2, inplace=True))

        while csize > 4:
            in_feat = cndf
            out_feat = cndf * 2
            main.add_module('pyramid:{0}-{1}:conv'.format(in_feat, out_feat),
                            Conv2d_S1(in_feat, out_feat, 4, 2, 1, bias=False,clip=clip))
            main.add_module('pyramid:{0}:batchnorm'.format(out_feat),
                            BatchNorm2d_S1(out_feat,clip=clip))
            main.add_module('pyramid:{0}:relu'.format(out_feat),
                            nn.LeakyReLU(0.2, inplace=True))
            cndf = cndf * 2
            csize = csize / 2

        # state size. K x 4 x 4
        main.add_module('final:{0}-{1}:conv'.format(cndf, 1),
                        Conv2d_S1(cndf, 1, 4, 1, 0, bias=False,clip=clip))
        self.main = main


    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else: 
            output = self.main(input)
            
        output = output.mean(0)
        return output.view(1)