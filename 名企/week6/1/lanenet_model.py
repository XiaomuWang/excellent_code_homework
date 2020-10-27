import torch
import torch.nn as  nn
import decoder_fcn
from resnet import resnet18,resnet34
#-------------------------------------以ReNet为骨架，由一个编码和一个解码组成的lanet网络----------------------------------------------------
class LaneNet_FCN_Res_1E2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=resnet18(pretrained=True)
        self.decoder_embed = decoder_fcn.Decoder_LaneNet_TConv_Embed()
        self.decoder_logit = decoder_fcn.Decoder_LaneNet_TConv_Logit()

    def forward(self,input):
        x=self.encoder.forward(input)
        input_tensor_list = [self.encoder.c1, self.encoder.c2, self.encoder.c3, self.encoder.c4, x]
        embedding = self.decoder_embed.forward(input_tensor_list)
        logit = self.decoder_logit.forward(input_tensor_list)

        return embedding, logit
if __name__ == '__main__':
    #-------------------------------网络能否正常运行的测试代码-----------------------------------------------
    img = torch.randn(1, 3, 512, 512)
    model = LaneNet_FCN_Res_1E2D()
    embedding, logit = model(img)

    print(embedding.shape)