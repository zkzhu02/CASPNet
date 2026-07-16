import torch.nn as nn
from function import adaptive_instance_normalization as adain
from function import calc_mean_std  
from function import mean_variance_norm,normal
import torch.nn.functional as F
import torch



class SRNet(nn.Module):
    def __init__(self, in_channels):
        super(SRNet, self).__init__()

        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, in_channels, kernel_size=3),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, in_channels, kernel_size=3),
            nn.ReLU(inplace=True)
        )

        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        
        self.conv3 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, in_channels, kernel_size=3),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.avgpool(out)
        out = self.conv3(out)
        return out

class FinalModulator(nn.Module):
    def __init__(self, style_dim=256, target_dim=256):
        super(FinalModulator, self).__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(style_dim * 2, style_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(style_dim, style_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(style_dim, target_dim * 2, 1) # 输出 w 和 b
        )

    def forward(self, style_feat):
        mu = style_feat.mean(dim=[2, 3], keepdim=True)
        std = style_feat.std(dim=[2, 3], keepdim=True)
        stats = torch.cat([mu, std], dim=1)
        params = self.mlp(stats)
        w, b = torch.split(params, params.size(1) // 2, dim=1)
        return w, b

class ModulatedDecoder(nn.Module):
    def __init__(self):
        super(ModulatedDecoder, self).__init__()
        self.srnet = SRNet(in_channels=256)
        self.modulator = FinalModulator(style_dim=256, target_dim=256)
        self.block1 = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 256, (3, 3)),
            nn.ReLU()
        )
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.block2 = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 128, (3, 3)),
            nn.ReLU()
        )
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.block3 = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 128, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 64, (3, 3)),
            nn.ReLU()
        )
        self.upsample3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.block4 = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 3, (3, 3))
        )

    def forward(self, x, style_feat3):
        x = self.block1(x)
        style_srnet = self.srnet(style_feat3)
        w, b = self.modulator(style_srnet)
        x = x *  w + b
        x = self.upsample1(x)
        x = self.block2(x)
        x = self.upsample2(x)
        x = self.block3(x)
        x = self.upsample3(x)
        x = self.block4(x)
        return x





vgg = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1, this is the last layer used
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU()  # relu5-4
)



 




class SP(nn.Module):
    def __init__(self, channels=512):
        super(SP, self).__init__()
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        self.out_conv = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, c, s):

        combined = torch.cat([c, s], dim=1) 
        mask = self.spatial_gate(combined) 
        

        out = c * mask + s * (1 - mask)
        

        return self.out_conv(out)
    
class SA(nn.Module):
    def __init__(self, in_planes):
        super(SA, self).__init__()
        self.f = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.g = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.h = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.sm = nn.Softmax(dim=-1)
        self.out_conv = nn.Conv2d(in_planes, in_planes, (1, 1))

        self.sp = SP()

        self.adapter_layer2 = nn.Sequential(
            nn.Conv2d(128, in_planes, 1, 1, 0),
            nn.ReLU(inplace=True)
        )
        self.adapter_layer3 = nn.Sequential(
            nn.Conv2d(256, in_planes, 1, 1, 0),
            nn.ReLU(inplace=True)
        )
        
        # 融合层：输入 512*3 -> 输出 512
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_planes * 3, in_planes, 1, 1, 0),
        )

    def forward(self, content, style, style_feat2, style_feat3):
        F_feat = self.f(mean_variance_norm(content))
        G_feat = self.g(mean_variance_norm(style))
        H_feat = self.h(style)
        

        b, c, h, w = F_feat.size()
        bG, cG, hG, wG = G_feat.size()


        F_view = F_feat.view(b, -1, w * h).permute(0, 2, 1)
        G_view = G_feat.view(bG, -1, wG * hG)
        

        S = torch.bmm(F_view, G_view) 
        S = self.sm(S) 
        

        H_view = H_feat.view(bG, -1, wG * hG)
        O_current = torch.bmm(H_view, S.permute(0, 2, 1))
        O_current = O_current.view(b, c, h, w)


        s2 = F.interpolate(style_feat2, size=(hG, wG), mode='bilinear', align_corners=False)
        s2 = self.adapter_layer2(s2)
        s2_flat = s2.view(b, -1, hG * wG)
        

        O_layer2 = torch.bmm(s2_flat, S.permute(0, 2, 1))
        O_layer2 = O_layer2.view(b, c, h, w)


        s3 = F.interpolate(style_feat3, size=(hG, wG), mode='bilinear', align_corners=False)
        s3 = self.adapter_layer3(s3)
        s3_flat = s3.view(b, -1, hG * wG)
        
        O_layer3 = torch.bmm(s3_flat, S.permute(0, 2, 1))
        O_layer3 = O_layer3.view(b, c, h, w)


        O_cat = torch.cat([O_current, O_layer2, O_layer3], dim=1)
        O_fused = self.fusion_conv(O_cat)


        glb = adain(O_fused, style)
        result = self.sp(content, glb)
        
        return result
    
class SRSFM(nn.Module):
    def __init__(self):
        super(SRSFM,self).__init__()
        self.sr=SR()
        self.sa=SA(512)
        
    def forward(self, content_feats,style_feats,s1,s2):
        wei_s=self.sr(style_feats)
        style=style_feats*wei_s
        result=self.sa(content_feats,style,s1,s2)
        return result








    
class Net(nn.Module):
    def __init__(self, encoder, decoder,SRSFM):
        super(Net, self).__init__()
        enc_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1
        self.enc_5 = nn.Sequential(*enc_layers[31:44])  # relu4_1 -> relu5_1
        self.decoder = decoder
        self.mse_loss = nn.MSELoss()
        self.SRSFM= SRSFM
        self.max_sample = 64 * 64
        self.seed = 6666
        self.up=nn.Upsample(size=(32,32), mode='bilinear', align_corners=False)

        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4','enc_5']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

    # extract relu1_1, relu2_1, relu3_1, relu4_1 relu5_1 from input image
    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(5):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    # extract relu5_1 from input image
    def encode(self, input):
        for i in range(5):
            input = getattr(self, 'enc_{:d}'.format(i + 1))(input)
        return input

    def calc_content_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        return self.mse_loss(input, target)

    def calc_style_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + \
            self.mse_loss(input_std, target_std)
    
    
    
    def ls_loss(self, c_feats, s_feats, stylized_feats):
        ls = 0

        return ls
    
    
    def forward(self, contents, styles, alpha=1.0):
        assert 0 <= alpha <= 1
        style_feats = self.encode_with_intermediate(styles)
        content_feats = self.encode_with_intermediate(contents)
        s_feat_2 = style_feats[1] 
        s_feat_3 = style_feats[2]

        
        result1 = self.SRSFM(content_feats[-1], style_feats[-1], s_feat_2, s_feat_3)
        m=self.up(result1)
        result2 = self.SRSFM(content_feats[-2], style_feats[-2], s_feat_2, s_feat_3)
        result=result2+m

        cc_result1 = self.SRSFM(content_feats[-1],content_feats[-1],content_feats[1],content_feats[2])
        cc_result2 = self.SRSFM(content_feats[-2],content_feats[-2],content_feats[1],content_feats[2])
        cc_result1=self.up(cc_result1)
        cc_result=cc_result2+cc_result1

        ss_result1 = self.SRSFM(style_feats[-1], style_feats[-1],style_feats[1],style_feats[2])
        ss_result2 = self.SRSFM(style_feats[-2], style_feats[-2],style_feats[1],style_feats[2])
        ss_result1=self.up(ss_result1)
        ss_result=ss_result1+ss_result2


        g_t = self.decoder(result,style_feats[2])
        r_cc = self.decoder(cc_result,content_feats[2])
        r_ss = self.decoder(ss_result,style_feats[2])

       
        
        g_t_feats = self.encode_with_intermediate(g_t)
        
        loss_c = self.calc_content_loss(normal(g_t_feats[-2]),normal(content_feats[-2]))+self.calc_content_loss(normal(g_t_feats[-1]),normal(content_feats[-1]))
        loss_s = self.calc_style_loss(g_t_feats[0], style_feats[0])
        for i in range(1, 5):
            loss_s += self.calc_style_loss(g_t_feats[i], style_feats[i])

        loss_lambda1 = self.calc_content_loss(r_cc, contents) + self.calc_content_loss(r_ss, styles)
        Icc_feats = self.encode_with_intermediate(r_cc)
        Iss_feats = self.encode_with_intermediate(r_ss)
        loss_lambda2 = self.calc_content_loss(Icc_feats[0], content_feats[0]) + self.calc_content_loss(Iss_feats[0],style_feats[0])
        for i in range(1, 5):
            loss_lambda2 += self.calc_content_loss(Icc_feats[i], content_feats[i]) + self.calc_content_loss(Iss_feats[i], style_feats[i])
            
        
        ls = self.ls_loss(content_feats, style_feats, g_t_feats)


        return ls,loss_lambda1, loss_lambda2, loss_c, loss_s, g_t
