import torch
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k_bilinear = nn.Bilinear(n_h,n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        # c:torch.Size([1, 64]) c_x :torch.Size([1, 3550, 64]) h_pl :torch.Size([1, 3550, 64])
        c_x = torch.unsqueeze(c, 1) # c: summary vector, h_pl: positive, h_mi: negative
        c_x = c_x.expand_as(h_pl)
        #c_x =c
       # print(c_x.shape) #[1, 3550, 2000]
      #  print(h_pl.shape)#[1, 3550, 64]

     #   print(h_mi.shape)

        sc_1 = torch.squeeze(self.f_k_bilinear(h_pl, c_x), 2) # sc_1 = 1 x nb_nodes torch.Size([1, 3550])
        sc_2 = torch.squeeze(self.f_k_bilinear(h_mi, c_x), 2) # sc_2 = 1 x nb_nodes torch.Size([1, 3550])

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2
        logits = torch.cat((sc_1, sc_2), 1)

        return logits  #torch.Size([1, 7100])
        
        
        
class Discriminator2(nn.Module): # 借鉴了deep infomax的代码啊
    def __init__(self, n_h1, n_h2):
        super(Discriminator2, self).__init__()
        self.f_k = nn.Bilinear(n_h1,n_h2, 1) # 双线性
        self.act = nn.Sigmoid()

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)
                    # 隐特征，原始特征

    # h_c torch.Size([1, 3550, 64])  h_pl torch.Size([1, 3550, 64])
    def forward(self, h_c, h_pl, sample_list, s_bias1=None, s_bias2=None):
        sc_1 = torch.squeeze(self.f_k(h_pl, h_c), 2) # torch.Size([1, 3550]) 正例的分数
        #sc_1 = self.act(sc_1)
        sc_2_list = []
        for i in range(len(sample_list)): # 从另一个view下选
            h_mi = torch.unsqueeze(h_c[0][sample_list[i]],0) # unsqueeze 第0维度 增加 1。
            sc_2_iter = torch.squeeze(self.f_k(h_mi, h_c), 2) # torch.Size([1, 3550])
            sc_2_list.append(sc_2_iter)
        for i in range(len(sample_list)):# 从当前view下选
            h_mi = torch.unsqueeze(h_c[0][sample_list[i]],0) # unsqueeze 第0维度 增加 1。
            sc_2_iter = torch.squeeze(self.f_k(h_mi, h_c), 2) # torch.Size([1, 3550])
            sc_2_list.append(sc_2_iter)
     #       print(sc_2_iter.shape)

        #print(torch.stack(sc_2_list,0).shape)
        # sc_2list里面每个 的维度torch.Size([1, 3550])
        a=torch.stack(sc_2_list,0)
        b=torch.stack(sc_2_list,1)
        sc_2_stack = torch.squeeze(torch.stack(sc_2_list,0),1)
    #    sc_2 = self.act(sc_2_stack)
        sc_2 = sc_2_stack
        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2
       # print(sc_1.shape)
       # print(sc_2.shape)

        logits = torch.cat((sc_1, sc_2.reshape(1,-1)), 1)
        # logits: 1*17750
        return logits