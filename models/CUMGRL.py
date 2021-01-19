import torch
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import torch.nn as nn
from embedder import embedder
from layers import GCN, Discriminator, Attention
import numpy as np
np.random.seed(0)
from evaluate import evaluate
from models import LogReg
import pickle as pkl
from  utils import  process
class CUMGRL(embedder):
    def __init__(self, args):
        embedder.__init__(self, args)
        self.args = args

    def training(self):
        torch.cuda.empty_cache()
        features = [feature.to(self.args.device) for feature in self.features]
        adj = [adj_.to(self.args.device) for adj_ in self.adj]
        model = modeler(self.args).to(self.args.device)
        optimiser = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=self.args.l2_coef)
        cnt_wait = 0; best = 1e9
        b_xent = nn.BCEWithLogitsLoss()
        xent = nn.CrossEntropyLoss()
     
        # return
    #    xent_consensus = nn.CrossEntropyLoss()
        for epoch in range(self.args.nb_epochs):
            print(epoch)
            xent_loss = None
            xent_consensus_loss = None
            reconstruct_loss = None
            model.train()
            optimiser.zero_grad()
            idx = np.random.permutation(self.args.nb_nodes)

            shuf = [feature[:, idx, :] for feature in features]
            shuf = [shuf_ft.to(self.args.device) for shuf_ft in shuf]

            lbl_1 = torch.ones(self.args.batch_size, self.args.nb_nodes)
            lbl_2 = torch.zeros(self.args.batch_size, self.args.nb_nodes)
            lbl = torch.cat((lbl_1, lbl_2), 1).to(self.args.device)

            result = model(features, adj, shuf, self.args.sparse, None, None, None)
            logits = result['logits']
            logits_consensus = result['logits_consensus']
           

            for view_idx, logit in enumerate(logits):
                if xent_loss is None:
                    xent_loss = b_xent(logit, lbl)
                else:
                    xent_loss += b_xent(logit, lbl)
     
            loss = xent_loss
            for view_idx, logit in enumerate(logits_consensus):
                if xent_consensus_loss is None:
                    xent_consensus_loss = b_xent(logit, lbl)

                else:
                    xent_consensus_loss+= b_xent(logit, lbl)
           
            loss =loss+self.args.reg_coef * xent_consensus_loss

          

            if loss < best:
                best = loss
                cnt_wait = 0
                torch.save(model.state_dict(), 'saved_model/best_{}_{}_{}.pkl'.format(self.args.dataset, self.args.embedder, self.args.metapaths))
            else:
                cnt_wait += 1

            if cnt_wait == self.args.patience:
                break
            torch.cuda.empty_cache()
            loss.backward()
            optimiser.step()

        model.load_state_dict(torch.load('saved_model/best_{}_{}_{}.pkl'.format(self.args.dataset, self.args.embedder, self.args.metapaths)))

        # Evaluation
        model.eval()
        idx = np.random.permutation(self.args.nb_nodes)

        shuf = [feature[:, idx, :] for feature in features]
        shuf = [shuf_ft.to(self.args.device) for shuf_ft in shuf]

        lbl_1 = torch.ones(self.args.batch_size, self.args.nb_nodes)
        lbl_2 = torch.zeros(self.args.batch_size, self.args.nb_nodes)
        lbl_1 =lbl_1.to(self.args.device)


        result = model(features, adj, shuf, self.args.sparse, None, None, None)
        logits_consensus = result['logits_consensus']









#         weight = {}
#         tmp_sum = 0
#         for i in range(self.args.nb_graphs):
#             weight[i]=0
#         for view_idx, logit in enumerate(logits_consensus):

#             weight[int((view_idx+1)/self.args.nb_graphs)] += 1-b_xent(logit, lbl).item()
   

#         
#         tmp_sum = 0
#         for i in range(self.args.nb_graphs):
#             weight[i]= weight[i]
#             tmp_sum =tmp_sum+weight[i]
#         for i in range(self.args.nb_graphs):
#             weight[i] = weight[i] / tmp_sum
#         
#         #tmp = model.get_emb(features, adj, self.args.sparse, None, None, None).detach()
#         #print(tmp.shape)

#         evaluate(model.get_emb(features, adj, self.args.sparse, None, None, None,weight).detach(), self.idx_train, self.idx_val, self.idx_test, self.labels, self.args.device,filename=self.args.embedder+self.args.dataset+'weighted_1')







        for i in range(self.args.nb_graphs):
            weight[i]=1
        for i in range(self.args.nb_graphs):
            weight[i] = weight[i] / self.args.nb_graphs
        evaluate(model.get_emb(features, adj, self.args.sparse, None, None, None, weight).detach(), self.idx_train,
                 self.idx_val, self.idx_test, self.labels, self.args.device,
                 filename=self.args.embedder + self.args.dataset)


class modeler(nn.Module):
    def __init__(self, args):
        super(modeler, self).__init__()
        self.args = args
        self.gcn = nn.ModuleList([GCN(args.ft_size, args.hid_units, args.activation, args.drop_prob, args.isBias) for _ in range(args.nb_graphs)])
        #self.gcn = GCN(args.ft_size, args.hid_units, args.activation, args.drop_prob, args.isBias)
        self.disc = Discriminator(args.hid_units)
        
       # self.disc2 = Discriminator(args.hid_units)  # 用于一致性约束
        self.sigm =nn.Sigmoid()
        self.H = nn.Parameter(torch.FloatTensor(1, args.nb_nodes, args.hid_units))
        self.readout_func = self.args.readout_func
        if args.isAttn:
            self.attn = nn.ModuleList([Attention(args) for _ in range(args.nheads)])

        if args.isSemi:
            self.logistic = LogReg(args.hid_units, args.nb_classes).to(args.device)

        self.init_weight()

    def init_weight(self):
        nn.init.xavier_normal_(self.H)
    def get_emb(self, feature, adj, sparse, msk, samp_bias1, samp_bias2,weight):
        h=None
        for i in range(self.args.nb_graphs):
            if(h is None):
                h = self.gcn[i](feature[i], adj[i], sparse)*weight[i]
            else:
                h = h+self.gcn[i](feature[i], adj[i], sparse)*weight[i]

        return h/self.args.nb_graphs
         
           
        
    def forward(self, feature, adj, shuf, sparse, msk, samp_bias1, samp_bias2):
        h_1_all = []; h_2_all = []; c_all = []; logits = []; logits_consensus = []
        result = {}
       
        for i in range(self.args.nb_graphs):
            h_1 = self.gcn[i](feature[i], adj[i], sparse)
            # how to readout positive summary vector
            c = self.readout_func(h_1)
            c = self.args.readout_act_func(c)  # equation 9
            h_2 = self.gcn[i](shuf[i], adj[i], sparse)
            logit = self.disc(c, h_1, h_2, samp_bias1, samp_bias2)

            h_1_all.append(h_1)
            h_2_all.append(h_2)
            c_all.append(c)
            logits.append(logit)
        result['logits'] = logits
       
        for i in range(self.args.nb_graphs):
            #h_1 = h_1_all[i]  
            #h_2 = h_2_all[i]  
            c = c_all[i]
            for j in range(self.args.nb_graphs):
                if(i==j):
                    continue
                c = c_all[j]
                h_1 = h_1_all[j]  #
                h_2 = h_2_all[j]  #
                logit = self.disc(c, h_1, h_2, samp_bias1, samp_bias2)
                logits_consensus.append(logit)
        result['logits_consensus'] = logits_consensus

        # 
        return result
