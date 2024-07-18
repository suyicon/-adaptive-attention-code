import os

import torch
from torch import nn
from nn_layers import BERT
from parameters import args_parser
from utils import PositionalEncoder_fixed, Power_reallocate


class RFC(nn.Module):
    def __init__(self, args):
        super(RFC, self).__init__()
        self.args = args
        self.truncated = args.truncated
        self.pe = PositionalEncoder_fixed()
        ########################################################################################################################
        self.Tmodel = BERT("trx", args.block_size+2*(args.truncated-1), args.block_size, args.d_model_trx, args.N_trx, args.heads_trx, args.dropout, args.custom_attn,args.multclass)
        self.Rmodel = BERT("rec", args.block_class+args.truncated,args.block_size, args.d_model_trx, args.N_trx+1, args.heads_trx, args.dropout, args.custom_attn,args.multclass)
        # self.Rmodel = BERT("rec", args.block_class + args.truncated, args.block_size, args.d_model_trx, args.N_trx + 1,
        #                   args.heads_trx, args.dropout, args.custom_attn, args.multclass)
        ######### Power Reallocation as in deepcode work ###############
        if self.args.reloc == 1:
            self.total_power_reloc = Power_reallocate(args)

    def power_constraint(self, inputs, isTraining, eachbatch, idx=0, direction='fw'):
        # direction = 'fw' or 'fb'
        if isTraining == 1:
            # training
            this_mean = torch.mean(inputs, 0)
            this_std = torch.std(inputs, 0)
        elif isTraining == 0:
            # test
            if eachbatch == 0:
                this_mean = torch.mean(inputs, 0)
                this_std = torch.std(inputs, 0)
                if not os.path.exists('statistics'):
                    os.mkdir('statistics')
                torch.save(this_mean, 'statistics/this_mean' + str(idx) + direction)
                torch.save(this_std, 'statistics/this_std' + str(idx) + direction)
            elif eachbatch <= 100:
                this_mean = torch.load('statistics/this_mean' + str(idx) + direction) * eachbatch / (
                            eachbatch + 1) + torch.mean(inputs, 0) / (eachbatch + 1)
                this_std = torch.load('statistics/this_std' + str(idx) + direction) * eachbatch / (
                            eachbatch + 1) + torch.std(inputs, 0) / (eachbatch + 1)
                torch.save(this_mean, 'statistics/this_mean' + str(idx) + direction)
                torch.save(this_std, 'statistics/this_std' + str(idx) + direction)
            else:
                this_mean = torch.load('statistics/this_mean' + str(idx) + direction)
                this_std = torch.load('statistics/this_std' + str(idx) + direction)

        outputs = (inputs - this_mean) * 1.0 / (this_std + 1e-8)
        return outputs

    ########### IMPORTANT ##################
    # We use unmodulated bits at encoder
    #######################################
    def forward(self,belief_threshold, eachbatch, bVec, fwd_noise_par, isTraining = 1):
        ###############################################################################################################################################################
        # combined_noise_par = fwd_noise_par + fb_noise_par # The total noise for parity bits
        bVec_md = 2*bVec-1
        belief = torch.full((self.args.batchSize, self.args.numb_block, self.args.block_class), fill_value=1 /self.args.block_class,requires_grad=False).to(self.args.device)
        es = []
        belief_all = []
        early_stop = 0
        for idx in range(self.truncated): # Go through parity bits
            ############# deal with beief ###################################################
            mask = (torch.max(belief, dim=2)[0]> belief_threshold) & torch.ones(args.batchSize, args.numb_block, dtype=torch.bool).to(args.device)
            early_stop = torch.sum(mask) - sum(es[:idx])
            es.append(early_stop.item())
            if mask.all():
                break
            if idx == 0: # phase 0
                src = torch.cat([bVec_md,torch.zeros(self.args.batchSize, self.args.numb_block,2*(self.truncated-1)).
                                to(self.args.device)],dim=2)
            else:
                src_new = torch.cat([bVec_md, parity_all,torch.zeros(self.args.batchSize, args.numb_block, self.truncated-(idx+1)).to(self.args.device),
                                     fwd_noise_par[:, :, :idx],
                                     torch.zeros(self.args.batchSize, args.numb_block, self.truncated - (idx + 1)).to(
                                         self.args.device)],dim=2)
                src = torch.where(mask.unsqueeze(2),src,src_new)
            ############# Generate the output ###################################################
            output = self.Tmodel(src, None,self.pe)
            parity = self.power_constraint(output, isTraining, eachbatch, idx)
            if self.args.reloc == 1:
                parity = self.total_power_reloc(parity,idx)
            if idx == 0:
                parity_all = parity
                received = torch.cat([parity + fwd_noise_par[:,:,0].unsqueeze(-1),torch.zeros(self.args.batchSize, self.args.numb_block,self.truncated-1).
                                to(self.args.device),belief], dim= 2)
            else:
                parity_all = torch.cat([parity_all, parity], dim=2)
                received_new = torch.cat([parity_all+ fwd_noise_par[:,:,:idx+1],torch.zeros(self.args.batchSize,self.args.numb_block,self.truncated-(1+idx)).
                                to(self.args.device),belief], dim = 2)
                received = torch.where(mask.unsqueeze(2),received,received_new)
            # received = parity + fwd_noise_par[:,:,idx].unsqueeze(-1)
            # received = torch.cat([received, belief],dim = -1)
            belief_new = self.Rmodel(received, None,self.pe)  # Decode the sequence
            belief = torch.where(mask.unsqueeze(2), belief, belief_new)
            if idx>6:
                belief_all.append(belief)
        if len(belief_all) == 0:
            belief_all.append(belief)
            #belief is a probability,shape(batchsize,seq_len,prob)->(4096,16,8).
            # if the prob arrive a threshold, then the whole 2nd dim will be masked
        # ------------------------------------------------------------ receiver
        #print(received.shape)
        return belief_all,idx,es

args = args_parser()
args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
args.d_model_trx = args.heads_trx * args.d_k_trx # total number of features

model = RFC(args).to(args.device)
checkpoint = torch.load(args.test_model)
# # ======================================================= load weights
model.load_state_dict(checkpoint)
if args.device == 'cuda':
    model = torch.nn.DataParallel(model)
    torch.backends.cudnn.benchmark = True
print(model)