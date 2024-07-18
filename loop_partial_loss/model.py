import torch, time, pdb, os, random
import numpy as np
import torch.nn as nn
from utils import *
from nn_layers import *
from parameters import *
import numpy as np
import torch.optim as optim


class RFC(nn.Module):
    def __init__(self, args):
        super(RFC, self).__init__()
        self.args = args
        self.truncated = args.truncated
        self.pe = PositionalEncoder_fixed()
        ######### Initialize encoder and decoder ###############
        self.Tmodel = BERT(mod="trx", 
                           input_size=args.block_size+2*(args.truncated-1), 
                           block_size=args.block_size, 
                           d_model=args.d_model_trx, 
                           N=args.N_trx, 
                           heads=args.heads_trx, 
                           dropout=args.dropout, 
                           custom_attn=args.custom_attn,
                           multclass=args.multclass,
                           temp=args.temp)
        
        self.Rmodel = BERT(mod="rec", 
                           input_size=args.block_class+args.truncated,
                           block_size=args.block_size, 
                           d_model=args.d_model_trx, 
                           N=args.N_trx+1, 
                           heads=args.heads_trx, 
                           dropout=args.dropout, 
                           custom_attn=args.custom_attn,
                           multclass=args.multclass,
                           temp=args.temp)
        
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
    def forward_train(self, belief_threshold, eachbatch, bVec, fwd_noise_par, ys, optimizer):

        bVec_md = 2*bVec-1
        belief = torch.full((self.args.batchSize, 
                             self.args.numb_block, 
                             self.args.block_class), 
                             fill_value=1 /self.args.block_class,
                             requires_grad=False).to(self.args.device)
        mask = torch.zeros(self.args.batchSize, 
                           self.args.numb_block,dtype=torch.bool).to(self.args.device)
        train_log = []
        es=[]
        for idx in range(self.truncated): # Go through parity bits
            optimizer.zero_grad()
            ############# Generate the input features ###################################################
            if idx == 0: # phase 0
                src = torch.cat([bVec_md,torch.zeros(self.args.batchSize, 
                                                     self.args.numb_block,
                                                     2*(self.truncated-1)).to(self.args.device)],dim=2)
            else:
                src_new = torch.cat([bVec_md, 
                                     parity_all,
                                     torch.zeros(self.args.batchSize, self.args.numb_block, self.truncated-(idx+1)).to(self.args.device),
                                     fwd_noise_par[:, :, :idx],
                                     torch.zeros(self.args.batchSize,self.args.numb_block, self.truncated - (idx + 1)).to(
                                         self.args.device)],dim=2)
                src = torch.where(mask.unsqueeze(2),src,src_new)

            ############# Generate the parity ###################################################
            output = self.Tmodel(src, None,self.pe)
            parity = self.power_constraint(output,
                                           eachbatch=eachbatch,
                                           idx=idx,
                                           isTraining=1)
            if self.args.reloc == 1:
                parity = self.total_power_reloc(parity,idx)

            ############# Generate the received symbols ###################################################
            if idx == 0:
                parity_all = parity
                received = torch.cat([parity + fwd_noise_par[:,:,0].unsqueeze(-1),
                                      torch.zeros(self.args.batchSize, self.args.numb_block,self.truncated-1).
                                to(self.args.device),belief], dim= 2)
            else:
                parity_all = torch.cat([parity_all, parity], dim=2)
                received_new = torch.cat([parity_all+ fwd_noise_par[:,:,:idx+1],
                                          torch.zeros(self.args.batchSize,self.args.numb_block,self.truncated-(1+idx)).to(self.args.device),
                                          belief], dim = 2)
                received = torch.where(mask.unsqueeze(2),received,received_new)

            ############# Update the received beliefs ###################################################
            belief_new = self.Rmodel(received, None,self.pe)  # Decode the sequence
            belief = torch.where(mask.unsqueeze(2), belief, belief_new)
            if idx>5:
                ############# Update the decoding decision ###################################################
                mask = (torch.max(belief, dim=2)[0] > belief_threshold) & torch.ones(self.args.batchSize,
                                                                                     self.args.numb_block,
                                                                                     dtype=torch.bool).to(self.args.device)
                if mask.all():
                    break
                ############# Backwarding and update gradient ###################################################
                preds = torch.log(belief.contiguous().view(-1, belief.size(-1))) #=> (Batch*K) x 2
                loss = F.nll_loss(preds, ys.to(self.args.device))
                loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(self.parameters(), self.args.clip_th)
                optimizer.step()
                ############# logging early_stop ###################################################
                early_stop = torch.sum(mask) - sum(es[:idx])
                es.append(early_stop.item())
                train_log.append({"round":idx,"loss":loss.item(),"early_stop":early_stop.item()})

        return train_log,preds
    

    def forward_evaluate(self, belief_threshold, eachbatch, bVec, fwd_noise_par):

        bVec_md = 2*bVec-1
        belief = torch.full((self.args.batchSize, 
                             self.args.numb_block, 
                             self.args.block_class), 
                             fill_value=1 /self.args.block_class,
                             requires_grad=False).to(self.args.device)
        mask = torch.zeros(self.args.batchSize, self.args.numb_block,dtype=torch.bool).to(self.args.device)
        early_stop = 0
        test_log = {}
        es = []
        map_vec = torch.tensor([1, 2, 4]).to(self.args.device)

        for idx in range(self.truncated): # Go through parity bits
            ############# Generate the input features ###################################################
            if idx == 0: # phase 0
                src = torch.cat([bVec_md,torch.zeros(self.args.batchSize, 
                                                     self.args.numb_block,
                                                     2*(self.truncated-1)).to(self.args.device)],dim=2)
            else:
                src_new = torch.cat([bVec_md, 
                                     parity_all,
                                     torch.zeros(self.args.batchSize, self.args.numb_block, self.truncated-(idx+1)).to(self.args.device),
                                     fwd_noise_par[:, :, :idx],
                                     torch.zeros(self.args.batchSize,self.args.numb_block, self.truncated - (idx + 1)).to(
                                         self.args.device)],dim=2)
                src = torch.where(mask.unsqueeze(2),src,src_new)

            ############# Generate the parity ###################################################
            output = self.Tmodel(src, None,self.pe)
            parity = self.power_constraint(output, eachbatch, idx, isTraining=0)
            if self.args.reloc == 1:
                parity = self.total_power_reloc(parity,idx)

            ############# Generate the received symbols ###################################################
            if idx == 0:
                parity_all = parity
                received = torch.cat([parity + fwd_noise_par[:,:,0].unsqueeze(-1),
                                      torch.zeros(self.args.batchSize, self.args.numb_block,self.truncated-1).
                                to(self.args.device),belief], dim= 2)
            else:
                parity_all = torch.cat([parity_all, parity], dim=2)
                received_new = torch.cat([parity_all+ fwd_noise_par[:,:,:idx+1],
                                          torch.zeros(self.args.batchSize,self.args.numb_block,self.truncated-(1+idx)).to(self.args.device),
                                          belief], dim = 2)
                received = torch.where(mask.unsqueeze(2),received,received_new)

            ############# Update the received beliefs ###################################################
            belief_new = self.Rmodel(received, None,self.pe)  # Decode the sequence
            belief = torch.where(mask.unsqueeze(2), belief, belief_new)

            ############# Update the decoding decision ###################################################
            mask = (torch.max(belief, dim=2)[0] > belief_threshold) & torch.ones(self.args.batchSize, 
                                                                                 self.args.numb_block,
                                                                                 dtype=torch.bool).to(self.args.device)
            if mask.all():
                break

            ############# logging early_stop ###################################################
            
            #compute errors in one round
            mask_old = mask
            if idx > 0:
                pred = belief.max(dim=-1)[1]
                decode_map = (mask ^ mask_old)
                bVec_mc = torch.sum(torch.mul(bVec,map_vec),dim = -1)
                errors = ((torch.sum((decode_map) & (bVec_mc != pred)))).item()
                
            early_stop = torch.sum(mask) - sum(es[:idx])
            es.append(early_stop.item())
            test_log.update({"round":idx,"errors":errors,"early_stop":early_stop.item()})

        return test_log,belief

    def forward(self,belief_threshold, eachbatch, bVec, fwd_noise_par,ys,isTraining=1):
        if isTraining:
            optimizer=self.args.optimizer
            return self.forward_train(belief_threshold, eachbatch, bVec, fwd_noise_par, ys,optimizer)
        else:
            return self.forward_evaluate(belief_threshold, eachbatch, bVec, fwd_noise_par)
        


