import json
from pathlib import Path
import torch, time, pdb, os, random
import numpy as np
import torch.nn as nn
from utils import *
from nn_layers import *
from parameters import *
import numpy as np
import torch.optim as optim



def ModelAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

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
        mask = torch.zeros(args.batchSize, args.numb_block,dtype=torch.bool).to(args.device)
        if isTraining == 0:
            map_vec = torch.tensor([1, 2, 4]).to(args.device)
        errors_list = []
        early_stop = 0
        for idx in range(self.truncated): # Go through parity bits
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
            if idx>args.desired_T-4 and idx<=args.desired_T:
                belief_all.append(belief)
            mask_old = mask
            mask = (torch.max(belief, dim=2)[0] > belief_threshold) & torch.ones(args.batchSize, args.numb_block,
                                                                                 dtype=torch.bool).to(args.device)
            if mask.all():
                break
            early_stop = torch.sum(mask) - sum(es[:idx])
            es.append(early_stop.item())
            if isTraining == 0:
                if idx > 0:
                    pred = belief.max(dim=-1)[1]
                    decode_map = (mask ^ mask_old)
                    bVec_mc = torch.sum(torch.mul(bVec,map_vec),dim = -1)
                    errors = ((torch.sum((decode_map) & (bVec_mc != pred)))).item()
                    errors_list.append(errors)
        if len(belief_all) == 0:
            belief_all.append(belief)
        return belief_all,idx,es,errors_list

def train_model(model, args):
    print("-->-->-->-->-->-->-->-->-->--> start training ...")
    model.train()
    # start = time.time()
    # epoch_loss_record = []
    # flag = 0
    map_vec = torch.tensor([1,2,4])# maping block of bits to class label

    # in each run, randomly sample a batch of data from the training dataset
    numBatch = 10000 * args.totalbatch + 1 + args.core # Total number of batches
    for eachbatch in range(args.start_step,numBatch):
        bVec = torch.randint(0, 2, (args.batchSize, args.numb_block, args.block_size))
        #################################### Generate noise sequence ##################################################
        ###############################################################################################################
        ###############################################################################################################
        ################################### Curriculum learning strategy ##############################################
        if eachbatch < args.core * 20000:
           snr1=3* (1-eachbatch/(args.core * 20000))+ (eachbatch/(args.core * 20000)) * args.snr1
           snr2 = 100
           belief_threshold = 0.999+0.00099*(eachbatch/(args.core * 20000))
        elif eachbatch < args.core * 40000:
           snr2= 100 * (1-(eachbatch-args.core * 20000)/(args.core * 20000))+ ((eachbatch-args.core * 20000)/(args.core * 20000)) * args.snr2
           snr1=args.snr1
           belief_threshold = 0.99999+(args.belief_threshold-0.99999)*((eachbatch-args.core * 20000)/(args.core * 20000))
        else:
           belief_threshold = args.belief_threshold
           snr2=args.snr2
           snr1=args.snr1
        ################################################################################################################
        std1 = 10 ** (-snr1 * 1.0 / 10 / 2) #forward snr
        std2 = 10 ** (-snr2 * 1.0 / 10 / 2) #feedback snr
        # Noise values for the parity bits
        # fwd_noise_par = torch.normal(0, std=std1, size=(args.batchSize, args.numb_block, args.parity_pb+args.block_size), requires_grad=False)
        # fb_noise_par = torch.normal(0, std=std2, size=(args.batchSize, args.numb_block, args.parity_pb+args.block_size), requires_grad=False)
        # if args.snr2 == 100:
        #     fb_noise_par = 0* fb_noise_par
        fwd_noise_par = torch.normal(0, std=std1,
                                     size=(args.batchSize, args.numb_block, args.truncated),
                                     requires_grad=False)
        fb_noise_par = torch.normal(0, std=std2,
                                     size=(args.batchSize, args.numb_block, args.truncated),
                                     requires_grad=False)
        if args.snr2 == 100:
            fb_noise_par = 0* fb_noise_par
        if np.mod(eachbatch, args.core) == 0:
            w_locals = []
            w0 = model.state_dict()
            w0 = copy.deepcopy(w0)
        else:
            # Use the common model to have a large batch strategy
            model.load_state_dict(w0)

        # feed into model to get predictions
        preds,turn,early_stop,_ = model(belief_threshold,eachbatch, bVec.to(args.device), fwd_noise_par.to(args.device), isTraining=1)
        args.optimizer.zero_grad()
        if args.multclass:
           bVec_mc = torch.matmul(bVec,map_vec)
           ys = bVec_mc.long().contiguous().view(-1)
        else:
        # expand the labels (bVec) in a batch to a vector, each word in preds should be a 0-1 distribution
           ys = bVec.long().contiguous().view(-1)
        loss = torch.tensor(0.).to(args.device)
        if len(preds) != 0:
            try:
                for idx in range(len(preds)):
                    preds[idx] = preds[idx].contiguous().view(-1, preds[idx].size(-1)) #=> (Batch*K) x 2
                    preds[idx] = torch.log(preds[idx])
                    loss += 10**idx*F.nll_loss(preds[idx], ys.to(args.device))########################## This should be binary cross-entropy loss
                    entropy = F.nll_loss(preds[idx], ys.to(args.device))
                loss.backward()
            except Exception as e:
                print(f"idx is {idx},len of preds is {len(preds)}, turn is {turn},error:{e}")
        ####################### Gradient Clipping optional ###########################
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_th)
        ##############################################################################
        args.optimizer.step()
        # Save the model
        w1 = model.state_dict()
        w_locals.append(copy.deepcopy(w1))
        ###################### untill core number of iterations are completed ####################
        if np.mod(eachbatch, args.core) != args.core - 1:
            continue
        else:
            ########### When core number of models are obtained #####################
            w2 = ModelAvg(w_locals) # Average the models
            model.load_state_dict(copy.deepcopy(w2))
            ##################### change the learning rate ##########################
            if args.use_lr_schedule:
                args.scheduler.step()
        ################################ Observe test accuracy##############################
        with torch.no_grad():
            probs, decodeds = preds[-1].max(dim=1)
            succRate = sum(decodeds == ys.to(args.device)) / len(ys)
            print('bert-code','Idx,round,early_stop,snr1,snr2,lr,BS,loss,BER,num,entropy=', (
            eachbatch,turn,early_stop,args.snr1,args.snr2,args.optimizer.state_dict()['param_groups'][0]['lr'], args.batchSize, loss.item(), 1 - succRate.item(),
            sum(decodeds != ys.to(args.device)).item(),entropy.item()))
            
            log_stats = {"epoch":eachbatch,"round":turn,"early_stop":early_stop,"snr1":args.snr1,"snr2":args.snr2,"lr":args.optimizer.state_dict()['param_groups'][0]['lr'],
                         "batchsize": args.batchSize,"loss":loss.item(),"BER": sum(decodeds != ys.to(args.device)).item(),"last_entropy":entropy.item()}
            
            output_dir = Path(args.output_dir)
            if args.output_dir:
                with (output_dir / "training_log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")
        ####################################################################################
        if np.mod(eachbatch, args.core * 5000) == args.core - 1 and eachbatch >= 5000:
            if not os.path.exists('weights'):
                os.mkdir('weights')
            saveDir = 'weights/model_weights_{}_{}_'.format(snr1,snr2) + str(eachbatch)
            torch.save(model.state_dict(), saveDir)
        else:
            if not os.path.exists('weights'):
                os.mkdir('weights')
            torch.save(model.state_dict(), 'weights/latest')


def EvaluateNets(model, args):
    checkpoint = torch.load(args.test_model)
    # # ======================================================= load weights
    model.load_state_dict(checkpoint)
    print(model)
    model.eval()
    map_vec = torch.tensor([1,2,4])

    args.numTestbatch = 100000000

    # failbits = torch.zeros(args.K).to(args.device)
    bitErrors = 0
    pktErrors = 0
    decode_list =[0]*args.truncated
    total_errors_list = [0]*args.truncated
    per_list = [0]*args.truncated
    for eachbatch in range(args.numTestbatch):
        # generate b sequence and zero padding
        bVec = torch.randint(0, 2, (args.batchSize, args.numb_block, args.block_size))
        # generate n sequence
        std1 = 10 ** (-args.snr1 * 1.0 / 10 / 2)
        std2 = 10 ** (-args.snr2 * 1.0 / 10 / 2)
        # fwd_noise_par = torch.normal(0, std=std1, size=(args.batchSize, args.numb_block, args.parity_pb+args.block_size), requires_grad=False)
        # fb_noise_par = torch.normal(0, std=std2, size=(args.batchSize, args.numb_block, args.parity_pb+args.block_size), requires_grad=False)
        # if args.snr2 == 100:
        #     fb_noise_par = 0* fb_noise_par
        fwd_noise_par = torch.normal(0, std=std1,
                                     size=(args.batchSize, args.numb_block, args.truncated),
                                     requires_grad=False)
        fb_noise_par = torch.normal(0, std=std2,
                                     size=(args.batchSize, args.numb_block, args.truncated),
                                     requires_grad=False)
        if args.snr2 == 100:
            fb_noise_par = 0* fb_noise_par

        # feed into model to get predictions
        with torch.no_grad():
            preds,turn,early_stop,errors_list= model(args.belief_threshold, eachbatch, bVec.to(args.device), fwd_noise_par.to(args.device), isTraining=0)
            avg_codelen = 0
            for idx in range(args.truncated):
                avg_codelen+= early_stop[idx]*(idx+1)
            # for idx in range(args.truncated):
            #     avg_codelen+= early_stop[idx]*idx
            #
            avg_codelen = (avg_codelen + args.truncated* ((args.batchSize*args.numb_block) - sum(early_stop)))/(args.batchSize*args.numb_block)

            if args.multclass:
                bVec_mc = torch.matmul(bVec,map_vec)
                ys = bVec_mc.long().contiguous().view(-1)
            else:
                ys = bVec.long().contiguous().view(-1)
            preds1 =  preds[-1].contiguous().view(-1, preds[-1].size(-1))
            #print(preds1.shape)
            probs, decodeds = preds1.max(dim=1)
            decisions = decodeds != ys.to(args.device)
            bitErrors += decisions.sum()
            BER = bitErrors / (eachbatch + 1) / args.batchSize / args.numb_block
            pktErrors += decisions.view(args.batchSize, args.numb_block).sum(1).count_nonzero()
            PER = pktErrors / (eachbatch + 1) / args.batchSize
            decode_list = [d+e for (d,e) in zip(decode_list,early_stop)]
            total_errors_list = [e+t for (e,t) in zip(errors_list,total_errors_list)]
            for i, (a,b) in enumerate(zip(total_errors_list,decode_list)):
                try:
                    per_list[i] = a/b
                except ZeroDivisionError:
                    per_list[i] = None
            print('num, avg_codelen,BER, errors, PER, errors,per_list,= ', eachbatch,avg_codelen, BER.item(), bitErrors.item(),
                  PER.item(), pktErrors.item(),per_list)


    BER = bitErrors.cpu() / (args.numTestbatch * args.batchSize * args.K)
    PER = pktErrors.cpu() / (args.numTestbatch * args.batchSize)
    print(BER)
    print("Final test BER = ", torch.mean(BER).item())
    pdb.set_trace()

if __name__ == '__main__':
    # ======================================================= parse args
    args = args_parser()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ########### path for saving model checkpoints ################################
    args.saveDir = 'weights/model_weights'  # path to be saved to
    ################## Model size part ###########################################
    args.d_model_trx = args.heads_trx * args.d_k_trx # total number of features
    ##############################################################################
    args.total_iter = 10000 * args.totalbatch + 1 + args.core
    # ======================================================= Initialize the model
    model = RFC(args).to(args.device)
    if args.device == 'cuda':
        model = torch.nn.DataParallel(model)
        torch.backends.cudnn.benchmark = True

    # ======================================================= run
    if args.train == 1:
        if args.opt_method == 'adamW':
            args.optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.wd, amsgrad=False)
        elif args.opt_method == 'lamb':
            args.optimizer = optim.Lamb(model.parameters(),lr= 1e-2, betas=(0.9, 0.999), eps=1e-8, weight_decay=args.wd)
        else:
            args.optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)
        if args.use_lr_schedule:
            lambda1 = lambda epoch: (1-epoch*args.core/(args.total_iter-args.start_step))
            args.scheduler = torch.optim.lr_scheduler.LambdaLR(args.optimizer, lr_lambda=lambda1)
            ######################## huggingface library ####################################################
            #args.scheduler = get_polynomial_decay_schedule_with_warmup(optimizer=args.optimizer, warmup_steps=1000, num_training_steps=args.total_iter, power=0.5)
        if args.start == 1:
            checkpoint = torch.load(args.start_model)
            model.load_state_dict(checkpoint)
            print("================================ Successfully load the pretrained data!")
        if not os.path.exists(args.output_dir):
            Path(args.output_dir).mkdir(parents=True, exist_ok=True)

        train_model(model, args)
    else:
        EvaluateNets(model, args)
