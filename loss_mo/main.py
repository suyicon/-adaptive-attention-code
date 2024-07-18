import torch, pdb, os
import numpy as np
import torch.nn as nn
from utils import *
from nn_layers import *
from parameters import *
import numpy as np
import torch.optim as optim
from model import RFC

def compute_avgcodelength(logs):
    es_list = []
    for log in logs:
        es_list.append(log['early_stop'])
    avg_codelen = 0
    for idx in range(args.truncated):
        avg_codelen+= es_list[idx]*(idx+1)
    avg_codelen = (avg_codelen + args.truncated* ((args.batchSize*args.numb_block) - sum(es_list)))/(args.batchSize*args.numb_block)
    return avg_codelen
def ModelAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

def train_model(model, args):
    print(args)
    print("-->-->-->-->-->-->-->-->-->--> start training ...")
    model.train()
    map_vec = torch.tensor([1,2,4])# maping block of bits to class label
    ###################Setting optimizer############################
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

    # in each run, randomly sample a batch of data from the training dataset
    numBatch = 10000 * args.totalbatch + 1 + args.core # Total number of batches
    for eachbatch in range(args.start_step,numBatch):
        #################################### Generate noise sequence ##################################################
        bVec = torch.randint(0, 2, (args.batchSize, args.numb_block, args.block_size))

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
        ####################################Set forward noise and feedback noise#############################################
        std1 = 10 ** (-snr1 * 1.0 / 10 / 2) #forward snr
        std2 = 10 ** (-snr2 * 1.0 / 10 / 2) #feedback snr
        fwd_noise_par = torch.normal(0, std=std1,
                                     size=(args.batchSize, args.numb_block, args.truncated),
                                     requires_grad=False)
        fb_noise_par = torch.normal(0, std=std2,
                                     size=(args.batchSize, args.numb_block, args.truncated),
                                     requires_grad=False)
        if args.snr2 == 100:
            fb_noise_par = 0* fb_noise_par
        ################################## Simulate multicores by singlecore ###############################################
        if np.mod(eachbatch, args.core) == 0:
            w_locals = []
            w0 = model.state_dict()
            w0 = copy.deepcopy(w0)
        else:
            # Use the common model to have a large batch strategy
            model.load_state_dict(w0)
        
        ################################## Training ###############################################
        if args.multclass:
           bVec_mc = torch.matmul(bVec,map_vec)
           ys = bVec_mc.long().contiguous().view(-1)
        else:
        # expand the labels (bVec) in a batch to a vector, each word in preds should be a 0-1 distribution
           ys = bVec.long().contiguous().view(-1)
        # feed into model to get predictions
        train_log,preds,losses = model(belief_threshold,
                            eachbatch, 
                            bVec.to(args.device), 
                            fwd_noise_par.to(args.device),
                            ys,
                            isTraining=1)
        # Save the model
        w1 = model.state_dict()
        w_locals.append(copy.deepcopy(w1))
        ###################### untill core number of iterations are completed ####################
        if np.mod(eachbatch, args.core) != args.core - 1:
            continue
        else:
            ########### When core number of models are obtained #####################
            w2 = ModelAvg(w_locals)  # Average the models
            model.load_state_dict(copy.deepcopy(w2))
            ##################### change the learning rate ##########################
            if args.use_lr_schedule:
                args.scheduler.step()
        ################################ Observe test accuracy##############################
        with torch.no_grad():
            decodeds = preds.max(dim=1)[1]
            succRate = sum(decodeds == ys.to(args.device)) / len(ys)
            log = {"batch":eachbatch,
                   "snr1":args.snr1,
                   "snr2":args.snr2,
                   "lr":args.optimizer.state_dict()['param_groups'][0]['lr'],
                   "BER":1 - succRate.item(),
                   "num":sum(decodeds != ys.to(args.device)).item(),
                   "final_loss":train_log[-1]['loss'],
                   "losses":losses,
                   "train_log":train_log}
            print(log)
        #############################Save Model###########################
        if np.mod(eachbatch, args.core * 20000) == args.core - 1 and eachbatch >= 40000:
            if not os.path.exists('weights'):
                os.mkdir('weights')
            saveDir = 'weights/model_weights_{}_{}_'.format(snr1,snr2) + str(eachbatch)
            torch.save(model.state_dict(), saveDir)
        else:
            if not os.path.exists('weights'):
                os.mkdir('weights')
            torch.save(model.state_dict(), 'weights/latest')


def evaluate_model(model, args):
    checkpoint = torch.load(args.test_model)
    # # ======================================================= load weights
    model.load_state_dict(checkpoint)
    print(args)
    print("-->-->-->-->-->-->-->-->-->--> start testing ...")
    model.eval()
    map_vec = torch.tensor([1,2,4])
    args.numTestbatch = 100000000
    # failbits = torch.zeros(args.K).to(args.device)
    bitErrors = 0
    pktErrors = 0

    for eachbatch in range(args.numTestbatch):
        # generate b sequence and zero padding
        bVec = torch.randint(0, 2, (args.batchSize, args.numb_block, args.block_size))
        # generate n sequence
        std1 = 10 ** (-args.snr1 * 1.0 / 10 / 2)
        std2 = 10 ** (-args.snr2 * 1.0 / 10 / 2)
        fwd_noise_par = torch.normal(0, std=std1,
                                     size=(args.batchSize, args.numb_block, args.truncated),
                                     requires_grad=False)
        fb_noise_par = torch.normal(0, std=std2,
                                     size=(args.batchSize, args.numb_block, args.truncated),
                                     requires_grad=False)
        if args.snr2 == 100:
            fb_noise_par = 0* fb_noise_par
        if args.multclass:
            bVec_mc = torch.matmul(bVec,map_vec)
            ys = bVec_mc.long().contiguous().view(-1)
        else:
            ys = bVec.long().contiguous().view(-1)
        # feed into model to get predictions
        with torch.no_grad():
            test_log,preds = model(args.belief_threshold, eachbatch, bVec.to(args.device), fwd_noise_par.to(args.device), ys,isTraining=0)
            avg_codelen = compute_avgcodelength(test_log)

            preds1 =  preds.contiguous().view(-1, preds.size(-1))
            #print(preds1.shape)
            decodeds = preds1.max(dim=1)[0]
            decisions = decodeds != ys.to(args.device)
            bitErrors += decisions.sum()
            BER = bitErrors / (eachbatch + 1) / args.batchSize / args.numb_block
            pktErrors += decisions.view(args.batchSize, args.numb_block).sum(1).count_nonzero()
            PER = pktErrors / (eachbatch + 1) / args.batchSize
            log = {"batch":eachbatch,
                   "avg_codelen":avg_codelen,
                   "snr1":args.snr1,
                   "snr2":args.snr2,
                   "BER":BER.item(),
                   "bitErrors":bitErrors.item(),
                   "PER":PER.item(),
                   "num":sum(decodeds != ys.to(args.device)).item(),
                   "train_log":test_log}
            print(log)
    BER = bitErrors.cpu() / (args.numTestbatch * args.batchSize * args.K)
    PER = pktErrors.cpu() / (args.numTestbatch * args.batchSize)
    print(BER)
    print(PER)
    print("Final test BER = ", torch.mean(BER).item())
    print("Final test PER = ", torch.mean(PER).item())
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
        train_model(model, args)
    else:
        evaluate_model(model, args)