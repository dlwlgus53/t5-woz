import utils
import time
import torch
import logging
import argparse
import datetime
from dataset import Dataset
import init

from collections import OrderedDict
from trainer import valid, train, test
from torch.utils.data import DataLoader

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import T5Tokenizer, T5ForConditionalGeneration,Adafactor

parser = argparse.ArgumentParser()

parser.add_argument('--data_rate' ,  type = float, default=0.01)
parser.add_argument('--alpha' ,  type = float, default=1.0)
parser.add_argument('--do_train' ,  type = int, default=1)
parser.add_argument('--do_short' ,  type = int, default=0)
parser.add_argument('--do_test' ,  type = int, default=1)
parser.add_argument('--max_length' ,  type = int, default=128)
parser.add_argument('--dst_student_rate' ,  type = float, default=1.0)
parser.add_argument('--seed' ,  type = int, default=1)
parser.add_argument('--batch_size' , type = int, default=4)
parser.add_argument('--test_batch_size' , type = int, default=16)
parser.add_argument('--port' , type = int,  default = 12355)
parser.add_argument('--max_epoch' ,  type = int, default=1)
parser.add_argument('--base_trained', type = str, default = "google/t5-large-ssm-nq", help =" pretrainned model from 🤗")
parser.add_argument('--pretrained_model' , type = str,  help = 'pretrainned model')
parser.add_argument('--debugging' , type = bool,  default = False, help = "Don't save file")
parser.add_argument('--dev_path' ,  type = str,  default = '../woz-data/MultiWOZ_2.1/dev_data.json')
parser.add_argument('--train_path' , type = str,  default = '../woz-data/MultiWOZ_2.1/train_data.json')
parser.add_argument('--test_path' , type = str,  default = '../woz-data/MultiWOZ_2.1/test_data.json')
parser.add_argument('--detail_log' , type = int,  default = 0)
parser.add_argument('--save_prefix', type = str, help = 'prefix for all savings', default = '')
parser.add_argument('-n', '--nodes', default=1,type=int, metavar='N')
parser.add_argument('-g', '--gpus', default=4, type=int,help='number of gpus per node')
parser.add_argument('-nr', '--nr', default=0, type=int,help='ranking within the nodes')
parser.add_argument('--never_split_file',  default='./asset/never_split.txt', type=str,help='number of gpus per node')
parser.add_argument('--aux',  default=1, type=int, help='number of gpus per node')
parser.add_argument('--zeroshot_domain', type=str, choices=["restaurant", "hotel", "attraction", "train", "taxi"],help='restaurant|hotel|attraction|train|taxi')
parser.add_argument('--train_continue', type=int, default = 0)

args = parser.parse_args()
init.init_experiment(args)
logger = logging.getLogger("my")
       
def load_trained(args,model, optimizer = None):
    logger.info(f"User pretrained model{args.pretrained_model}")
    state_dict = torch.load(args.pretrained_model)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove 'module.' of DataParallel/DistributedDataParallel
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    if optimizer:
        opt_path = "./model/optimizer/" + args.pretrained_model[7:] #todo
        optimizer.load_state_dict(torch.load(opt_path))
    print("load safely")

def get_loader(dataset,batch_size):
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    shuffle = False
    pin_memory = True
    loader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=batch_size, pin_memory=pin_memory,
        num_workers=0, shuffle=shuffle, sampler=train_sampler,  collate_fn=dataset.collate_fn)
    return loader       
       
def main_worker(gpu, args):
    logger.info(f'{gpu} works!')
    batch_size = int(args.batch_size / args.gpus)
    
    torch.distributed.init_process_group(
        backend='nccl',
        init_method=f'tcp://127.0.0.1:{args.port}',
        world_size=args.gpus,
        rank=gpu)
    
    torch.cuda.set_device(gpu)

        
    model = T5ForConditionalGeneration.from_pretrained(args.base_trained, return_dict=True).to(gpu)
    
    if args.pretrained_model and args.train_continue:
        logger.info(f"Use pretrained model{args.pretrained_model} in train")
        load_trained(args,model)
    else:        
        pass        
    model = DDP(model, device_ids=[gpu])
    
    train_dataset =Dataset(args, args.train_path, 'train')
    val_dataset =Dataset(args, args.dev_path, 'val')
    
    train_loader = get_loader(train_dataset, batch_size)
    dev_loader = get_loader(val_dataset, batch_size)
    
    optimizer = Adafactor(model.parameters(),lr=1e-3,
                    eps=(1e-30, 1e-3),
                    clip_threshold=1.0,
                    decay_rate=-0.8,
                    beta1=None,
                    weight_decay=0.0,
                    relative_step=False,
                    scale_parameter=False,
                    warmup_init=False)    


    min_loss = float('inf')
    best_performance = {}
    logger.info("Trainning start")
    for epoch in range(args.max_epoch):
        if gpu==0: logger.info(f"Epoch : {epoch}")
        train(args, gpu, model, train_loader, optimizer, train_dataset)
        loss = valid(args, gpu, model, dev_loader, args.data_rate, val_dataset)
        logger.info("Epoch : %d,  Loss : %.04f" % (epoch, loss))
        dist.all_reduce(loss); loss /= dist.get_world_size()
        if gpu == 0 and loss < min_loss:
            logger.info("Epoch : %d,  Reduced Loss : %.04f" % (epoch, loss))
            logger.info("New best")
            min_loss = loss
            best_performance['min_loss'] = min_loss.item()
            if not args.debugging:
                torch.save(model.state_dict(), f"model/woz{args.save_prefix}{args.data_rate}.pt")
                torch.save(optimizer.state_dict(), f"model/optimizer/woz{args.save_prefix}{args.data_rate}.pt")
                logger.info(f"safely saved in model/woz{args.save_prefix}{args.data_rate}.pt")
                
    if gpu==0:            
        logger.info(f"Best Score :  {best_performance}" )
    dist.barrier()
    
    
def evaluate():
    test_dataset =Dataset(args, args.test_path, 'test')
    loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=args.test_batch_size, pin_memory=True,
        num_workers=0, shuffle=False, collate_fn=test_dataset.collate_fn)
    model = T5ForConditionalGeneration.from_pretrained(args.base_trained, return_dict=True).to('cuda:0')
    
    
    if args.pretrained_model:
        logger.info(f"User pretrained model{args.pretrained_model} in eval")
        load_trained(args,model)
        
    joint_goal_acc, slot_acc, domain_acc, schema_acc, detail_wrong, loss = test(args, model, loader, test_dataset)
    
    logger.info(f'JGA : {joint_goal_acc} Slot Acc : {slot_acc} Loss : {loss}')
    logger.info(f'domain_acc : {domain_acc}')
    logger.info(f'schema_acc : {schema_acc}')
    
    schema_acc['JGA'] = joint_goal_acc
    schema_acc['schema_acc'] = slot_acc
    schema_acc.update(domain_acc)
    schema_acc['loss'] = loss
    
    
    utils.dict_to_csv(schema_acc, f'{args.save_prefix}{args.data_rate}.csv')
    
    if args.detail_log:
        utils.dict_to_json(detail_wrong, f'{args.save_prefix}{args.data_rate}.json')
    
    
    
def main():
    logger.info(args)
    
    with open(args.never_split_file, "r") as f:
        never_split = f.read().splitlines() 
    special_tokens_dict = {'additional_special_tokens': never_split}
    print(special_tokens_dict)
    
    args.world_size = args.gpus * args.nodes 
    args.tokenizer = T5Tokenizer.from_pretrained(args.base_trained)
    args.tokenizer.add_special_tokens(special_tokens_dict)
    if args.do_train:
        try:
            mp.spawn(main_worker,
                nprocs=args.world_size,
                args=(args,),
                join=True)
        except Exception as e:    # 모든 예외의 에러 메시지를 출력할 때는 Exception을 사용
            logger.error(e)
            print(e)
        
    evaluate()

if __name__ =="__main__":
    utils.makedirs("./data"); utils.makedirs("./logs"); utils.makedirs("./model"); utils.makedirs("./out")
    utils.makedirs("./model/optimizer"); utils.makedirs("./logs/csvs"); utils.makedirs("./logs/jsons"); 
    logger.info(f"{'-' * 30}")
    logger.info("Start New Trainning")
    start = time.time()
    main()
    result_list = str(datetime.timedelta(seconds=time.time() - start)).split(".")
    logger.info(f"take time : {result_list[0]}")
    logger.info("End The Trainning")
    logger.info(f"{'-' * 30}")
    

