import os
import time
import torch
import argparse
import datetime
from dataset import Dataset
from trainer import valid, train, test
from torch.utils.data import DataLoader
# from knockknock import email_sender

from tqdm import tqdm
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration,Adafactor
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP


from base_logger import logger

now_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
parser = argparse.ArgumentParser()

parser.add_argument('--data_rate' ,  type = float, default=0.01)
parser.add_argument('--batch_size' , type = int, default=4)
parser.add_argument('--test_batch_size' , type = int, default=16)
parser.add_argument('--port' , type = int,  default = 12355)
parser.add_argument('--max_epoch' ,  type = int, default=1)
parser.add_argument('--base_trained', type = str, default = "google/t5-small-ssm-nq", help =" pretrainned model from 🤗")
parser.add_argument('--pretrained_model' , type = str,  help = 'pretrainned model')
parser.add_argument('--debugging' , type = bool,  default = False, help = "Don't save file")
parser.add_argument('--dev_path' ,  type = str,  default = '../woz-data/MultiWOZ_2.1/dev_data.json')
parser.add_argument('--train_path' , type = str,  default = '../woz-data/MultiWOZ_2.1/train_data.json')
parser.add_argument('--test_path' , type = str,  default = '../woz-data/MultiWOZ_2.1/test_data.json')
parser.add_argument('-n', '--nodes', default=1,type=int, metavar='N')
parser.add_argument('-g', '--gpus', default=2, type=int,help='number of gpus per node')
parser.add_argument('-nr', '--nr', default=0, type=int,help='ranking within the nodes')
args = parser.parse_args()

def makedirs(path): 
   try: 
        os.makedirs(path) 
   except OSError: 
       if not os.path.isdir(path): 
           raise
       
def get_loader(dataset,batch_size):
    print('get_loader')
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    shuffle = False
    pin_memory = True
    print("after sampler")
    loader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=batch_size, pin_memory=pin_memory,
        num_workers=0, shuffle=shuffle, sampler=train_sampler,  collate_fn=dataset.collate_fn)
    
    print('after data loader')
    return loader       
       
def main_worker(gpu, args):
    makedirs("./data"); makedirs("./logs"); makedirs("./model");
    logger.info(f'{gpu} works!')
    batch_size = int(args.batch_size / args.gpus)
    
    torch.distributed.init_process_group(
        backend='nccl',
        init_method=f'tcp://127.0.0.1:{args.port}',
        world_size=args.gpus,
        rank=gpu)
    
    torch.cuda.set_device(gpu)
    train_dataset =Dataset(args.train_path, 'train', args.data_rate, args.tokenizer, debug=True)
    val_dataset =Dataset(args.train_path, 'val', args.data_rate, args.tokenizer, debug=True)
    
        
    model = T5ForConditionalGeneration.from_pretrained(args.base_trained, return_dict=True).to(gpu)
    model = DDP(model, device_ids=[gpu])
    
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
    map_location = {'cuda:%d' % 0: 'cuda:%d' % gpu}
    
    logger.info("Load model")
    if args.pretrained_model:
        logger.info(f"use trained model{args.pretrained_model}")
        model.load_state_dict(
            torch.load(args.pretrained_model, map_location=map_location))

    logger.info("Trainning start")
    for epoch in range(args.max_epoch):
        if gpu==0: logger.info(f"Epoch : {epoch}")
        train(gpu, model, train_loader, optimizer)
        loss = valid(gpu, model, dev_loader)
        logger.info("Epoch : %d,  Loss : %.04f" % (epoch, loss))

        if gpu == 0 and loss < min_loss:
            logger.info("New best")
            min_loss = loss
            best_performance['min_loss'] = min_loss.item()
            if not args.debugging:
                torch.save(model.state_dict(), f"model/woz{args.data_rate}.pt")
            logger.info("safely saved")
                
    if gpu==0:            
        logger.info(f"Best Score :  {best_performance}" )
    dist.barrier()
    
    
def evaluate():
    test_dataset =Dataset(args.test_path, 'test', args.data_rate, args.tokenizer, debug=True)
    
    loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=args.test_batch_size, pin_memory=True,
        num_workers=0, shuffle=False, collate_fn=test_dataset.collate_fn)
    model = T5ForConditionalGeneration.from_pretrained(args.base_trained, return_dict=True).to(0)
    
    logger.info("Load model")
    if args.pretrained_model:
        logger.info(f"use trained model{args.pretrained_model}")
        model.load_state_dict(torch.load(f"model/woz{args.data_rate}.pt"))
        
    joint_goal_acc, slot_acc, loss = test(args, model, loader)
    logger.info(f'JGA : {joint_goal_acc} Slot Acc : {slot_acc} Loss : {loss}')
    
    
    
def main():
    logger.info(args)
    makedirs("./data");  makedirs("./model");makedirs("./out");
    args.world_size = args.gpus * args.nodes 
    args.tokenizer = T5Tokenizer.from_pretrained(args.base_trained)
    mp.spawn(main_worker,
        nprocs=args.world_size,
        args=(args,),
        join=True)
    evaluate()

if __name__ =="__main__":
    logger.info(f"{'-' * 30}")
    logger.info("Start New Trainning")
    start = time.time()
    main()
    result_list = str(datetime.timedelta(seconds=time.time() - start)).split(".")
    logger.info(f"take time : {result_list[0]}")
    logger.info("End The Trainning")
    logger.info(f"{'-' * 30}")
    

