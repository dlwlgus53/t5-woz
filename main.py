import utils
import json
import time
import torch
import logging
import argparse
import datetime
from dataset import Dataset
from loopdataset import LoopDataset
import init

from collections import OrderedDict
from trainer import train, test, tag
from torch.utils.data import DataLoader

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import T5Tokenizer, T5ForConditionalGeneration,Adafactor

parser = argparse.ArgumentParser()

parser.add_argument('--data_rate' ,  type = float, default=0.01)
parser.add_argument('--do_train' ,  type = int, default=1)
parser.add_argument('--do_loop' ,  type = int, default=0)
parser.add_argument('--do_short' ,  type = int, default=1)
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
parser.add_argument('--untagged_path' , type = str,  default = '../woz-data/MultiWOZ_2.1/untagged.json')
parser.add_argument('--test_path' , type = str,  default = '../woz-data/MultiWOZ_2.1/test_data.json')
parser.add_argument('--detail_log' , type = int,  default = 0)
parser.add_argument('--save_prefix', type = str, help = 'prefix for all savings', default = '')
parser.add_argument('-n', '--nodes', default=1,type=int, metavar='N')
parser.add_argument('-g', '--gpus', default=4, type=int,help='number of gpus per node')
parser.add_argument('-nr', '--nr', default=0, type=int,help='ranking within the nodes')
parser.add_argument('--never_split_file',  default='./asset/never_split.txt', type=str,help='number of gpus per node')
parser.add_argument('--aux',  default=1, type=int, help='number of gpus per node')
parser.add_argument('--zeroshot_domain', type=str, choices=["restaurant", "hotel", "attraction", "train", "taxi"],help='restaurant|hotel|attraction|train|taxi')
parser.add_argument('--temp_folder', type=str, default = './looptemp')
parser.add_argument('--self_step', type=int, default = 500)

args = parser.parse_args()
init.init_experiment(args)
logger = logging.getLogger("my")
       
def get_loader(dataset,batch_size):
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    shuffle = False
    pin_memory = True
    loader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=batch_size, pin_memory=pin_memory,
        num_workers=0, shuffle=shuffle, sampler=train_sampler,  collate_fn=dataset.collate_fn)
    return loader       
       
def main_worker(gpu, args):
    logger.info(f'In main, {gpu} works!')
    batch_size = int(args.batch_size / args.gpus)
    torch.distributed.init_process_group(
        backend='nccl',
        init_method=f'tcp://127.0.0.1:{args.port}',
        world_size=args.gpus,
        rank=gpu)
    torch.cuda.set_device(gpu)

    model = T5ForConditionalGeneration.from_pretrained(args.base_trained, return_dict=True).to(gpu)
    model.resize_token_embeddings(len(args.tokenizer))
    model = DDP(model, device_ids=[gpu])
    optimizer = Adafactor(model.parameters(),lr=1e-3,
                    eps=(1e-30, 1e-3),
                    clip_threshold=1.0,
                    decay_rate=-0.8,
                    beta1=None,
                    weight_decay=0.0,
                    relative_step=False,
                    scale_parameter=False,
                    warmup_init=False)
    logger.info("Trainning start")
    
    for epoch in range(args.max_epoch):
        train_dataset =Dataset(args, args.train_path, 'train', epoch) # None or not
        train_loader = get_loader(train_dataset, batch_size)
        if gpu==0: logger.info(f"Epoch : {epoch}")
        train(args, gpu, model, train_loader, optimizer, train_dataset)

    if gpu == 0:
        if not args.debugging:
            torch.save(model.state_dict(), f"model/woz{args.save_prefix}{args.data_rate}.pt")
            logger.info("safely saved")
    dist.barrier()
    
           
def loop_worker(gpu, args):
        
    logger.info(f'In loop, {gpu} works!')
    batch_size = int(args.batch_size / args.gpus)
    self_step = int(args.self_step/args.gpus)
    test_batch_size = int(args.test_batch_size / args.gpus)
    
    torch.distributed.init_process_group(
        backend='nccl',
        init_method=f'tcp://127.0.0.1:{args.port}',
        world_size=args.gpus,
        rank=gpu)
    torch.cuda.set_device(gpu)
    
    model = T5ForConditionalGeneration.from_pretrained(args.base_trained, return_dict=True).to(gpu)
    model = load_trained(args, model)
    model.resize_token_embeddings(len(args.tokenizer))
    model = DDP(model, device_ids=[gpu])
        
    optimizer = Adafactor(model.parameters(),lr=1e-3,
                    eps=(1e-30, 1e-3),
                    clip_threshold=1.0,
                    decay_rate=-0.8,
                    beta1=None,
                    weight_decay=0.0,
                    relative_step=False,
                    scale_parameter=False,
                    warmup_init=False)

    logger.info("Tag start")
    tag_dataset = LoopDataset(args, 'tag')
    tag_loader = get_loader(tag_dataset, test_batch_size)
    high_confidence = tag(args, model, gpu, tag_loader,self_step)
    with open(f"{args.temp_folder}/confidence/c_{gpu}.txt", 'w') as f:
        for item in high_confidence:
            f.write(','.join(item)+'\n')
            
    with open(f"{args.temp_folder}/worked_list/w_{gpu}.txt", 'a') as f:
        for item in high_confidence:
            f.write(','.join(item)+'\n')
    dist.barrier()

    ctrain_dataset =LoopDataset(args,'train') # None or not
    ctrain_loader = get_loader(ctrain_dataset, batch_size)
    train(args, gpu, model, ctrain_loader, optimizer, ctrain_dataset, save_belief = False)

    if gpu == 0:
        if not args.debugging:
            torch.save(model.state_dict(), f"model/woz{args.save_prefix}{args.data_rate}.pt")
            logger.info(f"safely saved in model/woz{args.save_prefix}{args.data_rate}.pt")


    dist.barrier()



def load_trained(args,model):
    logger.info(f"User pretrained model{args.pretrained_model}")
    
    state_dict = torch.load(args.pretrained_model)
    new_state_dict = OrderedDict()
    
    for k, v in state_dict.items():
        name = k[7:] # remove 'module.' of DataParallel/DistributedDataParallel
        new_state_dict[name] = v
    model.resize_token_embeddings(len(args.tokenizer))
    model.load_state_dict(new_state_dict)
    return model
    
    
def evaluate(): # 여기는 건들 것 없지
    test_dataset =Dataset(args, args.test_path, 'test')
    loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=args.test_batch_size, pin_memory=True,
        num_workers=0, shuffle=False, collate_fn=test_dataset.collate_fn)

    model = T5ForConditionalGeneration.from_pretrained(args.base_trained, return_dict=True).to('cuda:0')
    
    if args.pretrained_model:
        logger.info(f"User pretrained model{args.pretrained_model}")
        model = load_trained(args,model)
    
    joint_goal_acc, slot_acc, domain_acc, schema_acc, detail_wrong, loss = test(args, model, loader, test_dataset)
    print(f'JGA : {joint_goal_acc}')
    
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
    
    
    # evaluate()
    
    if args.do_loop:
        while True:
            try:
                mp.spawn(loop_worker,
                    nprocs=args.world_size,
                    args=(args,),
                    join=True)
            except Exception as e:    # 모든 예외의 에러 메시지를 출력할 때는 Exception을 사용
                logger.error(e)
                print(e)
            torch.cuda.empty_cache()
            evaluate()    
            with open(f"{args.temp_folder}/confidence/c_{gpu}.txt", 'r') as f:
                if len(f.read().splitlines()) < int(args.self_step/args.gpus) : 
                    break
            torch.cuda.empty_cache()
            
if __name__ =="__main__":
    utils.makedirs("./data");  utils.makedirs("./model"); utils.makedirs("./out");
    utils.makedirs("./looptemp/confidence"); utils.makedirs("./logs/csvs"); utils.makedirs("./logs/jsons");
    utils.makedirs("./looptemp/worked_list"); 
    
    for gpu in range(args.gpus):
        with open(f"{args.temp_folder}/confidence/c_{gpu}.txt", 'w') as f:
            f.write('')
                
        with open(f"{args.temp_folder}/worked_list/w_{gpu}.txt", 'w') as f:
            f.write('')
            
    logger.info(f"{'-' * 30}")
    logger.info("Start New Trainning")
    start = time.time()
    main()
    result_list = str(datetime.timedelta(seconds=time.time() - start)).split(".")
    logger.info(f"take time : {result_list[0]}")
    logger.info("End The Trainning")
    logger.info(f"{'-' * 30}")
    

