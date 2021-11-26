import torch
import pdb 
import json
from base_logger import logger
import ontology
from utils import*
from collections import defaultdict

def train(gpu, model, train_loader, optimizer):
    model.train()
    if gpu==0: logger.info("Train start")
    for iter, batch in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(input_ids=batch['input']['input_ids'], labels=batch['target']['input_ids'])
        loss =outputs.loss
        loss.backward()
        optimizer.step()
    
        if (iter + 1) % 10 == 0 and gpu==0:
            logger.info('gpu {} step : {}/{} Loss: {:.4f}'.format(
                gpu,
                iter, 
                str(len(train_loader)),
                loss.detach())
            )
        

def valid(gpu, model, dev_loader):
    model.eval()
    loss_sum = 0
    logger.info("Validation start")
    with torch.no_grad():
        print("no_grad")
        for iter,batch in enumerate(dev_loader):
            output = model(input_ids=batch['input']['input_ids'], labels=batch['target']['input_ids'])
            loss_sum += output.loss.detach()
            if (iter + 1) % 10 == 0 and gpu == 0:
                logger.info('step : {}/{} Loss: {:.4f}'.format(
                iter, 
                str(len(dev_loader)),
                output.loss.detach()
                ))
           
    return  loss_sum/iter



def test(args, model, test_loader):
    belief_state= defaultdict(lambda : defaultdict(dict))# dial_id, # turn_id # schema
    model.eval()
    loss_sum = 0
    logger.info("Test start")
    with torch.no_grad():
        for iter,batch in enumerate(test_loader):
            outputs = model(input_ids=batch['input']['input_ids'].to('cuda:0'), labels=batch['target']['input_ids'].to('cuda:0'))
            outputs_text = model.generate(input_ids=batch['input']['input_ids'].to('cuda:0'))
            outputs_text = [args.tokenizer.decode(o).replace('</s>','').replace('<pad>','').strip() for o in outputs_text]
            
            
            for idx in range(args.test_batch_size):
                if outputs_text[idx] == ontology.QA['NOT_MENTIONED'] : continue
                dial_id = batch['dial_id'][idx]
                turn_id = batch['turn_id'][idx]
                schema = batch['schema'][idx]
                belief_state[dial_id][turn_id][schema] = outputs_text[idx]

            if (iter + 1) % 10 == 0:
                logger.info('step : {}/{}'.format(
                iter+1, 
                str(len(test_loader)),
                ))
                break
                

    test_file = json.load(open(args.test_path , "r"))
    joint_goal_acc, slot_acc = evaluate_metrics(belief_state, test_file , ontology.QA['all-domain'])
    
    loss_sum += outputs.loss.detach()

    return  joint_goal_acc, slot_acc, loss_sum/iter
        
        