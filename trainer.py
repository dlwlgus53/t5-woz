import torch
import pdb 
import json
import logging
import ontology
from utils import*
from collections import defaultdict
from queue import PriorityQueue
from utils import save_pickle

logger = logging.getLogger("my")


def train(args, gpu, model, train_loader, optimizer, train_dataset, save_belief = True):
    model.train()
    if gpu==0: logger.info("Train start")
    for iter, batch in enumerate(train_loader):
        optimizer.zero_grad()
        input_ids = batch['input']['input_ids'].to(f'cuda:{gpu}')
        labels = batch['target']['input_ids'].to(f'cuda:{gpu}')
        
        outputs = model(input_ids=input_ids, labels=labels)
        outputs_text = model.module.generate(input_ids=input_ids)
        outputs_text = [args.tokenizer.decode(o).replace('</s>','').replace('<pad>','').strip() for o in outputs_text]
        
        for idx in range(len(outputs_text)):
            if outputs_text[idx] == ontology.QA['NOT_MENTIONED'] : continue
            dial_id = batch['dial_id'][idx]
            turn_id = batch['turn_id'][idx]
            schema = batch['schema'][idx]
            if save_belief:
                train_dataset.belief_state[dial_id][turn_id][schema] = outputs_text[idx]
            
        loss =outputs.loss
        loss.backward()
        optimizer.step()
    
        if (iter + 1) % 50 == 0 and gpu==0:
            logger.info('gpu {} step : {}/{} Loss: {:.4f}'.format(
                gpu,
                iter, 
                str(len(train_loader)),
                loss.detach())
            )


def tag(args, model, gpu, tag_loader,self_step):
    confidence_list = []
    model.eval()
    que = PriorityQueue()
    with torch.no_grad():
        for iter,batch in enumerate(tag_loader):
            outputs = model(input_ids=batch['input']['input_ids'].to(f'cuda:{gpu}'), labels=batch['target']['input_ids'].to(f'cuda:{gpu}'))
            outputs_text = model.generate(input_ids=batch['input']['input_ids'].to('cuda'))
            outputs_text = [args.tokenizer.decode(o).replace('</s>','').replace('<pad>','').strip() for o in outputs_text]

            logits = outputs.logits.to('cpu')
            
            for idx, logit in enumerate(logits):
                dial_id = batch['dial_id'][idx]
                turn_id = batch['turn_id'][idx]
                schema = batch['schema'][idx]
                output_text = outputs_text[idx]
                ans_confidence =0 
                id_ans_list = [dial_id, str(turn_id), schema, output_text]
                for word_idx, word in enumerate(logit):
                    id = int(torch.argmax(word))
                    value = float(torch.max(word))
                    ans_confidence += value
                    if id == args.tokenizer.eos_token_id:
                        ans_confidence = -(ans_confidence/(word_idx+1)) # for priority
                        break
                que.put((ans_confidence, id_ans_list))
                
            if (iter + 1) % 50 == 0:
                logger.info('step : {}/{}'.format(
                iter+1, 
                str(len(tag_loader)),
                ))
        
    for _ in range(self_step):
        confidence_list.append(que.get()[1])
        if que.empty():
            break
    return confidence_list
        
        

def test(args, model, test_loader, test_dataset):
    belief_state= defaultdict(lambda : defaultdict(dict))# dial_id, # turn_id # schema
    
    model.eval()
    loss_sum = 0
    logger.info("Test start")
    with torch.no_grad():
        for iter,batch in enumerate(test_loader):
            outputs = model(input_ids=batch['input']['input_ids'].to('cuda'), labels=batch['target']['input_ids'].to('cuda'))
            outputs_text = model.generate(input_ids=batch['input']['input_ids'].to('cuda'))
            outputs_text = [args.tokenizer.decode(o).replace('</s>','').replace('<pad>','').strip() for o in outputs_text]

            for idx in range(len(outputs_text)):
                dial_id = batch['dial_id'][idx]
                turn_id = batch['turn_id'][idx]
                schema = batch['schema'][idx]
                if turn_id not in belief_state[dial_id].keys():
                    belief_state[dial_id][turn_id] = {}
                
                if outputs_text[idx] == ontology.QA['NOT_MENTIONED'] : continue
                belief_state[dial_id][turn_id][schema] = outputs_text[idx]
                test_dataset.belief_state[dial_id][turn_id][schema] = outputs_text[idx]
            

            if (iter + 1) % 50 == 0:
                logger.info('step : {}/{}'.format(
                iter+1, 
                str(len(test_loader)),
                ))
         
        with open('logs/pred_belief.json', 'w') as fp:
            json.dump(belief_state, fp, indent=4)
    
    if args.do_short: args.test_path = '../woz-data/MultiWOZ_2.1/train_data0.001.json'
    
    test_file = json.load(open(args.test_path , "r"))
    belief_state = json.load(open('logs/pred_belief.json',"r"))

    joint_goal_acc, slot_acc, domain_acc,  schema_acc, detail_wrong = evaluate_metrics(belief_state,test_file ,  args.detail_log)
    

    loss_sum += outputs.loss.cpu()

    return  joint_goal_acc, slot_acc, domain_acc, schema_acc, detail_wrong, loss_sum/iter
