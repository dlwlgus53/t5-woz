# these are from trade-dst, https://github.com/jasonwu0731/trade-dst
import os
import csv, json

import logging
import ontology
from collections import defaultdict
import pdb
logger = logging.getLogger("my")
import pickle


def idx_to_text(tokenizer, idx):
    pass
def dict_to_csv(data, file_name):
    w = csv.writer(open(f'./logs/csvs/{file_name}', "w"))
    for k,v in data.items():
        w.writerow([k,v])
    w.writerow(['===============','================='])
    
    
def dict_to_json(data, file_name):
    with open(f'./logs/jsons/{file_name}', 'w') as fp:
        json.dump(data, fp,  indent=4)

def makedirs(path): 
   try: 
        os.makedirs(path) 
   except OSError: 
       if not os.path.isdir(path): 
           raise
       
def evaluate_metrics(all_prediction, raw_file, detail_log):
    # schema = ontology.QA['all-domain'][:-1] # next response 는 제외
    schema = ontology.QA['all-domain']# next response 는 제외
    domain = ontology.QA['bigger-domain']
    
    detail_wrongs= defaultdict(lambda : defaultdict(list))# dial_id, # turn_id # schema
    turn_acc, joint_acc, turn_cnt, joint_cnt = 0, 0, 0, 0
    schema_acc = {s:0 for s in schema}
    domain_acc = {s:0 for s in domain}
    
    for key in raw_file.keys():
        if key not in all_prediction.keys(): continue
        dial = raw_file[key]['log']
        for turn_idx, turn in enumerate(dial):
            belief_label = turn['belief']
            belief_pred = all_prediction[key][str(turn_idx)]
            
            belief_label = [f'{k} : {v}' for (k,v) in belief_label.items()] 
            belief_pred = [f'{k} : {v}' for (k,v) in belief_pred.items()] 
            if turn_idx == len(dial)-1:
                logger.info(key)
                logger.info(f'label : {sorted(belief_label)}')
                logger.info(f'pred : {sorted(belief_pred)}')
                
            if set(belief_label) == set(belief_pred):
                joint_acc += 1
            joint_cnt +=1
            
            ACC, schema_acc_temp, domain_acc_temp, detail_wrong  = compute_acc(belief_label, belief_pred, schema,domain,detail_log)
            
            turn_acc += ACC
            schema_acc = {k : v + schema_acc_temp[k] for (k,v) in schema_acc.items()}
            domain_acc = {k : v + domain_acc_temp[k] for (k,v) in domain_acc.items()}
            
            if detail_log:
                detail_wrongs[key][turn_idx] = detail_wrong
            
            turn_cnt += 1
            
    domain_acc = {k : v/turn_cnt for (k,v) in domain_acc.items()}
    schema_acc = {k : v/turn_cnt for (k,v) in schema_acc.items()}
    
    return joint_acc/joint_cnt, turn_acc/turn_cnt, domain_acc, schema_acc, detail_wrongs

def evaluate_response(belief_state, response):
    e = Evaluator(bleu=True, success=True, richness=True)
    answer_dict = {}
    dial_keys = belief_state.keys()
    
    for dial_idx in dial_keys:
        turn_answers = []
        for turn_idx in range(len(response[dial_idx])):
            turn_answer = {}
            states = defaultdict(dict)
            for k,v in belief_state[dial_idx][str(turn_idx)].items():
                states[k.split("-")[0]][k.split("-")[1]] = v
            turn_answer['state'] = states
            turn_answer['response'] = response[dial_idx][turn_idx]
            turn_answers.append(turn_answer)
        
        answer_dict[dial_idx.lower()] = turn_answers
    with open('logs/response.json', 'w') as fp:
        json.dump(answer_dict, fp, indent=4)
    results = e.evaluate(answer_dict)
    logger.info(results)
            
def compute_acc(gold, pred, slot_temp, domain, detail_log):
    # import pdb; pdb.set_trace()
    detail_wrong = []
    miss_gold = 0
    miss_slot = []
    schema_acc = {s:1 for s in slot_temp}
    domain_acc = {s:1 for s in domain}
    
    
    
    for g in gold:
        if g not in pred:
            miss_gold += 1
            schema_acc[g.split(" : ")[0]] -=1
            domain_acc[g.split("-")[0]] -=1
            miss_slot.append(g.split(" : ")[0])
            if detail_log:    
                for p in pred:
                    if p.startswith(miss_slot[-1]):
                        detail_wrong.append((g,p))
                        break
                else:
                    detail_wrong.append((g,ontology.QA['NOT_MENTIONED']))
                                    
            
    wrong_pred = 0
    for p in pred:
        if p not in gold and p.split(" : ")[0] not in miss_slot:
            wrong_pred += 1
            schema_acc[p.split(" : ")[0]] -=1
            domain_acc[p.split("-")[0]] -=1
            if detail_log:
                detail_wrong.append((ontology.QA['NOT_MENTIONED'],p))
            
        
    ACC_TOTAL = len(slot_temp)
    ACC = len(slot_temp) - miss_gold - wrong_pred
    ACC = ACC / float(ACC_TOTAL)
    return ACC, schema_acc, domain_acc, detail_wrong



def save_pickle(file_name, data):
    with open(file_name, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

def load_pickle(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)
    