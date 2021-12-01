# these are from trade-dst, https://github.com/jasonwu0731/trade-dst
import os
import csv
import logging
from collections import defaultdict
logger = logging.getLogger("my")

def dict_to_csv(data, file_name):
    w = csv.writer(open(f'./logs/{file_name}', "a"))
    for k,v in data.items():
        w.writerow([k,v])
    w.writerow(['===============','================='])
    
    
def makedirs(path): 
   try: 
        os.makedirs(path) 
   except OSError: 
       if not os.path.isdir(path): 
           raise
       
def evaluate_metrics(all_prediction, raw_file, slot_temp):
    turn_acc, joint_acc, turn_cnt, joint_cnt = 0, 0, 0, 0
    schema_acc = {s:0 for s in slot_temp}
    
    for key in raw_file.keys():
        if key not in all_prediction.keys(): continue
        dial = raw_file[key]['log']
        for turn_idx, turn in enumerate(dial):
            belief_label = turn['belief']
            belief_pred = all_prediction[key][turn_idx]
            
            belief_label = [f'{k} : {v}' for (k,v) in belief_label.items()] 
            belief_pred = [f'{k} : {v}' for (k,v) in belief_pred.items()] 
            if turn_idx == len(dial)-1:
                logger.info(key)
                logger.info(f'label : {sorted(belief_label)}')
                logger.info(f'pred : {sorted(belief_pred)}')
                
            if set(belief_label) == set(belief_pred):
                joint_acc += 1
            joint_cnt +=1
            
            ACC, schema_acc_temp = compute_acc(belief_label, belief_pred, slot_temp)
            
            turn_acc += ACC
            schema_acc = {k : v + schema_acc_temp[k] for (k,v) in schema_acc.items()}
            
            turn_cnt += 1
            
    # last one is schema acc
    
    return joint_acc/joint_cnt, turn_acc/turn_cnt, {k : v/turn_cnt for (k,v) in schema_acc.items()}

def compute_acc(gold, pred, slot_temp):
    miss_gold = 0
    miss_slot = []
    schema_acc = {s:1 for s in slot_temp}
    
    
    for g in gold:
        if g not in pred:
            miss_gold += 1
            schema_acc[g.split(" : ")[0]] -=1
            miss_slot.append(g.split(" : ")[0])
            
    wrong_pred = 0
    for p in pred:
        if p not in gold and p.split(" : ")[0] not in miss_slot:
            wrong_pred += 1
            schema_acc[g.split(" : ")[0]] -=1
            
    ACC_TOTAL = len(slot_temp)
    ACC = len(slot_temp) - miss_gold - wrong_pred
    ACC = ACC / float(ACC_TOTAL)
    return ACC, schema_acc
