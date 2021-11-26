import pdb
# these are from trade-dst, https://github.com/jasonwu0731/trade-dst

def evaluate_metrics(all_prediction, raw_file, slot_temp):
    turn_acc, joint_acc, turn_cnt, joint_cnt = 0, 0, 0, 0
    
    for key in raw_file.keys():
        if key not in all_prediction.keys(): continue
        dial = raw_file[key]['log']
        for turn_idx, turn in enumerate(dial):
            belief_label = turn['belief']
            belief_pred = all_prediction[key][turn_idx]
            
            belief_label = [f'{k} : {v}' for (k,v) in belief_label.items()] 
            belief_pred = [f'{k} : {v}' for (k,v) in belief_pred.items()] 
            if set(belief_label) == set(belief_pred):
                joint_acc += 1
            joint_cnt +=1
            
            turn_acc += compute_acc(belief_label, belief_pred, slot_temp)
            turn_cnt += 1
            
    return joint_acc/joint_cnt, turn_acc/turn_cnt

def compute_acc(gold, pred, slot_temp):
    miss_gold = 0
    miss_slot = []
    for g in gold:
        if g not in pred:
            miss_gold += 1
            miss_slot.append(g.split(" : ")[0])
    wrong_pred = 0
    for p in pred:
        if p not in gold and p.split(" : ")[0] not in miss_slot:
            wrong_pred += 1
    ACC_TOTAL = len(slot_temp)
    ACC = len(slot_temp) - miss_gold - wrong_pred
    ACC = ACC / float(ACC_TOTAL)
    return ACC

