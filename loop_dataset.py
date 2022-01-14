# for trainning loop
import re
import pdb
import json
import torch
import pickle
import ontology
import tokenizer_config as tc
from tqdm import tqdm
import logging
from log_conf import init_logger
from collections import defaultdict
import random


logger = logging.getLogger("my")
class Dataset(torch.utils.data.Dataset): # None 출력되도록
    def __init__(self, args, file_path, data_type, confidence_list=None, worked_file=None):
        random.seed(args.seed)
        self.data_type = data_type # tag, train
        self.tokenizer = args.tokenizer
        self.max_length = args.max_length
        self.zeroshot_domain = args.zeroshot_domain
        
        logger.info(f"load raw file in loop dataset.py {args.untagged}")
        raw_dataset = json.load(open(args.untagged , "r"))
        
        if data_type == 'tag':
            # pass in this list
            worked_index = load_file(worked_list)
            
        elif data_type == 'train':
            # work only in here
            work_index = load_file(confidence_folder) # read all confidence list

        turn_id, dial_id,  question, schema, answer, gold_belief_state, gold_context, user_say= self.seperate_data(raw_dataset, index)

        assert len(turn_id) == len(dial_id) == len(question)\
            == len(schema) == len(answer)
            
        self.answer = answer
        self.target = self.encode(answer)
        self.turn_id = turn_id
        self.dial_id = dial_id
        self.question = question
        self.schema = schema
            
    def encode(self, texts ,return_tensors="pt"):
        examples = []
        for i, text in enumerate(texts):
            # Truncate
            while True:
                tokenized = self.tokenizer.batch_encode_plus([text], padding=False, return_tensors=return_tensors) # TODO : special token
                if len(tokenized)> self.max_length:
                    idx = [m.start() for m in re.finditer("\[user\]", text)]
                    text = text[:idx[0]] + text[idx[1]:] # delete one turn
                else:
                    break
                
            examples.append(tokenized)
        return examples

    def __len__(self):
        return len(self.dial_id)
    
    
    def is_in_index(index_file, index):
        for i in index_file:
            # dial_id, turn_id, schema_id
            if i[0] == index[0] and\
                i[1] == index[1] and\
                i[2] == index[2]:
                    return True            
        return False
    
    
    def seperate_data(self, dataset, index):
        question = []
        answer = []
        schema = []
        dial_id = []
        turn_id = []
        
        for d_id in dataset.keys():
            dialogue = dataset[d_id]['log']
            dialogue_text = ""
            
            for t_id, turn in enumerate(dialogue):
                dialogue_text += ' [user] '
                dialogue_text += turn['user']
                for key_idx, key in enumerate(ontology.QA['all-domain']):
                    domain = key.split("-")[0]
                    
                    if self.zeroshot_domain and \
                        self.data_type != 'test' and domain == self.zeroshot_domain: continue
                        
                    if self.zeroshot_domain and \
                        self.data_type == 'test' and domain != self.zeroshot_domain: continue
                    
                    q = ontology.QA[key]['description']
                    c = dialogue_text
                    if key in turn['belief']: # 언급을 한 경우
                        a = turn['belief'][key]
                        if isinstance(a, list) : a= a[0] # in muptiple type, a == ['sunday',6]
                    else:
                        a = ontology.QA['NOT_MENTIONED']
                    
                    schema.append(key)
                    answer.append(a)
                    question.append(q)
                    dial_id.append(d_id)
                    turn_id.append(t_id)
                dialogue_text += ' [sys] '
                dialogue_text += turn['response']
                
                    
        for_sort = [[t,d,q,s,a] for (t,d,q,s,a) in zip(turn_id, dial_id,  question, schema, answer)]
        sorted_items = sorted(for_sort, key=lambda x: (x[0], x[1]))
        
        turn_id = [s[0] for s in sorted_items]
        dial_id = [s[1] for s in sorted_items]
        question = [s[2] for s in sorted_items]
        schema_sort = [s[3] for s in sorted_items]
        answer = [s[4] for s in sorted_items]
        
        
        # sort guaranteed to be stable : it is important because of question!   
        assert schema_sort == schema
        return turn_id, dial_id,  question, schema, answer

    def __getitem__(self, index):
        dial_id = self.dial_id[index]
        turn_id = self.turn_id[index]
        schema = self.schema[index]
        question = self.question[index]
        gold_context = self.gold_context[index]
        
        
        target = {k:v.squeeze() for (k,v) in self.target[index].items()}
        
        return {"target": target,"turn_id" : turn_id,"question" : question, "gold_context" : gold_context,\
            "dial_id" : dial_id, "schema":schema}
    
    
    def _belief_clean(self, belief_dict):
        clean_belief = str(belief_dict).replace('{','').replace('}','')
        clean_belief = clean_belief.replace("'","")
        clean_belief = clean_belief.replace(":", " is")
        clean_belief = clean_belief.replace("-", " ")
        return clean_belief
    
    def collate_fn(self, batch):
        """
        The tensors are stacked together as they are yielded.
        Collate function is applied to the output of a DataLoader as it is yielded.
        context = self.context[index]
        belief_state = self.belief_state[index]
        """
        
        do_dst_student = (random.random() < self.dst_student_rate)
        
        
        dial_id = [x["dial_id"] for x in batch]
        turn_id = [x["turn_id"] for x in batch]
        question = [x["question"] for x in batch]
        schema = [x["schema"] for x in batch]
        target_list = [x["target"] for x in batch]
        
        if do_dst_student or self.data_type == 'test':
            belief = [self.belief_state[d][t-1]for (d,t) in zip(dial_id, turn_id)] 
        else:
            belief = [self.gold_belief_state[d][t-1]for (d,t) in zip(dial_id, turn_id)] 
        
        history = [self.gold_context[d][t] for (d,t) in zip(dial_id, turn_id)]
        input_source = [f"question : {q} context : {c} belief : {self._belief_clean(b)}" for (q,c,b) in  \
            zip(question, history, belief)]
        
        source = self.encode(input_source)
        source_list = [{k:v.squeeze() for (k,v) in s.items()} for s in source]
            
        pad_source = self.tokenizer.pad(source_list,padding=True)
        pad_target = self.tokenizer.pad(target_list,padding=True)
        
        return {"input": pad_source, "target": pad_target,\
                 "schema":schema, "dial_id":dial_id, "turn_id":turn_id}
        

if __name__ == '__main__':
    import argparse
    init_logger(f'data_process.log')
    logger = logging.getLogger("my")

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_rate' ,  type = float, default=0.1)
    parser.add_argument('--student_rate' ,  type = float, default=0.2)
    parser.add_argument('--do_short' ,  type = int, default=0)
    parser.add_argument('--dst_student_rate' ,  type = float, default=0.5)
    parser.add_argument('--res_student_rate' ,  type = float, default=0.5)
    parser.add_argument('--seed' ,  type = float, default=1)
    parser.add_argument('--max_length' ,  type = float, default=128)
    parser.add_argument('--never_split_file',  default='./asset/never_split.txt', type=str,help='number of gpus per node')
    parser.add_argument('--base_trained', type = str, default = "google/t5-small-ssm-nq", help =" pretrainned model from 🤗")
    parser.add_argument('--zeroshot_domain', type=str, help='zeroshot option')
    
    args = parser.parse_args()

    args.data_path = '../woz-data/MultiWOZ_2.1/train_data.json'
    
    from transformers import T5Tokenizer
    args.tokenizer = T5Tokenizer.from_pretrained('t5-small')
    with open(args.never_split_file, "r") as f:
        never_split = f.read().splitlines() 
        
    special_tokens_dict = {'additional_special_tokens': never_split}
    args.tokenizer = T5Tokenizer.from_pretrained(args.base_trained)
    args.tokenizer.add_special_tokens(special_tokens_dict)
    
    dataset = Dataset(args, args.data_path, 'train', 0)
    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=16, collate_fn=dataset.collate_fn)
    
    for batch in loader:
        pdb.set_trace()
    
    