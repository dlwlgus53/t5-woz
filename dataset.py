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
class Dataset(torch.utils.data.Dataset):
    def __init__(self, args, data_path, data_type):
        random.seed(args.seed)
        self.aux = args.aux
        self.data_type = data_type
        self.tokenizer = args.tokenizer
        self.dst_student_rate = args.dst_student_rate
        self.max_length = args.max_length
        self.fewshot_domain = args.fewshot_domain
        self.belief_state= defaultdict(lambda : defaultdict(dict))# dial_id, # turn_id
        self.gold_belief_state= defaultdict(lambda : defaultdict(dict))# dial_id, # turn_id
        self.gold_context= defaultdict(lambda : defaultdict(str))# dial_id, # turn_id
        self.data_type = data_type
        self.data_rate = args.data_rate
        
        
        if self.data_type == 'train':
            raw_path = f'{data_path[:-5]}1.0.json'
        else:
            raw_path = f'{data_path[:-5]}.json'
        
        
        if args.do_short:
            raw_path = f'../woz-data/MultiWOZ_2.1/train_data0.001.json' 
                

        logger.info(f"load {self.data_type} raw file {raw_path}")
        raw_dataset = json.load(open(raw_path , "r"))
        turn_id, dial_id,  question, schema, answer, gold_belief_state, gold_context, user_say, is_aux= self.seperate_data(raw_dataset)

        assert len(turn_id) == len(dial_id) == len(question)\
            == len(schema) == len(answer) == len(is_aux)
            
        self.answer = answer # for debugging
        self.target = self.encode(answer)
        self.turn_id = turn_id
        self.dial_id = dial_id
        self.question = question
        self.schema = schema
        self.user_say = user_say
        self.is_aux = is_aux
        
        self.gold_belief_state = gold_belief_state
        self.gold_context = gold_context
            
            
            
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
    
    def seperate_data(self, dataset):
        user_say= defaultdict(lambda : defaultdict(str)) 
        gold_belief_state= defaultdict(lambda : defaultdict(dict))# dial_id, # turn_id
        gold_context= defaultdict(lambda : defaultdict(str))# dial_id, # turn_id
        
        question = []
        answer = []
        schema = []
        dial_id = []
        turn_id = []
        is_aux = []
        is_fewrate_over = False
        for d_idx, d_id in enumerate(dataset.keys()):
            if d_idx/len(dataset.keys()) > self.data_rate:
                is_fewrate_over = True
            dialogue = dataset[d_id]['log']
            dialogue_text = ""
            for t_id, turn in enumerate(dialogue):
                dialogue_text += ' [user] '
                dialogue_text += turn['user']
                user_say[d_id][t_id] = turn['user']
                for key_idx, key in enumerate(ontology.QA['all-domain']): # TODO
                    domain = key.split("-")[0]
                    if self.fewshot_domain and self.data_type == 'train'\
                        and domain == self.fewshot_domain and is_fewrate_over == True : continue 
                        
                    if self.fewshot_domain and self.data_type != 'train' \
                        and domain != self.fewshot_domain: continue
                        
                    ##################### changed part #################################
                    q = ontology.QA[key]['description1']
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
                    is_aux.append(False)
                
                gold_belief_state[d_id][t_id] = turn['belief']
                gold_context[d_id][t_id] = dialogue_text
                
                dialogue_text += ' [sys] '
                dialogue_text += turn['response']
                
        
        is_fewrate_over = False
        for d_idx, d_id in enumerate(dataset.keys()):
            if d_idx/len(dataset.keys()) > self.data_rate:
                is_fewrate_over = True
                
            dialogue = dataset[d_id]['log']
            dialogue_text = ""
            
            for t_id, turn in enumerate(dialogue):
                dialogue_text += ' [user] '
                dialogue_text += turn['user']
                user_say[d_id][t_id] = turn['user']
                if self.data_type == 'train' and self.aux == 1:

                    for key_idx, key in enumerate(ontology.QA['all-domain']): # TODO
                        domain = key.split("-")[0]
                        domain_name = " ".join(key.split("-"))
                        if self.fewshot_domain \
                            and domain == self.fewshot_domain and is_fewrate_over == True : continue 
                        q = ontology.QA["general-question"] + " "+domain_name + "?" 
                        c = dialogue_text
                        if key in turn['belief']: # 언급을 한 경우
                            a = 'yes'
                        else:
                            a = ontology.QA['NOT_MENTIONED']
                        schema.append(key)
                        answer.append(a)
                        question.append(q)
                        dial_id.append(d_id)
                        turn_id.append(t_id)
                        is_aux.append(True)

        return turn_id, dial_id,  question, schema, answer, gold_belief_state, gold_context, user_say, is_aux

    def __getitem__(self, index):
        dial_id = self.dial_id[index]
        turn_id = self.turn_id[index]
        schema = self.schema[index]
        question = self.question[index]
        is_aux = self.is_aux[index]
        
        gold_context = self.gold_context[index]
        gold_belief_state = self.gold_belief_state[index]
        
        
        target = {k:v.squeeze() for (k,v) in self.target[index].items()}
        
        return {"target": target,"turn_id" : turn_id,"question" : question, "gold_context" : gold_context,\
            "dial_id" : dial_id, "is_aux" : is_aux, "schema":schema,  "gold_belief_state" : gold_belief_state }
    
    def _belief_clean(self, belief_dict):
        clean_belief = str(belief_dict).replace('{','').replace('}','')
        clean_belief = clean_belief.replace("'","")
        clean_belief = clean_belief.replace("-", " ")
        return clean_belief
    
    def collate_fn(self, batch):
        """
        The tensors are stacked together as they are yielded.
        Collate function is applied to the output of a DataLoader as it is yielded.
        context = self.context[index]
        belief_state = self.belief_state[index]
        """
        
        dial_id = [x["dial_id"] for x in batch]
        turn_id = [x["turn_id"] for x in batch]
        question = [x["question"] for x in batch]
        schema = [x["schema"] for x in batch]
        target_list = [x["target"] for x in batch]
        is_aux = [x["is_aux"] for x in batch]
        
        
        if self.data_type == 'test':
            belief = [self.belief_state[d][t-1]for (d,t) in zip(dial_id, turn_id)] 
        elif self.data_type == 'train' or self.data_type == 'val':
            belief = [self.belief_state[d][t-1]for (d,t) in zip(dial_id, turn_id)]
            # belief = [self.gold_belief_state[d][t-1]for (d,t) in zip(dial_id, turn_id)] 
        
        history = [self.gold_context[d][t] for (d,t) in zip(dial_id, turn_id)]
        input_source = [f"question : {q} context : {c} belief : {self._belief_clean(b)}" for (q,c,b) in  \
            zip(question, history, belief)]
        
        source = self.encode(input_source)
        source_list = [{k:v.squeeze() for (k,v) in s.items()} for s in source]
            
        pad_source = self.tokenizer.pad(source_list,padding=True)
        pad_target = self.tokenizer.pad(target_list,padding=True)
        
        return {"input": pad_source, "target": pad_target,"is_aux" : is_aux, \
                "schema":schema, "dial_id":dial_id, "turn_id":turn_id }
        

if __name__ == '__main__':
    import argparse
    init_logger(f'data_process.log')
    logger = logging.getLogger("my")

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_rate' ,  type = float, default=0.1)
    parser.add_argument('--student_rate' ,  type = float, default=0.0)
    parser.add_argument('--do_short' ,  type = int, default=1)
    parser.add_argument('--dst_student_rate' ,  type = float, default=0.5)
    parser.add_argument('--res_student_rate' ,  type = float, default=0.5)
    parser.add_argument('--seed' ,  type = float, default=1)
    parser.add_argument('--max_length' ,  type = float, default=128)
    parser.add_argument('--never_split_file',  default='./asset/never_split.txt', type=str,help='number of gpus per node')
    parser.add_argument('--base_trained', type = str, default = "google/t5-small-ssm-nq", help =" pretrainned model from 🤗")
    parser.add_argument('--zeroshot_domain', type=str, help='zeroshot option')
    parser.add_argument('--aux', type=int,default = 1)
    
    args = parser.parse_args()
    args.data_path = '../woz-data/MultiWOZ_2.1/train_data.json'
    
    from transformers import T5Tokenizer
    args.tokenizer = T5Tokenizer.from_pretrained('t5-small')
    with open(args.never_split_file, "r") as f:
        never_split = f.read().splitlines() 
        
    special_tokens_dict = {'additional_special_tokens': never_split}
    args.tokenizer = T5Tokenizer.from_pretrained(args.base_trained)
    args.tokenizer.add_special_tokens(special_tokens_dict)
    
    dataset = Dataset(args, args.data_path, 'train')
    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=16, collate_fn=dataset.collate_fn)
    t  = args.tokenizer
    for batch in loader:
        print(t.decode(batch['input']['input_ids'][5]))
        print(t.decode(batch['target']['input_ids'][5]))
        
        pdb.set_trace()
    
    