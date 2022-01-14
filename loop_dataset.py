# for trainning loop
import re, glob
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
    def __init__(self, args, data_type):
        random.seed(args.seed)
        self.data_type = data_type # tag, train
        self.tokenizer = args.tokenizer
        self.max_length = args.max_length
        logger.info(f"load raw file in loop dataset.py {args.untagged}")
        self.raw_dataset = json.load(open(args.untagged , "r"))
        if data_type == 'tag':
            # pass in this list
            with open(f'{args.temp_folder}/worked_list.txt', "r") as file:
                index = file.read().splitlines()
                index =[i.split(",") for i in index]
            
        elif data_type == 'train':
            index = self.load_c_file(f"{args.temp_folder}/confidence/") # read all confidence list

        turn_id, dial_id,  question, schema, answer = self.seperate_data(self.raw_dataset, index)

        assert len(turn_id) == len(dial_id) == len(question)\
            == len(schema) == len(answer)
            
        self.answer = answer
        self.target = self.encode(answer)
        self.turn_id = turn_id
        self.dial_id = dial_id
        self.question = question
        self.schema = schema
            
    def load_c_file(self, path):
        index = []
        for file in glob.glob(f"{path}/*.txt"):
            with open(file, "r") as txt_file:
                c_index = txt_file.read().splitlines()
                index += [i.split(",") for i in c_index]
            
        return index
            
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
    
    def is_in_index(self, index_file, index):
        for i in index_file:
            # dial_id, turn_id, schema
            if i[0] == index[0] and\
                int(i[1]) == index[1] and\
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
            for t_id, turn in enumerate(dialogue):
                for key_idx, key in enumerate(ontology.QA['all-domain']):
                    # 리스트에 있는 것만 학습한다. 또는 리스트에 없는것만 태깅한다
                    if (self.data_type == 'train' and self.is_in_index(index,[d_id, t_id, key])) \
                        or (self.data_type == 'tag'and not self.is_in_index(index, [d_id, t_id, key])): 
                        q = ontology.QA[key]['description']
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
        return turn_id, dial_id,  question, schema, answer

    def __getitem__(self, index):
        dial_id = self.dial_id[index]
        turn_id = self.turn_id[index]
        schema = self.schema[index]
        question = self.question[index]
        target = {k:v.squeeze() for (k,v) in self.target[index].items()}
        
        return {"target": target,"turn_id" : turn_id,"question" : question, \
            "dial_id" : dial_id, "schema":schema}
    
    def _belief_clean(self, belief_dict):
        clean_belief = str(belief_dict).replace('{','').replace('}','')
        clean_belief = clean_belief.replace("'","")
        clean_belief = clean_belief.replace("-", " ")
        return clean_belief
    
    def get_belief_state(self, dial_id, turn_id):
        if turn_id<0:
            return {}
        else:
            return self.raw_dataset[dial_id]['log'][turn_id]['belief']
        
    def get_context(self, dial_id, turn_id):
        context = ''
        self.raw_dataset
        dials = self.raw_dataset[dial_id]['log']
        for dial in dials:
            context += ' [user] '
            context += dial['user']
            context += ' [sys] '
            context += dial['response']
            if turn_id == dial['turn_num']:break
        context = context[:context.rfind(' [sys]')]
        return context
    
    def collate_fn(self, batch):
        dial_id = [x["dial_id"] for x in batch]
        turn_id = [x["turn_id"] for x in batch]
        question = [x["question"] for x in batch]
        schema = [x["schema"] for x in batch]
        target_list = [x["target"] for x in batch]

        belief = [self.get_belief_state(d, t-1) for (d,t) in zip(dial_id, turn_id)] 
        history = [self.get_context(d,t) for (d,t) in zip(dial_id, turn_id)]
        
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
    parser.add_argument('--temp_folder', type=str, default = './looptemp')

    parser.add_argument('--do_short' ,  type = int, default=1)
    parser.add_argument('--seed' ,  type = float, default=1)
    parser.add_argument('--max_length' ,  type = float, default=128)
    parser.add_argument('--never_split_file',  default='./asset/never_split.txt', type=str,help='number of gpus per node')
    parser.add_argument('--base_trained', type = str, default = "t5-small", help =" pretrainned model from 🤗")
    args = parser.parse_args()

    args.untagged = '../woz-data/MultiWOZ_2.1/test_data_short.json'
    
    from transformers import T5Tokenizer
    args.tokenizer = T5Tokenizer.from_pretrained('t5-small')
    with open(args.never_split_file, "r") as f:
        never_split = f.read().splitlines() 
        
    special_tokens_dict = {'additional_special_tokens': never_split}
    args.tokenizer = T5Tokenizer.from_pretrained(args.base_trained)
    args.tokenizer.add_special_tokens(special_tokens_dict)
    
    dataset = Dataset(args,'tag')
    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=16, collate_fn=dataset.collate_fn)
    
    for batch in loader:
        print(args.tokenizer.decode(batch['input']['input_ids'][0]))
        # pdb.set_trace()
    
    