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
random.seed(1)
logger = logging.getLogger("my")

class Dataset(torch.utils.data.Dataset):
    def __init__(self, args, data_path, data_type, student_rate = 1.0):
        self.tokenizer = args.tokenizer
        self.prev_belief_state= defaultdict(lambda : defaultdict(dict))# dial_id, # turn_id
        self.student_rate = student_rate
        self.data_type = data_type
        
        
        if self.data_type == 'train':
            pickle_path = f'data/preprocessed_{self.data_type}{args.data_rate}.pickle'
            raw_path = f'{data_path[:-5]}{args.data_rate}.json'
        else:
            pickle_path = f'data/preprocessed_{self.data_type}.pickle'
            raw_path = f'{data_path[:-5]}.json'
        
        
        if args.do_short:
            pickle_path = f'data/preprocessed_train0.001.pickle'
            
        try:
            logger.info(f"load {pickle_path}")
            with open(pickle_path, 'rb') as f:
                item = pickle.load(f)
            
            
            self.turn_id, self.dial_id, self.source, self.target, self.schema = \
                self.sort_item(item)
        except Exception as e:
            logger.error(e)
            logger.info("Failed to load processed file. Start processing")
            raw_dataset = json.load(open(raw_path , "r"))
            context, question, answer,  belief, dial_id, turn_id, schema = self.seperate_n_sort_data(raw_dataset)
            # TODO belief에  mltiple 번호 나온다
            assert len(context)==len(question) == len(schema) == len(belief) == len(dial_id) == len(turn_id)
            
            input_text = [f"question: {q} context: {c} belief: {b}" for (q,c,b) in zip(question, context, belief)]

            logger.info("Encoding dataset (it will takes some time)")
            logger.info("encoding input text")
            self.source = self.encode(input_text)
            logger.info("encoding answer")
            self.target = self.encode(answer)
            self.schema = schema
            self.dial_id = dial_id
            self.turn_id = turn_id
            
            item = {
                'source' : self.source,
                'target' : self.target,
                'schema' : self.schema,
                'dial_id' : self.dial_id,
                'turn_id' : self.turn_id,
            }
            
            with open(pickle_path, 'wb') as f:
                pickle.dump(item, f, pickle.HIGHEST_PROTOCOL)
            
            
    def encode(self, texts ,return_tensors="pt"):
        examples = []
        for i, text in enumerate(texts):
            
            # Truncate
            while True:
                tokenized = self.tokenizer.batch_encode_plus([text], padding=False, return_tensors=return_tensors) # TODO : special token
                if len(tokenized)> self.tokenizer.model_max_length:
                    idx = [m.start() for m in re.finditer("\[user\]", text)]
                    text = text[:idx[0]] + text[idx[1]:] # delete one turn
                else:
                    break
                
            examples.append(tokenized)
        return examples

    def __getitem__(self, index):
        source = {k:v.squeeze() for (k,v) in self.source[index].items()}
        target = {k:v.squeeze() for (k,v) in self.target[index].items()}
            
        return {"source": source, "target": target, \
                "turn_id" : (self.turn_id[index]), "dial_id" : (self.dial_id[index]), "schema":(self.schema[index])}
    
    def __len__(self):
        return len(self.source)

    def sort_item(self,item):
        source = item['source']
        target = item['target']
        schema = item['schema']
        dial_id = item['dial_id']
        turn_id = item['turn_id']
        
        for_sort = [[turn,d,s,t,schema] for (turn,d,s,t,schema) in zip(turn_id, dial_id, source, target, schema)]
        sorted_items = sorted(for_sort, key=lambda x: (x[0], x[1]))
        
        
        turn_id = [s[0] for s in sorted_items]
        dial_id = [s[1] for s in sorted_items]
        source = [s[2] for s in sorted_items]
        target = [s[3] for s in sorted_items]
        schema = [s[4] for s in sorted_items]
        
        
        
        return turn_id, dial_id, source,target,schema
    
    def seperate_data(self, dataset):
        context = []
        question = []
        belief = []
        answer = []
        schema = []
        dial_id = []
        turn_id = []
        
        print(f"preprocessing data")
        for id in dataset.keys():
            dialogue = dataset[id]['log']
            dialogue_text = ""
            b = {}
            
            for i, turn in enumerate(dialogue):
                dialogue_text += '[user] '
                dialogue_text += turn['user']
                for key_idx, key in enumerate(ontology.QA['all-domain']): # TODO
                    q = ontology.QA[key]['description']
                    c = dialogue_text
                    
                    if key in turn['belief']: # 언급을 한 경우
                        a = turn['belief'][key]
                        if isinstance(a, list) : a= a[0] # in muptiple type, a == ['sunday',6]
                    else:
                        a = ontology.QA['NOT_MENTIONED']
                    
                    schema.append(key)
                    answer.append(a)
                    context.append(c)
                    question.append(q)
                    belief.append(b)
                    dial_id.append(id)
                    turn_id.append(i)
                    
                b = turn['belief'] #  하나씩 밀려서 들어가야함.! 유저 다이얼처럼
                dialogue_text += '[sys] '
                dialogue_text += turn['response']
                
        for_sort = [[t,d,c,q,s,a,b] for (t,d,c,q,s,a,b) in zip(turn_id, dial_id, context, question, schema, answer,belief)]
        sorted_items = sorted(for_sort, key=lambda x: (x[0], x[1]))
        
        
        turn_id = [s[0] for s in sorted_items]
        dial_id = [s[1] for s in sorted_items]
        context = [s[2] for s in sorted_items]
        question = [s[3] for s in sorted_items]
        schema = [s[4] for s in sorted_items]
        answer = [s[5] for s in sorted_items]
        belief = [s[6] for s in sorted_items]
        
        
        return context, question, answer,  belief, dial_id, turn_id, schema
    
    

    def collate_fn(self, batch):
        """
        The tensors are stacked together as they are yielded.
        Collate function is applied to the output of a DataLoader as it is yielded.
        """
        # truncate from here
        do_student = (random.random() < self.student_rate)
        
        schema = [x["schema"] for x in batch]
        dial_id = [x["dial_id"] for x in batch]
        turn_id = [x["turn_id"] for x in batch]
        
        if do_student and self.data_type == 'test':
            texts = [self.tokenizer.decode(x["source"]["input_ids"]) for x in batch]
            idxs = [t.rfind('belief:') for t in texts] # find from behind
            prior_texts = [t[:idx] for (t,idx) in zip(texts, idxs)]
            belief_teacher = [t[idx + len('belief: '):] for (t,idx) in zip(texts, idxs)]
            
            belief = [self.prev_belief_state[d][t]for (d,t) in zip(dial_id, turn_id)] 
            
            if self.data_type !='test':
                belief = [b if b!={} else b_teacher for (b,b_teacher) in zip(belief,belief_teacher)]
            
            texts = [t + f"belief: {b}" for (t,b) in zip(prior_texts,belief)]
            
            source = self.encode(texts)
            source_list = [{k:v.squeeze() for (k,v) in s.items()} for s in source]
            
        else:
            source_list = [x["source"] for x in batch]
            
        target_list = [x["target"] for x in batch]
            
        pad_source = self.tokenizer.pad(source_list,padding=True)
        pad_target = self.tokenizer.pad(target_list,padding=True)
        
        return {"input": pad_source, "target": pad_target,\
                 "schema":schema, "dial_id":dial_id, "turn_id":turn_id}
        

if __name__ == '__main__':
    import argparse
    init_logger(f'data_process.log')
    logger = logging.getLogger("my")

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_rate' ,  type = float, default=0.01)
    parser.add_argument('--student_rate' ,  type = float, default=0.2)
    
    args = parser.parse_args()

    args.data_path = '../woz-data/MultiWOZ_2.1/train_data.json'
    from transformers import T5Tokenizer
    args.tokenizer = T5Tokenizer.from_pretrained('t5-small')
    
    dataset = Dataset(args, 'train') 
    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=4, collate_fn=dataset.collate_fn)
        
    for batch in loader:
        pdb.set_trace()
    
    