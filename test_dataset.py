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
logger = logging.getLogger("my")


class Test_Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path, tokenizer):
        
        self.tokenizer = tokenizer
        self.prev_belief_state= defaultdict(lambda : defaultdict(dict))# dial_id, # turn_id
        
        raw_path = data_path
            
                            
        logger.info("Failed to load processed file. Start processing")
        raw_dataset = json.load(open(raw_path , "r"))
        context, question, answer, dial_id, turn_id, schema = self.seperate_n_sort_data(raw_dataset)
        assert len(context)==len(question) == len(schema) == len(dial_id) == len(turn_id) == len(answer)
        

        logger.info("Encoding dataset (it will takes some time)")

        self.context = context
        self.question = question
        self.answer = answer
        self.schema = schema
        self.dial_id = dial_id
        self.turn_id = turn_id
            
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
        return {"context": self.context[index],\
            "question": self.question[index], \
            "answer": self.answer[index], \
            "turn_id" : (self.turn_id[index]), \
            "dial_id" : (self.dial_id[index]), \
            "schema":(self.schema[index])}
    
    def __len__(self):
        return len(self.turn_id)

    def seperate_n_sort_data(self, dataset):
        context = []
        question = []
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
                    dial_id.append(id)
                    turn_id.append(i)
                    
                dialogue_text += '[sys] '
                dialogue_text += turn['response']
        # sort by turn_id => dial+id
        for_sort = [[t,d,c,q,s,a]for (t,d,c,q,s,a) in zip(turn_id, dial_id, context, question, schema, answer)]
        sorted_items = sorted(for_sort, key=lambda x: (x[0], x[1]))
        
        
        turn_id = [s[0] for s in sorted_items]
        dial_id = [s[1] for s in sorted_items]
        context = [s[2] for s in sorted_items]
        question = [s[3] for s in sorted_items]
        schema = [s[4] for s in sorted_items]
        answer = [s[5] for s in sorted_items]
        
        
        return context, question, answer, dial_id, turn_id, schema
    
    

    def collate_fn(self, batch):
        """
        The tensors are stacked together as they are yielded.

        Collate function is applied to the output of a DataLoader as it is yielded.
        """
        question = [x["question"] for x in batch]
        context = [x["context"] for x in batch]
        schema = [x["schema"] for x in batch]
        dial_id = [x["dial_id"] for x in batch]
        turn_id = [x["turn_id"] for x in batch]
        answer = [x["answer"] for x in batch]
        belief = [self.prev_belief_state[d][t] for (d,t) in zip(dial_id, turn_id)] 
        
        input_text = [f"question: {q} context: {c} belief {dict(b)}: " for (q,c,b) in zip(question, context, belief)]
        source = self.encode(input_text)
        target = self.encode(answer)
            
        source = [{k:v.squeeze() for (k,v) in s.items()} for s in source]
        target = [{k:v.squeeze() for (k,v) in s.items()} for s in target]
        
            
        # truncate from here
        pad_source = self.tokenizer.pad(source,padding=True)
        pad_target = self.tokenizer.pad(target,padding=True)
        
        return {"input": pad_source, "target": pad_target,\
                 "schema":schema, "dial_id":dial_id, "turn_id":turn_id}
        
        # return pad_source
    
    

if __name__ == '__main__':
    init_logger(f'data_process.log')
    logger = logging.getLogger("my")

    data_path = '../woz-data/MultiWOZ_2.1/train_data.json'
    from transformers import T5Tokenizer
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    type = 'train'
    
    dataset = Test_Dataset(data_path,  tokenizer= tokenizer) 
    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=4, collate_fn=dataset.collate_fn)
        
    for batch in loader:
        pdb.set_trace()
    
    