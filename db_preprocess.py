import json,  os, re, copy, zipfile
import spacy
import ontology, utils
from collections import OrderedDict
from tqdm import tqdm
from db_ops import MultiWozDB
from clean_dataset import clean_slot_values, clean_text


def get_db_values(value_set_path):
    processed = {}
    bspn_word = []
    nlp = spacy.load('en_core_web_sm')

    with open(value_set_path, 'r') as f:
        value_set = json.loads(f.read().lower())

    with open('db/ontology.json', 'r') as f:
        otlg = json.loads(f.read().lower())

    for domain, slots in value_set.items():
        processed[domain] = {}
        bspn_word.append('['+domain+']')
        for slot, values in slots.items():
            s_p = ontology.normlize_slot_names.get(slot, slot)
            if s_p in ontology.informable_slots[domain]:
                bspn_word.append(s_p)
                processed[domain][s_p] = []

    for domain, slots in value_set.items():
        for slot, values in slots.items():
            s_p = ontology.normlize_slot_names.get(slot, slot)
            if s_p in ontology.informable_slots[domain]:
                for v in values:
                    _, v_p = clean_slot_values(domain, slot, v)
                    v_p = ' '.join([token.text for token in nlp(v_p)]).strip()
                    processed[domain][s_p].append(v_p)
                    for x in v_p.split():
                        if x not in bspn_word:
                            bspn_word.append(x)

    for domain_slot, values in otlg.items():
        domain, slot = domain_slot.split('-')
        if domain == 'bus':
            domain = 'taxi'
        if slot == 'price range':
            slot = 'pricerange'
        if slot == 'book stay':
            slot = 'stay'
        if slot == 'book day':
            slot = 'day'
        if slot == 'book people':
            slot = 'people'
        if slot == 'book time':
            slot = 'time'
        if slot == 'arrive by':
            slot = 'arrive'
        if slot == 'leave at':
            slot = 'leave'
        if slot == 'leaveat':
            slot = 'leave'
        if slot not in processed[domain]:
            processed[domain][slot] = []
            bspn_word.append(slot)
        for v in values:
            _, v_p = clean_slot_values(domain, slot, v)
            v_p = ' '.join([token.text for token in nlp(v_p)]).strip()
            if v_p not in processed[domain][slot]:
                processed[domain][slot].append(v_p)
                for x in v_p.split():
                    if x not in bspn_word:
                        bspn_word.append(x)

    with open(value_set_path.replace('.json', '_processed.json'), 'w') as f:
        json.dump(processed, f, indent=2)
    with open('data/multi-woz-processed/bspn_word_collection.json', 'w') as f:
        json.dump(bspn_word, f, indent=2)

    print('DB value set processed! ')

def preprocess_db(db_paths):
    dbs = {}
    nlp = spacy.load('en_core_web_sm')
    for domain in ontology.all_domains:
        with open(db_paths[domain], 'r') as f:
            dbs[domain] = json.loads(f.read().lower())
            for idx, entry in enumerate(dbs[domain]):
                new_entry = copy.deepcopy(entry)
                for key, value in entry.items():
                    if type(value) is not str:
                        continue
                    del new_entry[key]
                    key, value = clean_slot_values(domain, key, value)
                    tokenize_and_back = ' '.join([token.text for token in nlp(value)]).strip()
                    new_entry[key] = tokenize_and_back
                dbs[domain][idx] = new_entry
        with open(db_paths[domain].replace('.json', '_processed.json'), 'w') as f:
            json.dump(dbs[domain], f, indent=2)
        print('[%s] DB processed! '%domain)



if __name__=='__main__':
    db_paths = {
            'attraction': 'db/attraction_db.json',
            'hospital': 'db/hospital_db.json',
            'hotel': 'db/hotel_db.json',
            'police': 'db/police_db.json',
            'restaurant': 'db/restaurant_db.json',
            'taxi': 'db/taxi_db.json',
            'train': 'db/train_db.json',
        }
    get_db_values('db/value_set.json')
    preprocess_db(db_paths)
    