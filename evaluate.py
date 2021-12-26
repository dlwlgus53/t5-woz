import json
from mwzeval.metrics import Evaluator
import json # import json module
import pdb
# with statement
with open('./data.json') as json_file:
    json_data = json.load(json_file)
pdb.set_trace()
temp = {}


for k ,v in json_data.items():
    temp[k.lower()] = v
e = Evaluator(bleu=True, success=True, richness=True)
# my_predictions = {}
# for item in data:
#     my_predictions[item.dialog_id] = model.predict(item)
#     ...
    
results = e.evaluate(temp)
# print(f"Epoch {epoch} BLEU: {results}")
print(results)