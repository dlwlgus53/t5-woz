import json
from mwzeval.metrics import Evaluator
import json # import json module

# with statement
with open('./pptod.json') as json_file:
    json_data = json.load(json_file)

e = Evaluator(bleu=True, success=True, richness=True)
# my_predictions = {}
# for item in data:
#     my_predictions[item.dialog_id] = model.predict(item)
#     ...
    
results = e.evaluate(json_data)
# print(f"Epoch {epoch} BLEU: {results}")
print(results)