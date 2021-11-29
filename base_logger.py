from datetime import datetime
import logging
import datetime
import os
def makedirs(path): 
   try: 
        os.makedirs(path) 
   except OSError: 
       if not os.path.isdir(path): 
           raise
       
       
makedirs("./logs");
now_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
logger = logging
log_file = f"./logs/0.1.log"
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler = logging.FileHandler(log_file)
file_handler.setFormatter(formatter)

logger = logging.getLogger()
logger.addHandler(file_handler)
logger.setLevel(logging.INFO)   
