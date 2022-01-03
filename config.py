import logging, time, os

class _Config:
    def __init__(self):
        self._multiwoz_damd_init()

    def _multiwoz_damd_init(self):
        self.dbs = {
            'attraction': 'db/attraction_db_processed.json',
            'hospital': 'db/hospital_db_processed.json',
            'hotel': 'db/hotel_db_processed.json',
            'police': 'db/police_db_processed.json',
            'restaurant': 'db/restaurant_db_processed.json',
            'taxi': 'db/taxi_db_processed.json',
            'train': 'db/train_db_processed.json',
        }

global_config = _Config()

