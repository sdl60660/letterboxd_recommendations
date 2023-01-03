import datetime
import json

update_time = datetime.datetime.now().strftime('%Y-%m-%d')
with open('../frontend/src/data/meta.json', 'w') as f:
    data = {'last_updated': update_time}
    json.dump(data, f)