import os
import yaml
import zipfile
import sys
import extras.logger as logg

params = yaml.safe_load(open('params.yaml'))['ingest']
data_path = os.path.join('data', 'prepared', f"v{params['dcount']}")
origimg_path = os.path.join('data', 'store', f"v{params['dcount']}")
datasets_path = os.path.join('datasets', f"v{params['dcount']}")

os.makedirs(data_path, exist_ok=True)
os.makedirs(origimg_path, exist_ok=True)
logger = logg.log("ingest.py")
logger.info('INGESTING DATASET')
sys.path.append('../')
with zipfile.ZipFile(f'buffer/dataset{params["dcount"]}.zip',"r") as zipf:
    zipf.extractall(data_path)
    zipf.extractall(origimg_path)
    zipf.extractall(datasets_path)