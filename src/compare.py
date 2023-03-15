import os
import sys
import pandas as pd
import yaml
from pathlib import Path
import extras.logger as logg

if len(sys.argv) != 4:
    sys.stderr.write('Arguments error. Usage:\n')
    sys.stderr.write(
        '\tpython3 src/train.py data/split data/trained data/augmented\n'
    )
    sys.exit(1)

def get_metrics(dir):
    path = [sorted(Path(dir).iterdir(), key=os.path.getmtime,reverse=True)][0][0]
    metrics_df = []
    for ex in os.listdir(path):
        if ex.endswith('.csv') and 'metrics' in ex:
            metrics_path = os.path.join(path,ex)
            metrics_df = pd.read_csv(metrics_path)
        else:
            pass
    return metrics_df

def get_best_model(met_dir,flag):
    count = params['ingest']['dcount']
    version = 0
    path = ''
    if flag == False:
        path = [sorted(Path(met_dir).iterdir(), key=os.path.getmtime,reverse=True)][0][0]
        path = str(path) + '/weights/best.pt'
        version = count
    else:
        if len(os.listdir(met_dir)) > 1:
            path = [sorted(Path(met_dir).iterdir(), key=os.path.getmtime,reverse=True)][0][1]
            path = str(path) + '/weights/best.pt'
            version = count-1
        else:
            path = 'pretrained/best.pt'
    return path,version

def compare_metrics(val_met,pred_met,met_dir):
    best_model_path = ''
    val_mAP50 = val_met['mAP50'][0]
    pred_mAP50 = pred_met['mAP50'][0]
    val_mAP95 = val_met['mAP50-95'][0]
    pred_mAP95 = pred_met['mAP50-95'][0]
    #False - Validated model is better
    #True - predicted model is better
    flag = False 
    if val_mAP50 > pred_mAP50 and val_mAP95 > pred_mAP95 :
        flag = False
        best_model_path = get_best_model(met_dir,flag)
    else:
        flag = True
        best_model_path = get_best_model(met_dir,flag)
    return best_model_path 

def yolov5Model():
    train_dir = params['yolov5']['outputs']['train_dir']
    val_dir = params['yolov5']['outputs']['val_dir']
    val_met = get_metrics(train_dir)
    pred_met = get_metrics(val_dir)
    print(val_met)
    print(pred_met)
    best_model,count = compare_metrics(val_met,pred_met,train_dir)
    print(best_model)
    logger.info('BEST MODEL PATH - '+best_model)
    params['yolov5']['weights'] = best_model
    params['yolov5']['version'] = count
    yaml.dump(params, open('params.yaml', 'w'), sort_keys=False)

def main():
    logger.info('COMPARING')
    if params['model'] == 'yolov5':
        yolov5Model()
    logger.info('COMPARING COMPLETED')

if __name__ == "__main__":
    logger = logg.log("compare.py")
    params = yaml.safe_load(open('params.yaml'))
    output = os.path.join(sys.argv[2],f"v{params['ingest']['dcount']}",'images')
    os.makedirs(output, exist_ok=True)
    main()