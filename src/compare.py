import os
import glob
import re
import pandas as pd


from pathlib import Path

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
    path = ''
    if flag == False:
        path = [sorted(Path(met_dir).iterdir(), key=os.path.getmtime,reverse=True)][0][0]
    else:
        path = [sorted(Path(met_dir).iterdir(), key=os.path.getmtime,reverse=True)][0][1]
    
    return path



def compare_metrics(val_met,pred_met,met_dir):
    best_model_path = ''

    val_mAP50 = val_met['mAP50'][0]
    pred_mAP50 = pred_met['mAP50'][0]
    val_mAP95 = val_met['mAP50-95'][0]
    pred_mAP95 = pred_met['mAP50-95'][0]

    #False - Validated model is better
    #True - predicted mode is better
    flag = False 

    if val_mAP50 > pred_mAP50 and val_mAP95 > pred_mAP95 :
        flag = False
        best_model_path = get_best_model(met_dir,flag)
    else:
        flag = True
        best_model_path = get_best_model(met_dir,flag)

    return best_model_path 




def main():


    train_dir = "/home/bdz1kor/Documents/GitHub/DVC_YoloV5/runs/train/"
    val_dir = "/home/bdz1kor/Documents/GitHub/DVC_YoloV5/runs/val/"
    val_met = get_metrics(train_dir)
    pred_met = get_metrics(val_dir)
    print(val_met)
    print(pred_met)
    best_model = compare_metrics(val_met,pred_met,train_dir)
    print(best_model)

if __name__ == "__main__":
    main()