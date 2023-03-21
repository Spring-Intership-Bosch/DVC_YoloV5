import sys
import os
import yaml
import extras.logger as logg

sys.path.insert(0, 'model/yolov5')
import train

if len(sys.argv) != 4:
    sys.stderr.write('Arguments error. Usage:\n')
    sys.stderr.write(
        '\tpython3 src/train.py data/split data/trained data/augmented\n'
    )
    sys.exit(1)

def yolov5Model():
    output = os.path.join(sys.argv[2],f"v{params['yolov5']['ingest']['dcount']}",'images')
    os.makedirs(output, exist_ok=True)
    args = params['yolov5']['hyps']
    wt = params['yolov5']['weights']
    train.run(data='person.yaml', imgsz=320, weights=wt, **args)

def main():
    logger.info('TRAINING')
    if params['model'] == 'yolov5':
        yolov5Model()
    logger.info('TRAINING COMPLETED')

if __name__ == "__main__":
    logger = logg.log("train.py")
    params = yaml.safe_load(open('params.yaml'))
   
    main()