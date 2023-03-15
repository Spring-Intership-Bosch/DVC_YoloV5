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
    args = {k:v for e in params['yolov5'] for (k,v) in e.items()}
    logger.info('TRAINING')
    train.run(data='person.yaml', imgsz=640, **args)
    logger.info('TRAINING COMPLETED')

def main():
    if params['model'] == 'yolov5':
        yolov5Model()


if __name__ == "__main__":
    logger = logg.log("train.py")
    params = yaml.safe_load(open('params.yaml'))
    output = os.path.join(sys.argv[2],f"v{params['ingest']['dcount']}",'images')
    os.makedirs(output, exist_ok=True)
    main()