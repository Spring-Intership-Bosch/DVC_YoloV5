import sys
import os
import yaml
import extras.logger as logg

sys.path.insert(0, 'model/yolov5')
import val

if len(sys.argv) != 4:
    sys.stderr.write('Arguments error. Usage:\n')
    sys.stderr.write(
        '\tpython3 src/train.py data/split data/trained data/augmented\n'
    )
    sys.exit(1)

def yolov5Model():
    output = os.path.join(sys.argv[2],f"v{params['yolov5']['ingest']['dcount']}",'images')
    os.makedirs(output, exist_ok=True)
    val.run(data='model/yolov5/data/person.yaml', weights=params['yolov5']['weights'], project='runs/yolov5/val')

def main():
    logger.info('INFERING')
    if params['model'] == 'yolov5':
        yolov5Model()
    logger.info('VALIDATING COMPLETED')

if __name__ == "__main__":
    logger = logg.log("predict.py")
    params = yaml.safe_load(open('params.yaml'))
    
    main()