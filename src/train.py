import sys
import os
import yaml
sys.path.insert(0, 'model/yolov5')
import train

if len(sys.argv) != 4:
    sys.stderr.write('Arguments error. Usage:\n')
    sys.stderr.write(
        '\tpython3 src/train.py data/split data/trained data/augmented\n'
    )
    sys.exit(1)

params = yaml.safe_load(open('params.yaml'))['ingest']

output = os.path.join(sys.argv[2],f"v{params['dcount']}",'images')
os.makedirs(output, exist_ok=True)


train.run(data='person.yaml', imgsz=640, weights='yolov5s.pt', epochs=1, batch=16)