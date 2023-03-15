import streamlit as st
import yaml
import os
import glob
import sys
import model.yolov5.detect as detect

def main():
    st.title("MLOps Pipeline")

    op = st.selectbox('',('Choose one of the following', 'Predict on an image', 'Train new dataset'))
    if op == 'Predict on an image':
        img = st.file_uploader("Upload Image")
        
        if img:
            with open(f'detect/image.png', "wb") as f:
                f.write(img.getbuffer())
            
            st.subheader('Predicting........')
            st.image('detect/image.png')
            detect.run(weights=params['yolov5']['weights'], source='detect/image.png')
            st.success('Prediction Successful')
            st.image('runs/detect/exp/image.png')

    elif op == 'Train new dataset':
        st.subheader('Choose Dataset')
        opts = os.listdir('buffer')
        opts.sort()
        option = st.selectbox('',opts)

        if st.button('Run Pipeline'):
            st.subheader('Running YoloV5 Pipeline..........')
            
            params['ingest']['dcount'] = params['ingest']['dcount'] +1
            params['ingest']['dpath'] = option
            yaml.dump(params, open('params.yaml', 'w'), sort_keys=False)

            if not os.system("dvc repro"):
                st.success('Pipeline executed successfully')
            else:
                st.error("Pipleine execution failed")
    
    # if imgs:
    #     params = yaml.safe_load(open('params.yaml'))['ingest']
    #     ddd = {'ingest': {'dcount': params['dcount']+1}}
    #     yaml.dump(ddd, open('params.yaml', 'w'))
        
    #     os.makedirs('buffer', exist_ok=True)
    #     for ff in os.listdir('buffer'):
    #         os.remove(f'buffer/{ff}')
    
    #     with open(f'buffer/dataset{params["dcount"]+1}.zip', "wb") as f:
    #         f.write(imgs.getbuffer())
        
    #     print('hello................', sys.executable)
        
        # if not os.system("dvc repro"):
        #     st.success('Pipeline executed successfully')
        #     imgname = os.listdir("data/store/v{}/evaluated".format(params["dcount"]+1))
        #     preds = glob.glob("data/store/v{}/evaluated/*.*".format(params["dcount"]+1), recursive=True)
        #     for index,im in enumerate(preds):
        #         st.image(im, imgname[index])
        #     print('done')
    # else:
    #     return

if __name__ == '__main__':
    params = yaml.safe_load(open('params.yaml'))
    main()