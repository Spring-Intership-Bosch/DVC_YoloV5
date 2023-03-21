import streamlit as st
import yaml
import os
import model.yolov5.detect as detect
import pandas as pd
import numpy as np
import shutil

params = yaml.safe_load(open('params.yaml'))

def pipeline():
    st.subheader('Choose Dataset')
    opts = os.listdir('buffer')
    opts.sort()
    option = st.selectbox('',opts)

    if st.button('Run Pipeline'):
        st.subheader('Running YoloV5 Pipeline..........')
        shutil.rmtree('.dvc/cache', ignore_errors=True) 
        
        params["yolov5"]['ingest']['dcount'] = params["yolov5"]['ingest']['dcount'] +1
        params["yolov5"]['ingest']['dpath'] = option
        yaml.dump(params, open('params.yaml', 'w'), sort_keys=False)
        
        if not os.system("dvc repro"):
            st.success('Pipeline executed successfully')
            show_metrics()

        else:
            st.error("Pipleine execution failed")


def show_metrics():
    dcou = params["yolov5"]['ingest']['dcount']
    if dcou != 0:
        if params["yolov5"]["weights"] == "pretrained/best.pt":
            bbb = 1
        else:
            if params["yolov5"]["weights"].split("/")[3] == 'exp':
                bbb=1
            else:
                bbb = params["yolov5"]["weights"].split("/")[3]
                bbb = bbb[-1]

        if int(bbb) == int(dcou):
            if bbb == 1:
                # st.write('current model is  runs/val/exp')
                # st.write('prev model is  runs/train/exp')
                prev_best_model = 'runs/yolov5/train/exp'
                current_model = 'runs/yolov5/val/exp'
            else:
                # st.write('current model is  runs/val/exp'+str(dcou))
                # st.write('prev model is  runs/train/exp'+str(dcou))
                prev_best_model = 'runs/yolov5/train/exp'+str(dcou)
                current_model = 'runs/yolov5/val/exp'+str(dcou)
        else:
            if bbb == 1:
                # st.write('current model is  runs/train/exp'+str(dcou))
                # st.write('prev model is  runs/val/exp'+str(dcou))
                prev_best_model = 'runs/yolov5/val/exp'
                current_model = 'runs/yolov5/train/exp'
            else:
                # st.write('current model is  runs/train/exp'+str(dcou))
                # st.write('prev model is  runs/val/exp'+str(dcou))
                prev_best_model = 'runs/yolov5/val/exp'+str(dcou)
                current_model = 'runs/yolov5/train/exp'+str(dcou)
        
        df1 = pd.read_csv(current_model+'/metrics.csv')
        df2 = pd.read_csv(prev_best_model+'/metrics.csv')
        
        coll1 = df2["F1-Score"]
        coll1 = coll1.to_numpy()
        coll1 = np.reshape(coll1,(3,1))

        coll2 = df1["F1-Score"]
        coll2 = coll2.to_numpy()
        coll2 = np.reshape(coll2,(3,1))

        chart_data = pd.DataFrame(np.concatenate((coll1,coll2), axis = 1), columns=['best model', 'New model'])
        st.write("## F1-Score")
        st.line_chart(chart_data)
        col1, col2 = st.columns(2)
        col1.write("## Previous Best model")
        col1.write("### Confusion Matrix")
        col1.image(os.path.join(prev_best_model,"confusion_matrix.png"))
        col1.write('\n')
        col1.write("### F1 Curve")
        col1.image(os.path.join(prev_best_model,"F1_curve.png"))

        col2.write("## New model")
        col2.write("### Confusion Matrix")
        col2.image(os.path.join(current_model,"confusion_matrix.png"))
        
        col2.write('\n')
        col2.write("### F1 Curve")
        col2.image(os.path.join(current_model,"F1_curve.png"))

        col1.write("### Previous Best metrics")
        metrics_path = os.path.join(prev_best_model,"metrics.csv")
        df = pd.read_csv(metrics_path)
        col1.write(df)

        col2.write("### New model metrics")
        metrics_path = os.path.join(current_model,"metrics.csv")
        df = pd.read_csv(metrics_path)
        col2.write(df)
    else:
        st.title('TRAIN A DATASET TO EVALUATE METRICS')


def hero_page():
    st.image('hero.jpeg', width=1000)

def predict_image():
    img = st.file_uploader("Upload Image")
    if img:
        with open(f'detect/image.png', "wb") as f:
            f.write(img.getbuffer())
        
        st.subheader('Predicting........')
        st.image('detect/image.png')
        detect.run(weights=params['yolov5']['weights'], source='detect/image.png')
        st.success('Prediction Successful')
        st.image('runs/yolov5/detect/exp/image.png')

def main():
    st.set_page_config(layout="wide")
    st.title("MLOps Pipeline for Pedestrian Detection")
    pages = {
        "Choose one of the following":hero_page,
        "Train Dataset": pipeline,
        "Predict on an Image": predict_image,
        "Metrics": show_metrics,
    }
    st.sidebar.image('Bosch_logo.png')
    
    st.markdown("""---""")
    st.sidebar.markdown("""---""")
    st.sidebar.title('Select Model -')
    opp = st.sidebar.selectbox('',('yolov5', 'detectron2'))
    if opp == 'yolov5':
        selected_page = st.sidebar.selectbox('',pages.keys())
        pages[selected_page]()
    else:
        st.title('DETECTRON2 UNDER CONSTRUCTION.......')

if __name__ == '__main__':
    main()