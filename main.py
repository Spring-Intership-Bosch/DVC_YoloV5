import streamlit as st
import yaml
import os
import glob
import sys
import model.yolov5.detect as detect

import pandas as pd
import numpy as np

params = yaml.safe_load(open('params.yaml'))

def pipeline():

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


def show_metrics():
    flag = False
    op = st.selectbox('',('Choose one of the following', 'Training Metrics', 'Testing Metrics'))
    if op == 'Training Metrics':
        option = "train_dir"
        flag = True
    elif op == 'Testing Metrics':
        option = "val_dir"
        flag = True
    
    if flag:
        path = os.path.join(params['yolov5']['outputs'][option], "exp{}".format(params["yolov5"]["best"]["version"]))

        if params["ingest"]["dcount"] - 1 == params["yolov5"]["best"]["version"]:
            st.write("### Best model metrics")
            st.write("Best model is latest run model")
            metrics_path = os.path.join(path,"metrics.csv")
            df = pd.read_csv(metrics_path)
            st.write(df)

            col1 = df["mAP50"]
            col1 = col1.to_numpy()
            col1 = np.reshape(col1,(3,1))
            chart_data = pd.DataFrame(col1, columns=['best model'])
            st.write("### mAP")
            st.line_chart(chart_data)

        else:
            st.write("### Best model metrics")
            metrics_path = os.path.join(path,"metrics.csv")
            df = pd.read_csv(metrics_path)
            st.write(df)

            col1 = df["mAP50"]
            col1 = col1.to_numpy()
            col1 = np.reshape(col1,(3,1))

            path = os.path.join(params['yolov5']['outputs'][option], "exp{}".format(params['ingest']['dcount'] - 1))
            st.write("### Current model metrics")
            metrics_path = os.path.join(path,"metrics.csv")
            df = pd.read_csv(metrics_path)
            st.write(df)

            col2 = df["mAP50"]
            col2 = col2.to_numpy()
            col2 = np.reshape(col2,(3,1))

            chart_data = pd.DataFrame(np.concatenate((col1,col2), axis = 1), columns=['best model', 'current model'])
            st.write("### mAP")
            st.line_chart(chart_data)




def plot_graphs():
    col1,col2 = st.columns(2)
    val_path = os.path.join(params['yolov5']['outputs']['val_dir'], "exp{}".format(params["yolov5"]["best"]["version"]))

    if params["ingest"]["dcount"] - 1 == params["yolov5"]["best"]["version"]:
        st.write("# Best model is latest run model")
        st.write("### Confusion Matrix")
        st.image(os.path.join(val_path,"confusion_matrix.png"))
        st.write('\n')
        st.write("### F1 Curve")
        st.image(os.path.join(val_path,"F1_curve.png"))
    
    else:
        col1.write("## Best model")

        col1.write("### Confusion Matrix")
        col1.image(os.path.join(val_path,"confusion_matrix.png"))
        col1.write('\n')
        col1.write("### F1 Curve")
        col1.image(os.path.join(val_path,"F1_curve.png"))

        val_path = os.path.join(params['yolov5']['outputs']['val_dir'], "exp{}".format(params['ingest']['dcount'] - 1))
        col2.write("## Current model")
        col2.write("### Confusion Matrix")
        col2.image(os.path.join(val_path,"confusion_matrix.png"))
        
        col2.write('\n')
        col2.write("### F1 Curve")
        col2.image(os.path.join(val_path,"F1_curve.png"))

    
        

def main():
    st.title("MLOps Pipeline")
    pages = {
        "Mlops pipeline": pipeline,
        "Metrics": show_metrics,
        "Plot Graphs": plot_graphs,
    }
    st.sidebar.title('Options')
    selected_page = st.sidebar.radio("Select a page", pages.keys())
    pages[selected_page]()

if __name__ == '__main__':
    main()