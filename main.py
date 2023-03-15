import streamlit as st
import yaml
import os
import glob
import sys

def main():
    st.title("YoloV5 MLOps Pipeline")
    # st.write(sys.executable)
    
    st.subheader('Choose Dataset')
    print(os.listdir('buffer'))
    opts = os.listdir('buffer')
    opts.sort()
    option = st.selectbox('',opts)
    st.write('You selected:', option)

    if st.button('Run Pipeline'):
        st.write('Running YoloV5 Pipeline..........')
        params = yaml.safe_load(open('params.yaml'))
        params['ingest']['dcount'] = params['ingest']['dcount'] +1
        params['ingest']['dpath'] = option
        yaml.dump(params, open('params.yaml', 'w'), sort_keys=False)

        if not os.system("dvc repro"):
            st.success('Pipeline executed successfully')
    
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
    main()