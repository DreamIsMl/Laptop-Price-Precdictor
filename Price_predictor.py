import pickle

import numpy as np
import streamlit as st

with open('F:\\class\\Machine Learning\\project\\Laptop price predection\\pipe.pkl', 'rb') as model_file:
    pipe = pickle.load(model_file)

# Load the DataFrame
with open('F:\\class\\Machine Learning\\project\\Laptop price predection\\df.pkl', 'rb') as df_file:
    df = pickle.load(df_file)

# Brand
company = st.selectbox("Brand",df['Company'].unique())

# type of laptop
type = st.selectbox('Type',df['TypeName'].unique())

# Ram
ram = st.selectbox('Ram(in GB)',[2,4,6,8,16,24,32,64])

# weight
weight = st.number_input("Enter Weight")

# Touchscreen
touchscreen = st.selectbox('Touchscreen',['NO','Yes'])

#  Ips
ips = st.selectbox('IPS',['Yes','No'])

# screensize
screen_size = st.number_input("Screen Size")

# resulation
resolution = st.selectbox('Screen Resolution',['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'])

# Cpu

cpu = st.selectbox('CPU',df['Cpu brand'].unique())
hdd = st.selectbox('HDD(in GB)',[0,128,256,512,1024])
sdd = st.selectbox('SSD(in GB)',[0,128,256,512,1024])
gpu = st.selectbox("GPU",df['Gpu brand'].unique())
os = st.selectbox("OS",df['os'].unique())

if st.button('predict price'):
    ppi = None
    if touchscreen == "Yes":
        touchscreen = 1
    else:
        touchscreen  = 0
        
    if ips == 'Yes':
        ips = 1
    else:
        ips = 0
        
    x_res = int(resolution.split("x")[0])
    y_res = int(resolution.split('x')[1])
    ppi = ((x_res**2) + (y_res**2))**0.5/screen_size
    qurey = np.array([company,type,ram,weight,touchscreen,ips,ppi,cpu,hdd,sdd,gpu,os])
    
    qurey = qurey.reshape(1,12)
    st.title("The prediction price of the configration is " + str(int(np.exp(pipe.predict(qurey)[0]))))