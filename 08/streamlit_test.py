import streamlit as st
import joblib

st.title('鳶尾花（Iris）預測')

checkbox1 = st.checkbox('I agree')
radio1 = st.radio('Pick one', ['cats', 'dogs'])
selectbox1 = st.selectbox('Pick one', ['cats', 'dogs'])
multiselect1 = st.multiselect('Buy', ['milk', 'apples', 'potatoes'])
slider1 = st.slider('Pick a number', 0, 100)
select_slider1 = st.select_slider('Pick a size', ['S', 'M', 'L'])
text_input1 = st.text_input('First name')
number_input1 = st.number_input('Pick a number', 0, 10)
text_area1 = st.text_area('Text to translate')
date_input1 = st.date_input('Your birthday')
time_input1 = st.time_input('Meeting time')
file_uploader1 = st.file_uploader('Upload a CSV')
# download_button1 = st.download_button('Download file', data)
camera_input1 = st.camera_input("Take a picture")
color_picker1 = st.color_picker('Pick a color')

if st.button('預測'):
    st.write('### checkbox：', checkbox1)
    st.write('### radio：', radio1)
    st.write('### selectbox：', selectbox1)
    st.write('### multiselect：', multiselect1)
    st.write('### slider：', slider1)
    st.write('### select_slider：', select_slider1)
    st.write('### text_input：', text_input1)
    st.write('### text_area：', text_area1)
    st.write('### date_input：', date_input1)
    st.write('### time_input：', time_input1)
    st.write('### color_picker：', color_picker1)
