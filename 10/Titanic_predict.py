# https://docs.streamlit.io/en/stable/api.html#streamlit.slider
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# load model
@st.cache
def load_model():
    clf = joblib.load('model.joblib')
    scaler = joblib.load('scaler.joblib')
    return clf, scaler
    
def convert_sex(sex1):
    return 1 if sex1 == '男性' else 0

def convert_age(age1):
    bins = [0, 12, 18, 25, 35, 60, 100]
    return pd.cut([age1], bins, labels=range(len(bins)-1))[0]
     

dict1 = { 'Southampton': 0, 'Cherbourg': 1, 'Queenstown': 2}
def convert_embark_town(embark1):
    return dict1[embark1]
        
clf, scaler = load_model()    

# 畫面設計
st.markdown('# 生存預測系統')
pclass_series = pd.Series([1, 2, 3])
sex_series = pd.Series(['女性', '男性'])
embark_town_series = pd.Series(['Southampton', 'Cherbourg', 'Queenstown'])

# '性別:', sex
sex = st.sidebar.radio('性別:', sex_series)

# '年齡:', age
age = st.sidebar.slider('年齡', 0, 100, 20)

# '兄弟姊妹同行人數:', sibsp
sibsp = st.sidebar.slider('兄弟姊妹同行人數', 0, 10, 0)

# '父母子女同行人數:', parch
parch = st.sidebar.slider('父母子女同行人數', 0, 10, 0)

# '上船港口:', embark_town
embark_town = st.sidebar.selectbox('上船港口:', embark_town_series)

# '艙等:', pclass
pclass = st.sidebar.selectbox('艙等:', pclass_series)

# '票價:', fare
fare = st.sidebar.slider('票價', 0, 100, 20)

st.image('./TITANIC.png')

if st.sidebar.button('預測'):
    # predict
    X = []
    # pclass	sex	age	sibsp	parch	fare	adult_male	embark_town
    adult_male = 1 if age>=20 and sex == '男性' else 0
    X.append([pclass, convert_sex(sex), convert_age(age), sibsp, parch, fare, adult_male, convert_embark_town(embark_town)])
    X=scaler.transform(np.array(X))
    
    if clf.predict(X) == 1:
        st.markdown(f'### ==> **生存, 生存機率={clf.predict_proba(X)[0][1]:.2%}**')
    else:
        st.markdown(f'### ==> **死亡, 生存機率={clf.predict_proba(X)[0][1]:.2%}**')
