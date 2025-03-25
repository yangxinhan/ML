# https://docs.streamlit.io/en/stable/api.html#streamlit.slider
import streamlit as st
import pandas

# load model
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np

model_file_name='model.pickle'
with open(model_file_name, 'rb') as f:
    random_forest = pickle.load(f)

#fare_clf= StandardScaler()
fare_file_name='fare.pickle'
with open(fare_file_name, 'rb') as f:
    fare_clf = pickle.load(f)


def far_transform(fare1):
    return fare_clf.transform([[fare1]])[0]

def convert_sex(sex1):
    return 0 if sex1 == '男性' else 1

def convert_age(age1):
    return age1 // 5 * 5

dict1 = { 'Cherbourg': 0, 'Queenstown': 1, 'Southampton': 2}
def convert_embark_town(embark1):
    x = dict1[embark1]
    if x==0:
        return 1,0,0
    elif x==1:
        return 0,1,0
    else:
        return 0,0,1
        



# 畫面設計
st.markdown('# 生存預測系統')
pclass_series = pandas.Series([1, 2, 3])
sex_series = pandas.Series(['男性', '女性'])
embark_town_series = pandas.Series(['Cherbourg', 'Queenstown', 'Southampton'])

sex = st.sidebar.selectbox('性別:', sex_series)
# '性別:', sex

age = st.sidebar.slider('年齡', 0, 100, 20)
# '年齡:', age

sibsp = st.sidebar.slider('兄弟姊妹同行人數', 0, 10, 0)
# '兄弟姊妹同行人數:', sibsp

parch = st.sidebar.slider('父母子女同行人數', 0, 10, 0)
# '父母子女同行人數:', parch

embark_town = st.sidebar.selectbox('上船港口:', embark_town_series)
# '上船港口:', embark_town

pclass = st.sidebar.selectbox('艙等:', pclass_series)
# '艙等:', pclass

fare = st.sidebar.slider('票價', 0, 100, 20)
# '票價:', fare

st.image('./TITANIC.png')

if st.sidebar.button('預測'):
    '性別:', sex
    '年齡:', age
    '兄弟姊妹同行人數:', sibsp
    '父母子女同行人數:', parch
    '上船港口:', embark_town
    '艙等:', pclass
    '票價:', fare

    # predict
    X = []
    # pclass	sex	age	sibsp	parch	fare	embark_town
    X.append([pclass, convert_sex(sex), convert_age(age), sibsp, parch, far_transform(fare), dict1[embark_town]]) #, *convert_embark_town(embark_town)])
    X=np.array(X)
    if random_forest.predict(X) == 1:
        st.markdown('==> **生存**')
    else:
        st.markdown('==> **死亡**')
