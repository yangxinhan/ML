# streamlit run web_solve.py

# 若出現錯誤：TypeError: Descriptors cannot not be created directly. 可執行下列指令：
# Set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# streamlit user guide
# https://docs.streamlit.io/en/stable/api.html#display-interactive-widgets

import streamlit as st
from sympy.solvers import solve
from sympy import symbols
from sympy.core import sympify

x = symbols('x y z')

exp1 = st.text_area('請輸入聯立方程式：', '(x+2)*(x-3), (x-5)*(x-6)')

if st.button('求解'):
    ans = solve(sympify(f"Eq({exp1})"))
    print('結果:', ans)
    st.write('結果:', ans[0])
