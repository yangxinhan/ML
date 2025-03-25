import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from sklearn.datasets import make_regression
import numpy as np
import tensorflow as tf 

# 圖形更新的頻率
PAUSE_INTERVAL=0.5

# 產生圖形大小
fig, ax = plt.subplots()
fig.set_size_inches(10, 6)

# 產生隨機資料
X, y= make_regression(n_samples=100, n_features=1, noise=15, bias=50)
X=X.ravel()
# print(X, y)            
plt.scatter(X,y)
line, = ax.plot(X, [0] * len(X), 'g')

# 求預測值(Y hat)
def predict(X):
    return w * X + b  

# 計算損失函數 MSE
def loss(y, y_pred):
    return tf.reduce_mean(tf.square(y - y_pred))

def draw(w, b):
    # 更新圖形Y軸資料
    y_new = [b + w * xplot for xplot in X]
    line.set_data(X, y_new)  # update the data.
    #ax.cla()
    plt.pause(PAUSE_INTERVAL)

# 定義訓練函數
def train(X, y, epochs=40, lr=0.1):
    current_loss=0                                # 損失函數值
    for epoch in range(epochs):                   # 執行訓練週期
        with tf.GradientTape() as t:              # 自動微分
            t.watch(tf.constant(X))               # 宣告 TensorFlow 常數參與自動微分
            current_loss = loss(y, predict(X))    # 計算損失函數值
        
        dw, db = t.gradient(current_loss, [w, b]) # 取得 w, b 個別的梯度

        # 更新權重：新權重 = 原權重 — 學習率(learning_rate) * 梯度(gradient)
        w.assign_sub(lr * dw) # w -= lr * dw
        b.assign_sub(lr * db) # b -= lr * db
        
        # 更新圖形
        draw(w, b)
        
        # 顯示每一訓練週期的損失函數
        print(f'Epoch {epoch}: Loss: {current_loss.numpy()}') 
        
# w、b 初始值均設為 0
w = tf.Variable(0.0)
b = tf.Variable(0.0)

# 執行訓練
train(X, y)

# w、b 的最佳解
print(f'w={w.numpy()}, b={b.numpy()}')
        
plt.show()