from tkinter import *

# 【辨識】處理函數
def classify(widget):
    pass

def clear(widget):
    pass


root = Tk()
root.title('手寫阿拉伯數字辨識')

c = Canvas(root, bg='white', width=280, height=280)
c.grid(row=1, columnspan=6)

classify_button = Button(root, text='辨識', command=lambda:classify(c))
classify_button.grid(row=0, column=0, columnspan=2, sticky='EWNS')

clear = Button(root, text='清畫面', command=clear)
clear.grid(row=0, column=2, columnspan=2, sticky='EWNS')
root.mainloop()