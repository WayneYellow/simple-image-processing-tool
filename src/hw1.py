"""
This is a simple image processing program with a simple GUI.
B113040040 黃楚文
Date: 2023/10/02
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

IMAGE_SHOW_SIZE = 600
class UI(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack()
        self.create_widgets()
        self.master.geometry("1000x600")
        self.master.title("Image Processing")

    def create_widgets(self):
        self.frame1 = tk.Frame(self)
        self.frame1.pack(side="left")
        self.image_box = tk.Label(self.frame1, padx=0, pady=0, anchor="nw", bd=5)
        self.image_box.pack()
        self.frame2 = tk.Frame(self)
        self.frame2.pack(side="right")

        self.contrast_group = tk.LabelFrame(self.frame2, text="Contrast/Brightness", padx=5, pady=5)
        self.contrast_group.grid(row=0, column=0, columnspan=2)
        self.lable1 = tk.Label(self.contrast_group, text="linear:")
        self.lable1.grid(row=0, column=0, columnspan=4)
        self.lablea = tk.Label(self.contrast_group, text="a:")
        self.lablea.grid(row=1, column=0)
        self.contrast_linear_a = tk.Spinbox(self.contrast_group, from_=0, to=9999999, increment=1, width=5, command=self.AdjustContrastLinear)
        self.contrast_linear_a.grid(row=1, column=1)
        self.lableb = tk.Label(self.contrast_group, text="b:")
        self.lableb.grid(row=1, column=2)
        self.contrast_linear_b = tk.Spinbox(self.contrast_group, from_=0, to=9999999, increment=1, width=5, command=self.AdjustContrastLinear)
        self.contrast_linear_b.grid(row=1, column=3)
        self.lable2 = tk.Label(self.contrast_group, text="exponantial:")
        self.lable2.grid(row=2, column=0, columnspan=4)
        self.lablea = tk.Label(self.contrast_group, text="a:")
        self.lablea.grid(row=3, column=0)
        self.contrast_exponantial_a = tk.Spinbox(self.contrast_group, from_=0, to=9999999, increment=1, width=5)
        self.contrast_exponantial_a.grid(row=3, column=1)
        self.lableb = tk.Label(self.contrast_group, text="b:")
        self.lableb.grid(row=3, column=2)
        self.contrast_exponantial_b = tk.Spinbox(self.contrast_group, from_=0, to=9999999, increment=1, width=5)
        self.contrast_exponantial_b.grid(row=3, column=3)
        self.lable3 = tk.Label(self.contrast_group, text="logarithmic:")
        self.lable3.grid(row=4, column=0, columnspan=4)
        self.lablea = tk.Label(self.contrast_group, text="a:")
        self.lablea.grid(row=5, column=0)
        self.contrast_logarithmic_a = tk.Spinbox(self.contrast_group, from_=0, to=9999999, increment=1, width=5)
        self.contrast_logarithmic_a.grid(row=5, column=1)
        self.lableb = tk.Label(self.contrast_group, text="b:")
        self.lableb.grid(row=5, column=2)
        self.contrast_logarithmic_b = tk.Spinbox(self.contrast_group, from_=0, to=9999999, increment=1, width=5)
        self.contrast_logarithmic_b.grid(row=5, column=3)

        self.open_btn= tk.Button(self.frame2)
        self.open_btn["text"] = "open image"
        self.open_btn["command"] = self.open_image
        self.open_btn.grid(row=8, column=0)
        self.save_btn= tk.Button(self.frame2)
        self.save_btn["text"] = "save image"
        self.save_btn["command"] = self.save_image
        self.save_btn.grid(row=8, column=1)


    #Open 256-gray-level images in the format of JPG/TIF
    def open_image(self):
        self.filename = filedialog.askopenfilename(title = "Select file",filetypes = (("jpeg files","*.jpg"),("tif files","*.tif")))
        self.img = cv2.imread(self.filename)
        self.show_image()
    
    #display image
    def show_image(self):
        self.img_s = self.img
        hight, width = self.img_s.shape[:2]
        newHight = hight
        newWidth = width
        if hight/width > 1:
            newHight = int(width*IMAGE_SHOW_SIZE/hight)
            self.img_s = cv2.resize(self.img_s,(newHight,IMAGE_SHOW_SIZE))
        else:
            newWidth = int(hight*IMAGE_SHOW_SIZE/width)
            self.img_s = cv2.resize(self.img_s,(IMAGE_SHOW_SIZE,newWidth))
        self.img_s = Image.fromarray(self.img_s)
        self.img_s = ImageTk.PhotoImage(self.img_s)
        self.image_box.configure(image=self.img_s)

    #save image
    def save_image(self):
        self.filename = filedialog.asksaveasfilename(title = "Save",filetypes = (("jpeg files","*.jpg"),("tif files","*.tif")))
        cv2.imwrite(self.filename,self.img)

    def AdjustContrastLinear(self):
        a = float(self.contrast_linear_a.get())
        b = float(self.contrast_linear_b.get())
        self.img = self.img * a + b
        self.show_image()
#class policy:

def main():
    root = tk.Tk()
    app = UI(master=root)
    app.mainloop()

if __name__ == '__main__':
    main()