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

        self.spin_var1 = tk.StringVar()
        self.spin_var2 = tk.StringVar()
        self.spin_var3 = tk.StringVar()
        self.spin_var4 = tk.StringVar()
        self.spin_var5 = tk.StringVar()
        self.spin_var6 = tk.StringVar()
        self.contrast_group = tk.LabelFrame(self.frame2, text="Contrast/Brightness", padx=5, pady=5)
        self.contrast_group.grid(row=0, column=0, columnspan=2)
        self.lable1 = tk.Label(self.contrast_group, text="linear:")
        self.lable1.grid(row=0, column=0, columnspan=4)
        self.lablea = tk.Label(self.contrast_group, text="a:")
        self.lablea.grid(row=1, column=0)
        self.contrast_linear_a = tk.Spinbox(self.contrast_group, from_=-127, to=127, increment=1, width=5, command=self.adjust_contrast_linear, textvariable=self.spin_var1)
        self.contrast_linear_a.grid(row=1, column=1)
        self.spin_var1.set("1")
        self.lableb = tk.Label(self.contrast_group, text="b:")
        self.lableb.grid(row=1, column=2)
        self.contrast_linear_b = tk.Spinbox(self.contrast_group, from_=0, to=9999999, increment=1, width=5, command=self.adjust_contrast_linear, textvariable=self.spin_var2)
        self.contrast_linear_b.grid(row=1, column=3)
        self.lable2 = tk.Label(self.contrast_group, text="exponantial:")
        self.lable2.grid(row=2, column=0, columnspan=4)
        self.lablea = tk.Label(self.contrast_group, text="a:")
        self.lablea.grid(row=3, column=0)
        self.contrast_exponantial_a = tk.Spinbox(self.contrast_group, from_=0, to=1, increment=0.01, width=5,textvariable=self.spin_var3, command=self.adjust_contrast_exponantial)
        self.contrast_exponantial_a.grid(row=3, column=1)
        self.spin_var3.set("0.04")
        self.lableb = tk.Label(self.contrast_group, text="b:")
        self.lableb.grid(row=3, column=2)
        self.contrast_exponantial_b = tk.Spinbox(self.contrast_group, from_=-1, to=1, increment=0.01, width=5,textvariable=self.spin_var4, command=self.adjust_contrast_exponantial)
        self.spin_var4.set("0")
        self.contrast_exponantial_b.grid(row=3, column=3)
        self.lable3 = tk.Label(self.contrast_group, text="logarithmic:")
        self.lable3.grid(row=4, column=0, columnspan=4)
        self.lablea = tk.Label(self.contrast_group, text="a:")
        self.lablea.grid(row=5, column=0)
        self.contrast_logarithmic_a = tk.Spinbox(self.contrast_group, from_=0, to=1, increment=0.01, width=5, textvariable=self.spin_var5, command=self.adjust_contrast_logarithmic)
        self.contrast_logarithmic_a.grid(row=5, column=1)
        self.spin_var5.set("0.1")
        self.lableb = tk.Label(self.contrast_group, text="b:")
        self.lableb.grid(row=5, column=2)
        self.contrast_logarithmic_b = tk.Spinbox(self.contrast_group, from_=1, to=300, increment=1, width=5, textvariable=self.spin_var6, command=self.adjust_contrast_logarithmic)
        self.contrast_logarithmic_b.grid(row=5, column=3)
        self.spin_var6.set("1")

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
        file_path = filedialog.askopenfilename(title = "Select file",filetypes = (("jpeg files","*.jpg"),("tif files","*.tif")))
        if file_path:
            self.image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            self.original_img = np.copy(self.image)
            self.display_image(self.image)
    
    # display image
    def show_image(self):
        hight, width = self.img.shape[:2]
        if hight/width > 1:
            hight = int(width*IMAGE_SHOW_SIZE/hight)
            self.img = cv2.resize(self.img,(hight,IMAGE_SHOW_SIZE))
        else:
            width = int(hight*IMAGE_SHOW_SIZE/width)
            self.img = cv2.resize(self.img,(IMAGE_SHOW_SIZE,width))
        self.img = Image.fromarray(self.img)
        self.img = ImageTk.PhotoImage(self.img)
        self.image_box.configure(image=self.img)

    def display_image(self, image):
        plt.imshow(image, cmap='gray')
        plt.axis('off')
        plt.show()

    def show_full_image(self):
        plt.imshow(self.image, cmap='gray')
        plt.axis('off')
        plt.show()
    #save image
    def save_image(self):
        if self.img is not None:
            filepath = filedialog.asksaveasfilename(title = "Save", filetypes = (("jpeg files","*.jpg"),("tif files","*.tif")))
            if filepath:
                cv2.imwrite(filepath,self.img)

    def adjust_contrast_linear(self):
        self.img = np.copy(self.original_img)
        a = int(self.contrast_linear_a.get())
        b = int(self.contrast_linear_b.get())
        self.img = cv2.convertScaleAbs(self.img, alpha=a, beta=b)
        self.display_image(self.img)

    #adjust contrast with a for-loop function exponantially
    def adjust_contrast_exponantial(self):
        self.img = np.copy(self.original_img)
        a = float(self.contrast_exponantial_a.get())
        b = float(self.contrast_exponantial_b.get())
        if a == 0:
            a = 0.001
        self.img = np.exp(a*(self.img+1)+b)
        min_val = np.min(self.img)
        max_val = np.max(self.img)
        self.img = 255 * (self.img - min_val) / (max_val - min_val)
        self.img = self.img.astype(np.uint8)
        self.show_image()

    def adjust_contrast_logarithmic(self):
        self.img = np.copy(self.original_img)
        a = float(self.contrast_logarithmic_a.get())
        b = float(self.contrast_logarithmic_b.get())
        self.img = np.log1p((self.img+1)*a+b)
        #scale the image to 0~255
        min_val = np.min(self.img)
        max_val = np.max(self.img)
        self.img = 255 * (self.img - min_val) / (max_val - min_val)
        self.img = self.img.astype(np.uint8)
        self.show_image()

    def correct_gray(self):
        if len(self.img.shape) < 3 or self.img.shape[2]  == 1:
            self.img = self.img
        else:
            self.img = self.img[:,:,0]

    #Zoom in and shrink with respect to the images' original size by using bilinear interpolation
    def zoom_image(self):
        if self.image is not None:
            zoom_factor = simpledialog.askfloat("Input", "Enter zoom factor:")
            height, width = self.image.shape
            new_height, new_width = int(height * zoom_factor), int(width * zoom_factor)
            resized_image = cv2.resize(self.image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        self.show_image()
#class policy:
    

if __name__ == '__main__':
    root = tk.Tk()
    app = UI(master=root)
    app.mainloop()