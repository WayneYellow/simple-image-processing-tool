"""
This is a simple image processing program with a simple GUI.
B113040040 黃楚文
Date: 2023/10/02
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk

class UI(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack()
        self.create_widgets()
        self.master.geometry("500x500")
        self.master.title("Image Processing")


    def create_widgets(self):
        self.open_btn= tk.Button(self)
        self.open_btn["text"] = "open image"
        self.open_btn["command"] = self.open_image
        self.open_btn.pack(side="top")

        self.quit = tk.Button(self, text="QUIT", fg="red", command=self.master.destroy)
        self.quit.pack(side="bottom")

    #Open/save/display 256-gray-level images in the format of JPG/TIF
    def open_image(self):
        #use tkinter file dialog
        filename = tk.filedialog.askopenfilename()
        self.img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        #show image in tk window
        cv2.imshow("image", self.img)
        cv2.waitKey(0)


        
        #

#class policy:

def main():
    root = tk.Tk()
    app = UI(master=root)
    app.mainloop()

if __name__ == '__main__':
    main()