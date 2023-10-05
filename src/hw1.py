"""
This is a simple image processing program with a simple GUI.
B113040040 黃楚文
Date: 2023/10/02
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
from PIL import Image, ImageTk

class ImageProcess(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack()
        self.original_img = None
        self.image = None
        self.create_widgets()
        self.master.geometry("1000x600")
        self.master.title("Image Processing")

    def create_widgets(self):
        self.frame1 = tk.Frame(self.master)
        self.frame1.pack()
        self.frame = tk.Frame(self.master, width=900, height=500)
        self.frame.pack()

        self.canvas = tk.Canvas(self.frame, width=900, height=500, bg='#fff')

        self.scrollX = tk.Scrollbar(self.frame, orient='horizontal')
        self.scrollX.pack(side='bottom', fill='x')
        self.scrollX.config(command=self.canvas.xview)

        self.scrollY = tk.Scrollbar(self.frame, orient='vertical')
        self.scrollY.pack(side='right', fill='y')
        self.scrollY.config(command=self.canvas.yview)

        self.canvas.config(xscrollcommand=self.scrollX.set, yscrollcommand=self.scrollY.set)
        self.canvas.pack(side='left')

        self.spin_var1 = tk.StringVar()
        self.spin_var2 = tk.StringVar()
        self.contrast_group = tk.LabelFrame(self.frame1, text="Contrast/Brightness", padx=5, pady=5)
        self.contrast_group.grid(row=1, column=0)
        self.lablea = tk.Label(self.contrast_group, text="a:")
        self.lablea.grid(row=0, column=0)
        self.contrast_a = tk.Spinbox(self.contrast_group, from_=-500, to=500, increment=1, width=5, textvariable=self.spin_var1)
        self.contrast_a.grid(row=0, column=1)
        self.spin_var1.set("1")
        self.lableb = tk.Label(self.contrast_group, text="b:")
        self.lableb.grid(row=0, column=2)
        self.contrast_b = tk.Spinbox(self.contrast_group, from_=-500, to=500, increment=1, width=5, textvariable=self.spin_var2)
        self.contrast_b.grid(row=0, column=3)
        self.spin_var2.set("1")
        self.linear_btn = tk.Button(self.contrast_group, text="linear", command=self.adjust_contrast_linear)
        self.linear_btn.grid(row=1, column=0)
        self.exponantial_btn = tk.Button(self.contrast_group, text="exponantial", command=self.adjust_contrast_exponantial)
        self.exponantial_btn.grid(row=1, column=1)
        self.logarithmic_btn = tk.Button(self.contrast_group, text="logarithmic", command=self.adjust_contrast_logarithmic)
        self.logarithmic_btn.grid(row=1, column=2)
        
        self.zoom_group = tk.LabelFrame(self.frame1, text="Zoom image", padx=5, pady=5)
        self.zoom_group.grid(row=1, column=1)
        self.lablefactor = tk.Label(self.zoom_group, text="factor:")
        self.lablefactor.grid(row=0, column=0)
        self.spin_var_factor = tk.StringVar()
        self.zoom_factor = tk.Spinbox(self.zoom_group, from_=0, to=10, increment=1, width=5, textvariable=self.spin_var_factor)
        self.spin_var_factor.set("1")
        self.zoom_factor.grid(row=0, column=1)
        self.zoom_btn = tk.Button(self.zoom_group, text="zoom", command=self.zoom_image)
        self.zoom_btn.grid(row=1, column=0)

        self.rotate_group = tk.LabelFrame(self.frame1, text="Rotate image", padx=5, pady=5)
        self.rotate_group.grid(row=1, column=2)
        self.lableangle = tk.Label(self.rotate_group, text="angle:")
        self.lableangle.grid(row=0, column=0)
        self.spin_var_angle = tk.StringVar()
        self.rotate_angle = tk.Spinbox(self.rotate_group, from_=-360, to=360, increment=1, width=5, textvariable=self.spin_var_angle)
        self.spin_var_angle.set("0")
        self.rotate_angle.grid(row=0, column=1)
        self.rotate_btn = tk.Button(self.rotate_group, text="rotate", command=self.rotate_image)
        self.rotate_btn.grid(row=1, column=0)

        self.gray_level_slice_group = tk.LabelFrame(self.frame1, text="Gray-level Slice", padx=5, pady=5)
        self.gray_level_slice_group.grid(row=1, column=3)
        self.lablelower = tk.Label(self.gray_level_slice_group, text="lower limit:", padx=5, pady=5)
        self.lablelower.grid(row=0, column=0)
        self.lower_limit = tk.Spinbox(self.gray_level_slice_group, from_=0, to=255, increment=1, width=5)
        self.lower_limit.grid(row=0, column=1)
        self.lableupper = tk.Label(self.gray_level_slice_group, text="upper limit:", padx=5, pady=5)
        self.lableupper.grid(row=0, column=2)
        self.upper_limit = tk.Spinbox(self.gray_level_slice_group, from_=0, to=255, increment=1, width=5)
        self.upper_limit.grid(row=0, column=3)
        self.gray_level_slice_btn = tk.Button(self.gray_level_slice_group, text="gray-level slice", command=self.gray_level_slice)
        self.gray_level_slice_btn.grid(row=1, column=0)

        self.bit_plane_images_btn = tk.Button(self.frame1, text="bit-plane images", command=self.bit_plane_images)
        self.bit_plane_images_btn.grid(row=2, column=0)
        self.histogram_btn = tk.Button(self.frame1, text="display histogram", command=self.display_histogram)
        self.histogram_btn.grid(row=2, column=1)
        self.auto_level_btn = tk.Button(self.frame1, text="auto level", command=self.auto_level)
        self.auto_level_btn.grid(row=2, column=2)
        self.smooth_btn = tk.Button(self.frame1, text="smoothing", command=self.smooth)
        self.smooth_btn.grid(row=2, column=3)
        self.sharpen_btn = tk.Button(self.frame1, text="sharpening", command=self.sharpen)
        self.sharpen_btn.grid(row=2, column=4)
        self.open_btn= tk.Button(self.frame1, text="open image", command=self.open_image)
        self.open_btn.grid(row=0, column=0)
        self.save_btn= tk.Button(self.frame1, text="save image", command=self.save_image)
        self.save_btn.grid(row=0, column=1)
        self.reset_btn = tk.Button(self.frame1, text="reset", command=self.reset)
        self.reset_btn.grid(row=0, column=2)
        

    #Open 256-gray-level images in the format of JPG/TIF
    def open_image(self):
        file_path = filedialog.askopenfilename(title = "Select file",filetypes = (("jpeg files","*.jpg"),("tif files","*.tif")))
        if file_path:
            self.original_img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            self.display_image(self.original_img)
        self.filepath = None

    def display_image(self, img):
        if img is not None:
            im = Image.fromarray(img)
            w, h = im.size
            tk_img = ImageTk.PhotoImage(im)     
            self.canvas.delete('all')               
            self.canvas.config(scrollregion=(0,0,w,h))   
            self.canvas.create_image(0, 0, anchor='nw', image=tk_img) 
            self.canvas.tk_img = tk_img

    #save image
    def save_image(self):
        if self.image is not None:
            if self.filepath is None:
                self.filepath = filedialog.asksaveasfilename(title = "Save", filetypes = (("jpeg files","*.jpg"),("tif files","*.tif")))
            if self.filepath:
                cv2.imwrite(self.filepath,self.image)
            self.original_img = self.image.copy()

    def reset(self):
        if self.original_img is not None:
            self.image = self.original_img.copy()
            self.display_image(self.image)

    def adjust_contrast_linear(self):
        if self.original_img is not None:
            img = self.original_img.copy()
            a = float(self.spin_var1.get())
            b = float(self.spin_var2.get())
            self.image = cv2.convertScaleAbs(img, alpha=a, beta=b).copy()
            self.display_image(self.image)

    #adjust contrast with a for-loop function exponantially
    def adjust_contrast_exponantial(self):
        if self.original_img is not None:
            img = self.original_img.copy()
            a = float(self.contrast_a.get())
            b = float(self.contrast_b.get())
            if a == 0:
                a = 0.001
            img = np.exp(a*(img+1)+b)
            min_val = np.min(img)
            max_val = np.max(img)
            img = 255 * (img - min_val) / (max_val - min_val)
            img = img.astype(np.uint8)
            self.image = img.copy()
            self.display_image(self.image)

    def adjust_contrast_logarithmic(self):
        if self.original_img is not None:
            img = self.original_img.copy()
            a = float(self.contrast_a.get())
            b = float(self.contrast_b.get())
            img = np.log1p((img+1)*a+b)
            #scale the image to 0~255
            min_val = np.min(img)
            max_val = np.max(img)
            img = 255 * (img - min_val) / (max_val - min_val)
            img = img.astype(np.uint8)
            self.image = img.copy()
            self.display_image(self.image)

    def zoom_image(self):
        if self.original_img is not None:
            img = self.original_img.copy()
            zoom_factor = float(self.zoom_factor.get())
            height, width = img.shape[:2]
            new_height, new_width = int(height * zoom_factor), int(width * zoom_factor)
            resized_image = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            self.image = resized_image
            self.display_image(self.image)

    def rotate_image(self):
        if self.original_img is not None:
            img = self.original_img.copy()
            angle = float(self.rotate_angle.get())
            height, width = img.shape[:2]
            rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
            rotated_image = cv2.warpAffine(img, rotation_matrix, (width, height))
            self.image = rotated_image
            self.display_image(self.image)

    def gray_level_slice(self):
        if self.original_img is not None:
            lower_limit = int(self.lower_limit.get())
            upper_limit = int(self.upper_limit.get())
            preserve_original = messagebox.askyesno("Input", "Preserve original values in unselected areas?")
            row, column = self.original_img.shape[:2]
            img = np.copy(self.original_img)
            sliced_image = np.zeros((row,column),dtype = 'uint8')
            for i in range(row):
                for j in range(column):
                    if img[i,j]>lower_limit and img[i,j]<upper_limit: 
                        sliced_image[i,j] = 255
                    else: 
                        if preserve_original:
                            sliced_image[i,j] = img[i-1,j-1] 
                        else:
                            sliced_image[i,j] = 0
            self.image = sliced_image
            self.display_image(sliced_image)

    def display_histogram(self):
        if self.original_img is not None:
            img = self.original_img.copy()
            plt.hist(img.flatten(), bins=256, range=[0, 256], color='gray', alpha=0.7)
            plt.title("Image Histogram")
            plt.show()
    
    def bit_plane_images(self):
        if self.original_img is not None:
            img = self.original_img.copy()
            bit_plane = simpledialog.askinteger("Input", "Enter bit-plane number (0-7):")
            bit_plane_image = (img >> bit_plane) & 1
            self.display_image(bit_plane_image * 255)

    def auto_level(self):
        if self.original_img is not None:
            img = self.original_img.copy()
            equalized_img = cv2.equalizeHist(img)

            plt.subplot(1, 2, 2)
            plt.imshow(equalized_img, cmap='gray')
            plt.title('Equalized Image')

            plt.figure(figsize=(12, 4))
            plt.subplot(1, 2, 1)
            plt.hist(img.flatten(), bins=256, range=[0, 256], color='r')
            plt.title('Original Image Histogram')

            plt.subplot(1, 2, 2)
            plt.hist(equalized_img.flatten(), bins=256, range=[0, 256], color='b')
            plt.title('Equalized Image Histogram')

            plt.show()

    def smooth(self):
        if self.original_img is not None:
            img = self.original_img.copy()
            kernel_size = simpledialog.askinteger("Input", "Enter kernel size:")
            if kernel_size is not None:
                smoothed_image = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
                self.image = smoothed_image
                self.display_image(self.image)

    def sharpen(self):
        if self.original_img is not None:
            img = self.original_img.copy()
            sharpening_strength = simpledialog.askfloat("Input", "Enter sharpening strength:")
            if sharpening_strength is not None:
                sharpening_kernel = np.array([[ 0, -1, 0],
                                            [-1, 5 + sharpening_strength, -1],
                                            [0, -1, 0]])
                sharpened_image = cv2.filter2D(img, -1, sharpening_kernel)
                self.image = sharpened_image
                self.display_image(self.image)

if __name__ == '__main__':
    root = tk.Tk()
    app = ImageProcess(master=root)
    app.mainloop()