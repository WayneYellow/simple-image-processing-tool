import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import *
from tkinter import filedialog
from tkinter import simpledialog
from tkinter import messagebox

class ImageProcessor:
    def __init__(self, master):
        self.master = master
        self.master.title("Image Processor")

        self.image = None

        self.create_widgets()

    def create_widgets(self):
        # Buttons
        self.btn_open = Button(self.master, text="Open Image", command=self.open_image)
        self.btn_open.pack(pady=10)

        self.btn_save = Button(self.master, text="Save Image", command=self.save_image)
        self.btn_save.pack(pady=10)

        self.btn_contrast = Button(self.master, text="Adjust Contrast", command=self.adjust_contrast)
        self.btn_contrast.pack(pady=10)

        self.btn_zoom = Button(self.master, text="Zoom", command=self.zoom_image)
        self.btn_zoom.pack(pady=10)

        self.btn_rotate = Button(self.master, text="Rotate", command=self.rotate_image)
        self.btn_rotate.pack(pady=10)

        self.btn_gray_slice = Button(self.master, text="Gray-level Slice", command=self.gray_level_slice)
        self.btn_gray_slice.pack(pady=10)

        self.btn_histogram = Button(self.master, text="Display Histogram", command=self.display_histogram)
        self.btn_histogram.pack(pady=10)

        self.btn_bit_plane = Button(self.master, text="Bit-Plane Images", command=self.bit_plane_images)
        self.btn_bit_plane.pack(pady=10)

        self.btn_smooth_sharpen = Button(self.master, text="Smoothing/Sharpening", command=self.smooth_sharpen)
        self.btn_smooth_sharpen.pack(pady=10)

    def open_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.tif")])
        if file_path:
            self.image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            self.display_image(self.image)

    def save_image(self):
        if self.image is not None:
            file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
            if file_path:
                cv2.imwrite(file_path, self.image)

    def display_image(self, image):
        plt.imshow(image, cmap='gray')
        plt.axis('off')
        plt.show()

    def adjust_contrast(self):
        if self.image is not None:
            a = simpledialog.askfloat("Input", "Enter 'a' value:")
            b = simpledialog.askfloat("Input", "Enter 'b' value:")
            
            # Apply linear, exponential, and logarithmic transformations
            linear_transform = a * self.image + b
            exponential_transform = np.exp(a * self.image + b)
            logarithmic_transform = np.log1p(a * self.image + b)

            # Display transformed images
            self.display_image(linear_transform)
            self.display_image(exponential_transform)
            self.display_image(logarithmic_transform)

    def zoom_image(self):
        if self.image is not None:
            zoom_factor = simpledialog.askfloat("Input", "Enter zoom factor:")
            height, width = self.image.shape
            new_height, new_width = int(height * zoom_factor), int(width * zoom_factor)
            resized_image = cv2.resize(self.image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            self.display_image(resized_image)

    def rotate_image(self):
        if self.image is not None:
            angle = simpledialog.askfloat("Input", "Enter rotation angle:")
            height, width = self.image.shape
            rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
            rotated_image = cv2.warpAffine(self.image, rotation_matrix, (width, height))
            self.display_image(rotated_image)

    def gray_level_slice(self):
        if self.image is not None:
            lower_limit = simpledialog.askinteger("Input", "Enter lower gray level:")
            upper_limit = simpledialog.askinteger("Input", "Enter upper gray level:")
            preserve_original = messagebox.askyesno("Input", "Preserve original values in unselected areas?")

            sliced_image = np.copy(self.image)
            sliced_image[(sliced_image < lower_limit) | (sliced_image > upper_limit)] = 0 if preserve_original else 255
            self.display_image(sliced_image)

    def display_histogram(self):
        if self.image is not None:
            plt.hist(self.image.flatten(), bins=256, range=[0, 256], color='gray', alpha=0.7)
            plt.title("Image Histogram")
            plt.show()

    def bit_plane_images(self):
        if self.image is not None:
            bit_plane = simpledialog.askinteger("Input", "Enter bit-plane number (0-7):")
            bit_plane_image = (self.image >> bit_plane) & 1
            self.display_image(bit_plane_image * 255)

    def smooth_sharpen(self):
        if self.image is not None:
            kernel_size = simpledialog.askinteger("Input", "Enter kernel size:")
            smoothing_type = messagebox.askyesno("Input", "Do you want to apply smoothing?")

            if smoothing_type:
                smoothed_image = cv2.GaussianBlur(self.image, (kernel_size, kernel_size), 0)
                self.display_image(smoothed_image)
            else:
                sharpening_strength = simpledialog.askfloat("Input", "Enter sharpening strength:")
                sharpening_kernel = np.array([[-1, -1, -1],
                                              [-1, 1 + sharpening_strength, -1],
                                              [-1, -1, -1]])
                sharpened_image = cv2.filter2D(self.image, -1, sharpening_kernel)
                self.display_image(sharpened_image)

if __name__ == "__main__":
    root = Tk()
    app = ImageProcessor(root)
    root.mainloop()
