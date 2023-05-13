from keras.models import load_model
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import pandas as pd
import numpy as np
import cv2

model = load_model("traffic_classifier_MyModel.h5")

df = pd.read_csv("data/Meta.csv")
classes = df["ClassLabel"]


def preprocess(img):
    img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255
    return img


def classify(file_path):
    img = preprocess(np.array(Image.open(file_path).resize((30, 30))))
    y_pred = model.predict(img.reshape(1, 30, 30, 1))
    predicted_class = np.argmax(y_pred)
    sign_text = classes[predicted_class]
    label.configure(text=f"[{predicted_class}] {sign_text}")


def enable_classify_button(file_path):
    classify_button.configure(command=lambda: classify(file_path), state="normal")


def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        img = Image.open(file_path)

        width, height = img.size
        larger_dimension = max(width, height)
        scale_factor = 200 / larger_dimension

        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        resized_image = img.resize((new_width, new_height))

        tk_img = ImageTk.PhotoImage(resized_image)
        sign_image.configure(image=tk_img)
        sign_image.image = tk_img
        label.configure(text="")
        enable_classify_button(file_path)
    except:
        pass


window = Tk()
window.geometry("525x430")
window.title("Traffic sign classification")
window.configure(bg="#EFF5F5")

label = Label(window, font=("Cartograph CF", 10), bg="#FFFFFF", fg="#497174")
sign_image = Label(window, bg="#FFFFFF")

upload_img_btn = Button(window, text="Upload an image", font=("Cartograph CF", 10), bg="#497174", fg="#FFFFFF",
                        borderwidth=2, relief="groove", command=upload_image)
classify_button = Button(window, text="Classify Image", font=("Cartograph CF", 10), bg="#497174", fg="#FFFFFF",
                         borderwidth=2, relief="groove")
classify_button.configure(state="disabled")

label.place(x=50, y=340, height=40, width=425)
sign_image.place(x=50, y=115, height=200, width=425)
upload_img_btn.place(x=50, y=50, height=40, width=200)
classify_button.place(x=275, y=50, height=40, width=200)
window.mainloop()
