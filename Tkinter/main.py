import random
from tkinter import *
from PIL import ImageTk, Image, ImageDraw, ImageGrab
import PIL
import numpy as np
import joblib
from  HOG_and_Sobel import *
from scipy import ndimage
from sklearn.base import BaseEstimator, TransformerMixin

# Step 0. Load model 
pipe = joblib.load('pickle_model.pkl')

def onEnter_btn0(event):
    global img
    img = ImageTk.PhotoImage(Image.open(r'img0_hover.png'))
    b0.config(image=img)

def onEnter_btn1(event):
    global img11
    img11 = ImageTk.PhotoImage(Image.open(r'img1_hover.png'))
    b1.config(image=img11)

def onLeave_btn0(event):
    global img
    img = ImageTk.PhotoImage(Image.open(r'img0.png'))
    b0.config(image=img)

def onLeave_btn1(event):
    global img11
    img11 = ImageTk.PhotoImage(Image.open(r'img1.png'))
    b1.config(image=img11)

def center_of_mass(image):

    (w, h) = image.size
    ratio = max(w, h) / 20
    image = image.resize((int(w // ratio), int(h // ratio)), PIL.Image.ANTIALIAS)
    
    # get center_of_mass of image
    cy, cx = ndimage.measurements.center_of_mass(np.array(image.convert('L')))
    
    IMG_SIZE = 28
    #create blank image and paste image in
    base_image = Image.new("L", (IMG_SIZE, IMG_SIZE), 0)
    base_image.paste(image.convert('L'), (int(IMG_SIZE / 2 - cx), int(IMG_SIZE / 2 - cy)))
    cy, cx = ndimage.measurements.center_of_mass(np.array(base_image))
    
    base_image.save('image_for_predict.png')

    return base_image

def predict():
    deleteText()
    createText("","   PLEASE WAIT...  ")

    # Step 1. crop and transform
    bbox = image1.getbbox()
    cropped = image1.crop(bbox)
    filename = "image.png"

    cropped.save(filename)

    image = PIL.Image.open("image.png")

    image_predict = center_of_mass(image)

    # Step 2. predict
    data = np.asarray(image_predict)
    data = data.flatten()

    y_pred = pipe.predict([data, ])
    
    deleteText()
    createText(y_pred[0])

def btn_clicked():
    print("Button Clicked")

def createText(x, pre_text = "THIS IS NUMBER: "):
    canvas.create_text(
        709.0, 424.0,
        text=f"{pre_text}{x}",
        fill="#d60000",
        font=("Roboto-Bold", int(15.0)),
        tag = "predict_text")

def deleteText():
    canvas.delete("predict_text")
def clear_frame():
    cv.delete('all')
    deleteText()
    createText("","READY")
    draw.rectangle((0, 0, 435, 308), fill="black")

def get_x_and_y(event):
    global lasx, lasy
    lasx, lasy = event.x, event.y


def draw_smth(event):
    global lasx, lasy
    cv.create_line((lasx, lasy, event.x, event.y), fill='white', width=4)
    draw.line([lasx, lasy, event.x, event.y], fill='white', width=4)
    lasx, lasy = event.x, event.y

window = Tk()
window.title("Number Predict")
window.geometry("930x449")
window.configure(bg = "#ffffff")
canvas = Canvas(
    window,
    bg = "#ffffff",
    height = 449,
    width = 930,
    bd = 0,
    highlightthickness = 0,
    relief = "ridge")
canvas.place(x = 0, y = 0)

img0 = PhotoImage(file = f"img0.png")
b0 = Button(
    image = img0,
    borderwidth = 0,
    highlightthickness = 0,
    command = clear_frame,
    relief = "flat")

b0.place(
    x = 714, y = 339,
    width = 200,
    height = 50)

img1 = PhotoImage(file = f"img1.png")
b1 = Button(
    image = img1,
    borderwidth = 0,
    highlightthickness = 0,
    command = predict,
    relief = "flat")

b1.place(
    x = 488, y = 339,
    width = 200,
    height = 50)

b0.bind('<Enter>',  onEnter_btn0)
b0.bind('<Leave>',  onLeave_btn0)

b1.bind('<Enter>',  onEnter_btn1)
b1.bind('<Leave>',  onLeave_btn1)

# canvas.create_rectangle(
#     483, 22, 483+435, 22+308,
#     fill = "#000000",
#     outline = "")
###
cv = Canvas(
    window,
    bg = "black",
    height = 308,
    width = 435,
    bd = 0,
    highlightthickness = 0,
    relief = "ridge")
cv.place(x = 483, y = 22)

image1 = PIL.Image.new("RGB", (435, 308), (0, 0, 0))
draw = ImageDraw.Draw(image1)

cv.bind("<Button-1>", get_x_and_y)
cv.bind("<B1-Motion>", draw_smth)
###
# createText('')
# canvas.create_text(
#     709.0, 424.0,
#     text = "THIS IS NUMBER:",
#     fill = "#d60000",
#     font = ("Roboto-Bold", int(15.0)))

background_img = PhotoImage(file = f"background.png")
background = canvas.create_image(
    235.5, 224.5,
    image=background_img)

window.resizable(False, False)
window.mainloop()
