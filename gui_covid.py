

from tkinter import *
import tkinter.messagebox
from PIL import ImageTk, Image
import cv2
import wget
from tkinter import filedialog

from tensorflow.keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np

img_path = ""

#select image
def fileselector():
    global img_path
    main_win = tkinter.Tk() 
    main_win.withdraw()

    main_win.overrideredirect(True)
    main_win.geometry('0x0+0+0')

    main_win.deiconify()
    main_win.lift()
    main_win.focus_force()

    main_win.sourceFile = filedialog.askopenfilename(filetypes = (("Image Files",("*.jpg","*.png","*.jpeg")),("All Files","*")),parent=main_win, initialdir= "./Testing",
    title='Please select a X-Ray Image')
    main_win.destroy()
    
    img_path = main_win.sourceFile
    print(img_path)
    tkinter.messagebox.showinfo("Image Selected","Click on Detect Button. \nTo get the COVID Prediction")
    

def predict():

    if(img_path==""):
        tkinter.messagebox.showinfo("Image Not Selected","Please Select X-Ray Image \nTo get the COVID Prediction")
    else:
        print("[INFO] loading network...")
        model =load_model('./covid_pypower.h5')

        labels = ['Covid','Normal'] #These labels will be used for showing output
        start_point = (15, 15)
        end_point = (230, 80) 
        thickness = -1
        
        print("[INFO] reading image...")
        frame = cv2.imread(img_path)

        roi_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        roi_gray = cv2.resize(frame,(224,224))
        roi = roi_gray.astype('float')/255.0 
        roi = img_to_array(roi)
        roi = np.expand_dims(roi,axis=0)

        print("[INFO] classifying image...")

        preds = model.predict(roi)[0]
        #print(preds)
        #print(preds.argmax())
        label=labels[preds.argmax()]


        if(label=='Covid'):
            image = cv2.rectangle(frame, start_point, end_point, (0,0,255), thickness)
            cv2.putText(image,label,(30,60),cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255),3)
        else:
            image = cv2.rectangle(frame, start_point, end_point, (0,255,0), thickness)
            cv2.putText(image,label,(30,60),cv2.FONT_HERSHEY_SIMPLEX,1.6,(0,0,0),3)
            
        cv2.imshow('COVID Detector',frame)

        print("[INFO] saving image...")
        cv2.imwrite("./Output/detected13.jpg",frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        if(label=='Covid'):
            tkinter.messagebox.showinfo("COVID Predicted!","Take Care. Be Alert.\t \nStay Safe.")
        else:
            tkinter.messagebox.showinfo("NORMAL Report","Don't Worry! \nYour Report is Normal")


    
root = Tk()
root.title("GUI : COVID Detection")

root.geometry("880x530")

root.configure(background = 'white')
Tops = Frame(root,bg = 'blue',pady = 1, width =1750, height = 90, relief = "ridge")
Tops.grid(row=0,column=0)


Title_Label = Label(Tops,font=('Comic Sans MS',20,'bold'),text = "     PyPower  Presents  'Coronavirus Detection' \n\t   using Artificial Intelligence\t\t",pady=9,bg= 'white',fg='blue',justify ="center")
Title_Label.grid(row=0,column=0)
MainFrame = Frame(root,bg = 'white',pady=2,padx=2, width =1350, height = 100, relief = RIDGE)
MainFrame.grid(row=1,column=0)



Label_1 =Label(MainFrame, font=('lato black', 17,'bold'), text="\tCOVID Prediction using Chest X-Ray",padx=2,pady=2, bg="white",fg ="black")
Label_1.grid(row=0, column=0)

Label_2 =Label(MainFrame, font=('arial', 15,'bold'), text="",padx=2,pady=2, bg="white",fg = "black")
Label_2.grid(row=1, column=0,sticky=W)

Label_9 =Button(MainFrame, font=('arial', 19,'bold'), text="  Select X-ray Image ",padx=2,pady=2, bg="red",fg = "white",command=fileselector)
Label_9.grid(row=2, column=0)

Label_9 =Button(MainFrame, font=('arial', 19,'bold'), text="  Detect Coronavirus ",padx=2,pady=2, bg="red",fg = "white",command=predict)
Label_9.grid(row=2, column=1,sticky=W)

Label_2 =Label(MainFrame, font=('arial', 10,'bold'), text="",padx=2,pady=2, bg="white",fg = "black")
Label_2.grid(row=3, column=0,sticky=W)

Label_3 =Label(MainFrame, font=('arial', 30,'bold'), text="          \t\t\t",padx=2,pady=2, bg="white",fg = "black")
Label_3.grid(row=4, column=0)


img = cv2.imread("./Picture1.png")
img = cv2.resize(img,(420,200))
cv2.imwrite('Picture1.png',img)
img = ImageTk.PhotoImage(Image.open("Picture1.png"))
panel = Label(MainFrame, image = img).grid(row=4,column=0,sticky=E)

img1 = cv2.imread("./Picture3.png")
img1 = cv2.resize(img1,(170,170))
cv2.imwrite('Picture3.png',img1)
img1 = ImageTk.PhotoImage(Image.open("Picture3.png"))
panel = Label(MainFrame, image = img1).grid(row=4,column=1,sticky=E)

Label_3 =Label(MainFrame, font=('arial', 10,'bold'), text="\t\t\t\t          ",padx=2,pady=2, bg="white",fg = "black")
Label_3.grid(row=5, column=1)





root.mainloop()


