from tkinter import *
import csv
from sklearn import neighbors
import numpy as np
import pygame
from PIL import Image
from resizeimage import resizeimage as ri


X = []
Y = []
clf = neighbors.KNeighborsClassifier()
result = int()
test = []
class GUI:
    def __init__(self, master):
        self.master = master
        master.title("System GUI")
        
        self.output = Text(master)
        self.output.pack()

        self.label = Label(master, text="Hand Written Digit Recognization System")
        self.label.pack()

        self.greet_button = Button(master, text="Train Machine", command=self.training)
        self.greet_button.pack()
        
        self.greet_button = Button(master, text="Draw", command=self.digitdraw)
        self.greet_button.pack()
        
        self.greet_button = Button(master, text="Predict", command=self.final)
        self.greet_button.pack()
        
        self.close_button = Button(master, text="Close", command=master.quit)
        self.close_button.pack()
        
        
        
       
        
    def Train(self):
        global X
        global Y
        self.output.insert(END,"Opening Training Data")
        hold = open("Final_train.csv", newline='')
        reader = csv.reader(hold)
        self.output.insert(END,"Part 1 Learning......")
        for i in reader:
            data = int(i[784])
            Y.append(data)
            temp = i[0:784]
            new = [int(j) for j in temp]
            X.append(new)
        X = np.array(X)    
        Y = np.array(Y)  
        hold.close()

    def fitting(self):
        global clf
        global X
        global Y
        self.output.insert(END,"Part 2 Learning......")
        clf = clf.fit(X, Y)
        
    def testdata(self):
        name = "output.png"
        hold = open("newdataset.csv", "w")
        writer = csv.writer(hold)

        img = Image.open(name)
        arr = np.array(img)
        newarr = []
        new =[]
        for i in arr:            
            for j in i:
                c = (sum(j)//3)
                newarr.append(c)
            new.append(newarr)
            newarr = []
        newarr = np.array(new)       
        newarr = newarr.flatten()
        for i in range(0, newarr.size):
            if newarr[i]<=50:
                newarr[i]=0
            else:
                newarr[i]=1
                
        
        writer.writerow(newarr)

    def testing(self):
        global result
        global test
        global clf
        hold2 = open("newdataset.csv", newline='')
        reader2 = csv.reader(hold2)

        self.output.insert(END,"Testing.....")
        for i in reader2:
            temp = i[0: ]
            new = [int(j) for j in temp]
            test.append(new)
 
            #result.append(int(clf.predict(i)))                     #For list of test data
            result = clf.predict(test)
    
        hold2.close()
        
    def out(self):
        global result
        self.output.insert(END,result[-1])
        self.output.insert(END,"Test Completed")      
       

    def digitdraw(self):
        pygame.init()
        screen = pygame.display.set_mode([480, 480])
        pygame.display.set_caption('click to draw')

        keep_going = True
        WHITE = (255,  255,  255)
        radius = 27
        mousedown = False

        while keep_going:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    keep_going = False
                if event.type == pygame.MOUSEBUTTONDOWN:
                    mousedown = True
                if event.type == pygame.MOUSEBUTTONUP:
                    mousedown = False
                if mousedown:
                    spot = pygame.mouse.get_pos()
                    pygame.draw.circle(screen,  WHITE,  spot,  radius)
            
                pygame.display.update()
        pygame.image.save(screen, "screenshot.png")
        imagefile = "screenshot.png"
        iml = Image.open(imagefile)
        im2 = ri.resize_cover(iml, [28, 28])
        im2.save("output.png", "PNG")         
            
        pygame.quit()
    
    def final(self):
        self.testdata()
        self.testing()
        self.out()
        
    def training(self):
        self.Train()
        self.fitting()
        self.output.insert(END,"Trained")
        
        
root = Tk()
my_gui = GUI(root)
root.mainloop()
