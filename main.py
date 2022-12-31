#!/usr/bin/python3

#importing libraries
import io
import picamera
import cv2
import numpy
import time
from PIL import Image, ImageDraw, ImageSequence
from google.cloud import vision
from demo_opts import get_device
from luma.core.sprite_system import framerate_regulator

device = get_device()
client = vision.ImageAnnotatorClient()

#getting the harrcascade file
faceCascade = cv2.CascadeClassifier("resources/haarcascade_frontalface_default.xml")

#blink function
def blink():
    regulator = framerate_regulator(fps=15)
    source = Image.open('resources/fastblink.gif')
    size = device.size

    for frame in ImageSequence.Iterator(source):
        with regulator:
            background = Image.new("RGB", device.size, "white")
            background.paste(frame)
            device.display(background.convert(device.mode))


def wakeup():
    print("Waking Yuki up")
    image = Image.open('resources/wakeup0.png')
    size = device.size
    regulator = framerate_regulator(fps=7)
    background = Image.new("RGB",device.size,"white")
    image2 = Image.open('resources/wakeup1.gif')
    image3 = Image.open('resources/wakeup2.gif')

    # the eye line thing
    background.paste(image)
    device.display(background.convert(device.mode))
    time.sleep(1.5)

    # the slightly opened part
    for frame in ImageSequence.Iterator(image2):
            with regulator:
                    background = Image.new("RGB", device.size, "white")
                    background.paste(frame)
                    device.display(background.convert(device.mode))

    time.sleep(2.5)

    # the final part, opening the eyes fully
    regulator = framerate_regulator(fps=20)
    for frame in ImageSequence.Iterator(image3):
        with regulator:
            background = Image.new("RGB", device.size, "white")
            background.paste(frame)
            device.display(background.convert(device.mode))
    time.sleep(2)

# function to show joy
def showjoy():
    image = Image.open('resources/happy.gif')
    size = device.size

    regulator = framerate_regulator(fps=10)
    
    for frame in ImageSequence.Iterator(image):
        with regulator:
            background = Image.new("RGB", device.size, "white")
            background.paste(frame)
            device.display(background.convert(device.mode))



#function to avoid emotion
def showavoid():
    image = Image.open('resources/avoid.gif')
    size = device.size
    regulator = framerate_regulator(fps=10)
    for frame in ImageSequence.Iterator(image):
            with regulator:
                    background = Image.new("RGB", device.size, "white")
                    background.paste(frame)
                    device.display(background.convert(device.mode))

#function for reverting to nuetral emotion
def showneutral():
    image = Image.open('resources/neutral.png')
    size= device.size
    background = Image.new("RGB",device.size,"white")
    background.paste(image)
    device.display(background.convert(device.mode))

def sleep():
    image0 = Image.open('resources/sleep.gif')
    image1 = Image.open('resources/wakeup0.png')
    size=device.size
    regulator = framerate_regulator(fps=10)
    for frame in ImageSequence.Iterator(image0):
            with regulator:
                    background = Image.new("RGB", device.size, "white")
                    background.paste(frame)
                    device.display(background.convert(device.mode))


    background.paste(image1)
    device.display(background.convert(device.mode))
    time.sleep(2)
    print("Yuki wishes you a good sleep!")


#function for taking a picture
def snap():
    with picamera.PiCamera() as camera:
            camera.resolution = (640,480)
            camera.capture(stream, format='jpeg')
            print("click")

wakeup()
time.sleep(1)
blink()
#an output that signals the program start (replacement for the led)
#need to add a stop case
print("Starting main loop")

while 1:
        stream = io.BytesIO()
        snap()
        buff = numpy.frombuffer(stream.getvalue(), dtype=numpy.uint8)
        img = cv2.imdecode(buff, 1)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.2, 5)
        key = cv2.waitKey(1) & 0xFF
        if len(faces)>0:
                print((faces[0][2]/2)+faces[0][0]) #abcissa of midpoint
                print((faces[0][1]/2)+faces[0][3]) #ordinate of midpoint
                cv2.imwrite('image.jpg',img) #saving the image for the API
                with open('image.jpg', 'rb') as image_file: #emotion detection of the API
                        content = image_file.read()
                image = vision.types.Image(content=content)
                response = client.face_detection(image=image)
                faces = response.face_annotations
                likelihood_name = ('UNKNOWN', 'VERY_UNLIKELY', 'UNLIKELY', 'POSSIBLE','LIKELY', 'VERY_LIKELY')
                for face in faces:
                        if likelihood_name[face.anger_likelihood] == 'LIKELY' or likelihood_name[face.anger_likelihood] == 'VERY_LIKELY':
                                showavoid()
                                time.sleep(3)
                                showneutral()
                        elif likelihood_name[face.joy_likelihood] =='LIKELY' or likelihood_name[face.joy_likelihood] =='VERY_LIKELY':
                                showjoy()
                                time.sleep(3)
                                showneutral()
                        elif likelihood_name[face.sorrow_likelihood] =='LIKELY' or likelihood_name[face.sorrow_likelihood] =='VERY_LIKELY':
                                showavoid()
                                time.sleep(3)
                                showneutral()
                        else:
                                blink()
                                showneutral()

        if key == ord("q"):
                break




cap.release()
cv2.destroyAllWindows()
sleep()
credential_path = "/path/to/creditionals" 
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path 
