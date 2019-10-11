
import time
import cv2
import numpy as np

## SEND EMAIL ALERTS
import smtplib 
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
#from sinchsms import SinchSMS 

def textPopUp(eventStream_name):
	#create event text pop up
    t = time.localtime()
    #print("notification-text pop up")
    current_time = time.strftime("%H:%M:%S", t)
    camera_name = eventStream_name + " at "+str(current_time)
    text = "Trespassing occurred on:"
    
    width1,height1 = cv2.getTextSize(text, cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, thickness=1)[0]
    width2,height2 = cv2.getTextSize(camera_name, cv2.FONT_HERSHEY_COMPLEX, fontScale=1, thickness=1)[0]
    
    max_wid = max(width1,width2)
    img = np.ones((100, max_wid+20,3))*(0,0,255)    #create a blank red window of adjustable width

    return (img,text,camera_name)

def email_alert(name, rtsp, reason):
    print('sending email alert')
    sender = "sujayrpi@gmail.com"
    receiver = ["rishabh@omnipresenttech.com", "prosenjita@gmail.com"]

    msg = MIMEMultipart()
    msg['From'] = sender
    msg['to'] = receiver
    msg['Subject'] = "[	ATTENTION USER	]"
    body = " Your camera '{}' of RTSP {} has encountered {}".format(name, rtsp, reason)
    msg.attach(MIMEText(body, 'plain'))

    try:
        # creates SMTP session
        s = smtplib.SMTP('smtp.gmail.com', 587)
        # start TLS for security
        s.starttls()
        # Authentication
        s.login(sender, "sujay2908")
        print("Logging into server account")

        # Converts the Multipart msg into a string
        text = msg.as_string()
        for i in receiver:
        	print(i)
        s.sendmail(sender, i, text)
        # terminating the session
        s.quit()
    except:
        print("Email not sent, No Internet Connection")
        pass


'''  
# function for sending SMS 
def sendSMS(message): 
  
    # enter all the details 
    # get app_key and app_secret by registering 
    # a app on sinchSMS
    number = '+917042903334'
    app_key = 'bb5ac3d6-d55b-4d34-9680-8bb712ecb4a4'
    app_secret = 'Fr79j61PxU+BuSexJD93Bg=='
  
    # enter the message to be sent 
    #message = "WARNING SOMEONE'S TRESSAPSSING"
 
    client = SinchSMS(app_key, app_secret) 
    print("Sending '%s' to %s" % (message, number)) 
  
    response = client.send_message(number, message) 
    message_id = response['messageId'] 
    response = client.check_status(message_id) 
  
    # keep trying unless the status retured is Successful 
    while response['status'] != 'Successful': 
        print(response['status']) 
        time.sleep(1) 
        response = client.check_status(message_id) 
  
    print(response['status'])
'''
