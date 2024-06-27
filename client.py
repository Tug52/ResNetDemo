import requests
from tkinter.filedialog import askopenfilename

url = 'http://127.0.0.1:5000/predict'

while 1:
    filename = askopenfilename()  # show an "Open" dialog box and return the path to the selected file
    print(filename)


    data = {'image_path': filename}
    response = requests.post(url, json=data)
    print(response.json())