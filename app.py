
from flask import *
import face_recognition
import pickle
import cv2
import io
import base64 
import numpy as np
from PIL import Image

app = Flask(__name__)

@app.route('/', methods = ['GET', 'POST'])
def handle_request():

    request_data = request.form.get('image')

    data = pickle.loads(open('face_enc', "rb").read())
    imgdata = base64.b64decode(request_data)
    image = Image.open(io.BytesIO(imgdata))
    rgb = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    encodings = face_recognition.face_encodings(rgb)
    names = []
    for encoding in encodings:
        matches = face_recognition.compare_faces(data["encodings"],
        encoding)
        name = "Unknown"
        if True in matches:
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}
            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1
                name = max(counts, key=counts.get)
    
            names.append(name)
        return name

    # res = recognize_face(request_data)

    # return res

if __name__ == "__main__":
    app.run()