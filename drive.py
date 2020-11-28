import argparse
import base64
from datetime import datetime
import os
import socketio
import eventlet
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO

import shutil
import torch
import torchvision.transforms as tf
from utils.getter import *
from training.checkpoint import load

val_transforms = tf.Compose([
    tf.ToTensor(),
    tf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

"""Reference: https://github.com/ManajitPal/DeepLearningForSelfDrivingCars in drive.py file"""

# start socket server
sio = socketio.Server()
# flask web app
app = Flask(__name__)
# init model and image (array) empty
model = None
prev_image_array = None

MAX_SPEED = 25
MIN_SPEED = 10

speed_limit = MAX_SPEED


@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        # take current throttle
        throttle = float(data['throttle'])
        # take drive angle
        steering_angle = float(data['steering_angle'])
        # take current speed
        speed = float(data['speed'])
        # image from front camera
        image = Image.open(BytesIO(base64.b64decode(data['image'])))

        try:
            image = val_transforms(image)
            print('*****************************************************')
            steering_angle = float(model.inference_img(image))

            # Toc do trong khoang quy dinh
            global speed_limit
            if speed > speed_limit:
                speed_limit = MIN_SPEED  # giam toc do
            else:
                speed_limit = MAX_SPEED

            throttle = 1.0 - steering_angle**2 - (speed/speed_limit)**2
            print('{:10.4f} {:10.4f} {:10.4f}'.format(
                steering_angle, throttle, speed))
            # send to software to autonomously driving
            send_control(steering_angle, throttle)

        except Exception as e:
            print(e)

        # save frame
        if args.image_folder != '':
            timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            image_filename = os.path.join(args.image_folder, timestamp)
            image.save('{}.jpg'.format(image_filename))
    else:
        # NOTE: DON'T EDIT THIS.
        sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    print('connect', sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        '--model',
        type=str,
        help='Path to model h5 file. Model should be on the same path.'
    )
    parser.add_argument(
        '--image_folder',
        type=str,
        nargs='?',
        default='',
        help='Path to image folder. This is where the images from the run will be saved.'
    )

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    model = Regressor(
        n_classes=1,
        optim_params={'lr': 1e-3},
        criterion=MSELoss(),
        optimizer=torch.optim.Adam,
        device=device
    )

    load(model, 'weights/udacity/NetworkLight_30.pt')

    if args.image_folder != '':
        print("Creating image folder at {}".format(args.image_folder))
        if not os.path.exists(args.image_folder):
            os.makedirs(args.image_folder)
        else:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        print("RECORDING THIS RUN ...")
    else:
        print("NOT RECORDING THIS RUN ...")

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
