import os

from flask import Flask, request
from flask_restful import Resource, Api, reqparse
from flask_jsonpify import jsonify
from colorama import init
from termcolor import colored
import numpy as np

from transfer.resnet50 import get_final_model
from transfer.predict_model import prep_from_image

def start_server(project, weights, extra_conv = False):

    app = Flask(__name__)
    api = Api(app)

    parser = reqparse.RequestParser()
    parser.add_argument('img_path', type = str)

    img_dim = 224 * project['img_size']
    conv_dim = 7 * project['img_size']
    model = get_final_model(img_dim, conv_dim, project['number_categories'], project[weights], extra_conv = extra_conv)

    class Predict(Resource):
        def post(self):
            args = parser.parse_args(strict = True)
            img_path = os.path.expanduser(args['img_path'])
            if os.path.isfile(img_path):
                if img_path.lower().find('.png') > 0 or img_path.lower().find('.jpg') > 0 or img_path.lower().find('.jpeg') > 0:
                    img = prep_from_image(img_path, img_dim, project['augmentations'])
                    predicted = model.predict(img)
                    pred_list = [float(p) for p in predicted[0]]
                    result = {'weights': project[weights],
                            'image_path': img_path,
                            'predicted': project['categories'][np.argmax(predicted)],
                            'classes': project['categories'],
                            'class_predictions': pred_list}

                    return jsonify(result)
                else:
                    return 'File must be a jpeg or png: ' + args['img_path']
            elif os.path.isdir(img_path):
                result = []
                for file_name in os.listdir(img_path):
                    if file_name.lower().find('.png') > 0 or file_name.lower().find('.jpg') > 0 or file_name.lower().find('.jpeg') > 0:
                        full_file_name = os.path.join(img_path, file_name)
                        img = prep_from_image(full_file_name, img_dim, project['augmentations'])
                        predicted = model.predict(img)
                        pred_list = [float(p) for p in predicted[0]]
                        result.append({'weights': project[weights],
                                'image_path': full_file_name,
                                'predicted': project['categories'][np.argmax(predicted)],
                                'classes': project['categories'],
                                'class_predictions': pred_list})
                if len(result) > 0:
                    return jsonify(result)
                else:
                    return 'No images found in directory: ' + args['img_path']

            else:
                return 'Image does not exist locally: ' + args['img_path']


    api.add_resource(Predict, '/predict')
    print('')
    print('To predict a local image, simply:')
    print('')
    print(colored('curl http://localhost:' + str(project['api_port']) + '/predict -d "img_path=/path/to/your/img.png" -X POST', 'green'))
    print('')
    print('or')
    print('')
    print(colored('curl http://localhost:' + str(project['api_port']) + '/predict -d "img_path=/path/to/your/img_dir" -X POST', 'green'))
    print('')
    app.run(port = str(project['api_port']))
