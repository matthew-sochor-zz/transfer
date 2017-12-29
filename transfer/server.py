import os

from flask import Flask, request
from flask_restful import Resource, Api, reqparse
from flask_jsonpify import jsonify
from colorama import init
from termcolor import colored
import numpy as np

from transfer.resnet50 import get_resnet_final_model
from transfer.xception import get_xception_final_model
from transfer.inception_v3 import get_inception_v3_final_model
from transfer.predict_model import prep_from_image, gen_from_directory, multi_predict

def start_server(project, weights):

    app = Flask(__name__)
    api = Api(app)

    parser = reqparse.RequestParser()
    parser.add_argument('img_path', type = str)

    img_dim = 224 * project['img_size']
    conv_dim = 7 * project['img_size']
    models = []
    for weight in project[weights]:
        if project['architecture'] == 'resnet50':
            models.append(get_resnet_final_model(img_dim, conv_dim, project['number_categories'], weight, project['is_final']))
        elif project['architecture'] == 'xception':
            models.append(get_xception_final_model(img_dim, conv_dim, project['number_categories'], weight, project['is_final']))
        else:
            models.append(get_inception_v3_final_model(img_dim, conv_dim, project['number_categories'], weight, project['is_final']))

    class Predict(Resource):
        def post(self):
            args = parser.parse_args(strict = True)
            img_path = os.path.expanduser(args['img_path'])
            if os.path.isfile(img_path):
                if img_path.lower().find('.png') > 0 or img_path.lower().find('.jpg') > 0 or img_path.lower().find('.jpeg') > 0:
                    aug_gen = prep_from_image(img_path, img_dim, project['augmentations'])
                    pred_list, predicted = multi_predict(aug_gen, models, project['architecture'])
                    pred_list = [[float(p) for p in pred] for pred in list(pred_list)]
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

                for aug_gen, file_name in gen_from_directory(img_path, img_dim, project):
                    pred_list, predicted = multi_predict(aug_gen, models, project['architecture'])
                    pred_list = [[float(p) for p in pred] for pred in list(pred_list)]
                    result.append({'weights': project[weights],
                            'image_path': file_name,
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
