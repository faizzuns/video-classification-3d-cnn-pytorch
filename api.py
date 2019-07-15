from flask import *
from flask_cors import *
import os
import datetime

api = Flask(__name__)
CORS(api)

DEFAULT_CLASSIFICATION_THRESHOLD = 0.25
DEFAULT_MAPPING_THRESHOLD = 10

@api.route('/')
def hello():
	return jsonify({
        'message': 'success'
    })

@api.route('/classify', methods=['POST'])
def classify():
    data = request.form
    
    if 'video_url' not in data:
        abort(400)

    video_url = data['video_url']
    mapping_threshold = float(data['mapping_threshold']) if 'mapping_threshold' in data else DEFAULT_MAPPING_THRESHOLD
    classification_threshold = float(data['classification_threshold']) if 'classification_threshold' in data else DEFAULT_CLASSIFICATION_THRESHOLD

    # download & put video in videos folder

    targeted_video = ['tes-video-1.mp4']
    timestamp = int(datetime.datetime.now().timestamp())
    # create input files
    input_filename = 'in_' + str(timestamp)
    output_filename = 'out_' + str(timestamp) + '.json'
    
    input_file = open('inputs/' + input_filename, 'w')
    text = ''
    for target in targeted_video:
        text = text + target + '\n'
    input_file.write(text)
    input_file.close()

    bs = 2
    script = 'python main.py ' 
    script = script + '--input ./inputs/' + input_filename + ' '
    script = script + '--video_root ./videos ' 
    script = script + '--output ./outputs/' + output_filename + ' ' 
    script = script + '--model trained_models/resnext-101-kinetics.pth ' 
    script = script + '--mode score --model_name resnext ' 
    script = script + '--model_depth 101 --resnet_shortcut B ' 
    script = script + '--batch_size ' + str(bs)
    
    os.system(script)

    # classname_list_dict = {}
    # cnt = 0

    # f = open('class_names_list', 'r')
    # f1 = f.readlines()
    # for line in f1:
    #     clear_line = line.replace('\n','')
    #     classname_list_dict[clear_line] = cnt
    #     cnt = cnt + 1
    # print(len(classname_list_dict))

    return jsonify({
        'mapping_threshold': mapping_threshold,
        'classification_threshold': classification_threshold,
        'video_url': video_url,
        'input_filename': input_filename,
        'output_filename': output_filename
    })

@api.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)

@api.errorhandler(400)
def bad_request(error):
    return make_response(jsonify({'error': 'Bad Request'}), 400)


if __name__ == '__main__':
	api.secret_key = os.urandom(12)
	api.run(debug=True)