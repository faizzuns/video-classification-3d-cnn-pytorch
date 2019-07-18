from flask import *
from flask_cors import *
from threading import Thread
import model_api as model
import os
import datetime
import time

api = Flask(__name__)
CORS(api)

DEFAULT_CLASSIFICATION_THRESHOLD = 0.25
DEFAULT_MAPPING_THRESHOLD = 10

@api.before_request
def before():
	if model.db.is_closed():
		model.db.connect()

@api.after_request
def after(request):
	if not model.db.is_closed():
		model.db.close()

	return request

@api.route('/')
def hello():
	return jsonify({
        'message': 'success'
    })

def getClassnameList():
    classname_list_dict = {}
    cnt = 0

    f = open('class_names_list', 'r')
    f1 = f.readlines()
    for line in f1:
        clear_line = line.replace('\n','')
        classname_list_dict[clear_line] = cnt
        cnt = cnt + 1
    
    return classname_list_dict

def getMappingClasses():
    result_mapping = {}

    f = open('mapping.txt', 'r')
    rows = f.readlines()

    for row in rows:
        clear_row = row.replace('\n','')
        content = clear_row.split(':')
        if len(content[1]) > 0:
            arr = content[1].split(",")
            for item in arr:
                if item not in result_mapping:
                    result_mapping[item] = [content[0]]
                else:
                    result_mapping[item].append(content[0])
    
    return result_mapping

def mapping(class_names_list, output_file, classification_threshold, mapping_threshold):
    result = []
    mapping_classlist = getMappingClasses()

    with open('outputs/' + output_file) as json_file:
        data_json = json.load(json_file)

        for result_video in data_json:
            result_item = {}
            labels = {}
            total = 0

            result_item['video_name'] = result_video['video']
        
            for clip in result_video['clips']:
                idx = class_names_list[clip['label']]
        
                if clip['scores'][idx] > classification_threshold:
                    for iab_label in mapping_classlist[clip['label']]:
                        total = total + clip['scores'][idx]
                        if iab_label not in labels:
                            labels[iab_label] = clip['scores'][idx]
                        else:
                            labels[iab_label] = labels[iab_label] + clip['scores'][idx]
            
        
            sorted_x = sorted(labels.items(), key=lambda kv: kv[1])
            import collections
            sorted_x.reverse()
            labels = collections.OrderedDict(sorted_x)
        
            result_labels = []
            for label in labels:
                if round((labels[label]/total)*100,2) >= mapping_threshold:
                    result_labels.append(label)
            
            result_item['labels'] = result_labels
            result.append(result_item)
    
    return result
        
def generateClassificationScript(input_filename, output_filename, batch_size):
    script = 'python3 main.py ' 
    script = script + '--input ./inputs/' + input_filename + ' '
    script = script + '--video_root ./videos ' 
    script = script + '--output ./outputs/' + output_filename + ' ' 
    script = script + '--model trained_models/resnext-101-kinetics.pth ' 
    script = script + '--mode score --model_name resnext ' 
    script = script + '--model_depth 101 --resnet_shortcut B ' 
    script = script + '--batch_size ' + str(batch_size)
    return script

def createInputFile(input_filename, targeted_video):
    input_file = open('inputs/' + input_filename, 'w')
    input_file.write(targeted_video + '\n')
    input_file.close()

def startAsync(script, data):
    os.system(script)
    result = mapping(getClassnameList(), data['output_filename'], data['classification_threshold'], data['mapping_threshold'])
    if len(result) > 0:
        data = model.Inference.get_by_id(data['inference_id'])
        text = ''
        for item in result[0]['labels']:
            text = text + item + ','
        text = text[:-1]
        data.result = str(text)
        data.status = 1
        data.save()
    

@api.route('/classify', methods=['POST'])
def classify():
    data = request.form

    if 'video_url' not in data:
        abort(400)

    video_url = data['video_url']
    batch_size = int(data['batch_size']) if 'batch_size' in data else 32
    mapping_threshold = float(data['mapping_threshold']) if 'mapping_threshold' in data else DEFAULT_MAPPING_THRESHOLD
    classification_threshold = float(data['classification_threshold']) if 'classification_threshold' in data else DEFAULT_CLASSIFICATION_THRESHOLD

    # download & put video in videos folder
    # DUMMY
    targeted_video = 'tes-video-1.mp4'
    # ------

    timestamp = int(datetime.datetime.now().timestamp())
    input_filename = 'in_' + str(timestamp)
    output_filename = 'out_' + str(timestamp) + '.json'
    
    data_model = model.Inference(video_url = video_url, input_file = input_filename, output_file = output_filename, status = 0, result='')
    data_model.save()

    createInputFile(input_filename, targeted_video)
    script = generateClassificationScript(input_filename, output_filename, batch_size)

    data = {
        'mapping_threshold': mapping_threshold,
        'classification_threshold': classification_threshold,
        'video_url': video_url,
        'input_filename': input_filename,
        'output_filename': output_filename,
        'inference_id': data_model.id
    }

    my_thread = Thread(target=startAsync, args=[script, data])
    my_thread.start()

    return makeCustomResponse(data, 'pending')

@api.route('/classify/<int:id>', methods=['GET'])
def getClassifyDetail(id):
    data = model.Inference.get_by_id(id)
    if data == None:
        return makeCustomResponse('Inference not found', 'failed')
    else:
        if data.status == 0:
            return makeCustomResponse(None, 'waiting')
        else:
            # json.loads(data.result)
            res = data.result.split(',')
            return makeCustomResponse(res, 'success')

def makeCustomResponse(data, status):
    return jsonify({
        'status': status,
        'data': data
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