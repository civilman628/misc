
from flask import Flask, Blueprint, abort
from flask import request
from flask import jsonify
from flask_cors import CORS, cross_origin
from flask_detect_7class_style_7777 import Detection
app = Flask(__name__)
CORS(app)
myobj = Detection(app)

@app.route('/')
def index():
    return "This is fashion detection"


@app.route('/api/detect', methods=['POST'])
@cross_origin()
def create_task():
    
    if not request.json or not 'url' in request.json:
        abort(400)
    try:
	print 'test'
        result = myobj.detect_obj.demo(request.json['url'])
	return_data = {}
	return_data['result'] = 'OK'
	return_data['data'] = result
        return jsonify(return_data), 201
    except Exception as e:
	return_data = {}
        return_data['result'] = 'ERROR'
        return_data['json'] = str(request.json)
        return_data['errorMessage'] = str(e)
        return jsonify(return_data), 201



@app.route('/api/base64_detect', methods=['POST'])
@cross_origin()
def create_base64_task():
    if not request.json or not 'base64' in request.json:
        abort(400)
    try:
	print 'base64'
        result = myobj.detect_obj.demobase64(request.json['base64'])
	return_data = {}
	return_data['result'] = 'OK'
	return_data['data'] = result
        return jsonify(return_data), 201
    except Exception as e:
	return_data = {}
        return_data['result'] = 'ERROR'
        return_data['json'] = str(request.json)
        return_data['errorMessage'] = str(e)
        return jsonify(return_data), 201


if __name__ == '__main__':
	print "aa"
	app.run(port=7777 ,debug=True, use_reloader=False, host='0.0.0.0')
