from demo_fashion_7class_new_gpu1 import fashiondetection
from flask import current_app 

try:
    from flask import _app_ctx_stack as stack
except ImportError:
    from flask import _request_ctx_stack as stack


class Detection(object):
    def __init__(self, app=None):
	self.app = app
	self.myObj = self.create_obj()
	#self.detect_obj = self.create_obj()
	


    def create_obj(self):
	obj = fashiondetection()
	return obj
	
    @property
    def detect_obj(self):
	ctx = stack.top
	if ctx is not None:
	    ctx.detect_obj = self.myObj
	return ctx.detect_obj
