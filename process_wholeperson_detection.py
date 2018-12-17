import csv
import pandas as pd
import os
import base64
from PIL import Image
import requests
from io import BytesIO
import json

path = '/home/scopeserver/RaidDisk/SocialMediaImages/ScopeStyle/'
api_address = 'http://45.79.98.24:5048/prediction'

fashionelement =[]


with open('./farfetch_detected_wholeperson_results.csv') as csvfile:
    csvread = csv.reader(csvfile,delimiter='\t')
    for i, row in enumerate(csvread):
        gender = int(row[3])
        label = row[4]

        if label == 'bags':
            fashionelement.append(('bags',0))
            continue

        x = int(row[5])
        y = int(row[6])
        w = int(row[7])
        h = int(row[8])
        print(x, y , w, h)
        file = os.path.join(path,row[1])
        image = Image.open(file)
        cropped = image.crop((x,y,x+w,y+h))
        buffer = BytesIO()
        cropped.save(buffer,format='JPEG')
        imagestring = str(base64.b64encode(buffer.getvalue()))[1:-1]
        #print(imagestring[0:100])

        if gender==0:
            content={
                "modelId":"men-match",
                "base64":imagestring
            }
        else:
            content={
                "modelId":"women-match",
                "base64":imagestring
            }

        req = requests.post(api_address, json=content, headers={"Content-type": "application/json", "Accept": "text/plain"})

        if req.status_code == 200:
            result = json.loads(req.text)['tags'][0]
            subclass = result['label']
            score = result['score']
            fashionelement.append((subclass,score))
            print (i,subclass, score)
        else:
            raise Exception(req.content)


df = pd.DataFrame(fashionelement)
df.to_csv('farfetch_fashionelement.csv')