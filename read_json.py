import json

with open('/home/scopeserver/RaidDisk/DeepLearning/mwang/data/scope_style_with_bbox.json') as f:
    data = json.load(f)

count=1

rows =[]

for item in data:
    if item['image_detected_result'] =='':
        continue
    else:
        #print item
        #print item['image_detected_result']
        filename = item['image_retina']
        gender=item['product_gender']

        if gender==1:
            continue

        singleproduct=item['image_primary']

        if singleproduct==1:
            continue

        d=json.loads(item['image_detected_result'])

        if d.has_key('data')==False:
            continue

        bbox=d['data']

        if len(bbox)<1:
            continue

        for box in bbox:
            x=box['x']
            y=box['y']
            w=box['w']
            h=box['h']
            name=box['name']

            if name == 'tops' or name == 'outerwear':
                if w > h:
                    continue

            rows.append([filename,gender,x,y,w,h,name])
            
            count=count+1
            print count      

csvfile=open('record_men.csv','w')

for row in rows:
    print>>csvfile, row

