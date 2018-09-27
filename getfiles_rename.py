import glob
import os

'''
filelist = glob.glob("/home/scopeserver/RaidDisk/DeepLearning/mwang/data/scopestyle_wholerperson_parts_7classes/subclassmen/merge_sub_classes/")

textfile = open('parts_footwear.txt','w')

for item in filelist:
    textfile.write("%s\n" % item)

'''

#for imagefile in filelist:
 #   print imagefile

directory = "/home/scopeserver/RaidDisk/DeepLearning/mwang/data/scopestyle_trend_women/"
file_paths = []  # List which will store all of the full filepaths.

    # Walk the tree.

#textfile = open('trend_men_imagepart.txt','w')

for root, directories, files in os.walk(directory):
    for filename in files:
        # Join the two strings in order to form the full filepath.
        filepath = os.path.join(root, filename)
        newname = filepath.replace(" ","")
        os.rename(filepath,newname)
        #file_paths.append(filepath)  # Add it to the list.
        #textfile.write("%s\n" % filepath)
