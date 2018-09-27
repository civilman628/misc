import glob
import os

'''
filelist = glob.glob("/home/scopeserver/RaidDisk/Filter_Full/tops/")

textfile = open('parts_footwear.txt','w')

for item in filelist:
    textfile.write("%s\n" % item)

'''

#for imagefile in filelist:
 #   print imagefile

directory = "/home/scopeserver/RaidDisk/DeepLearning/mwang/progressive_growing_of_gans/results/020-fake-grids-18/"
file_paths = []  # List which will store all of the full filepaths.

    # Walk the tree.

textfile = open('women_fake_tops.txt','w')

for root, directories, files in os.walk(directory):
    for filename in files:
        # Join the two strings in order to form the full filepath.
        filepath = os.path.join(root, filename)
        #file_paths.append(filepath)  # Add it to the list.
        textfile.write("%s\n" % filepath)
