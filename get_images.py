
import cv2
import time
import os
import glob
import numpy as np
import tensorflow as tf

def video_to_frames(input_loc, output_loc):
    """Function to extract frames from input video file
    and save them as separate frames in an output directory.
    Args:
        input_loc: Input video folder (file).
        output_loc: Output directory to save the frames.
    Returns:
        None
    """
    try:
        os.mkdir(output_loc)
    except OSError:
        pass
    # Log the time
    time_start = time.time()
    # Start capturing the feed

    video_file_list = glob.glob(input_loc)

    for video_file in video_file_list:

        filename,ext = os.path.splitext(os.path.basename(video_file))
        cap = cv2.VideoCapture(video_file)
        # Find the number of frames
        video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
        print ("Number of frames: ", video_length)
        count = 0
        print ("Converting video..\n")
        # Start converting the video
        

        while True:
            ret, frame = cap.read()
            if ret != True:
                break;
            count = count + 1
            #print (count)
            # Extract the frame
            if count % 2 != 0:
                continue

            #w, h, c = frame.shape
            
            #frame = cv2.resize(frame,(1224,1024),interpolation=cv2.INTER_CUBIC)
            #frame = frame[:717,:,:] #frame = frame[:918,:,:]

            #frame = cv2.resize(frame,(1024,768),interpolation=cv2.INTER_CUBIC)

            #frame[:,256:768:,] = 0

            #[w, h, c] = frame.shape
            
            #for i in range(w):
             #   for j in range(h):
              #      for k in range(c):
               #         if frame[i,j,k] >200:
                #            frame[i,j,k] = frame[i,j,k]-50

            
            #for element in frame.flat:
             #   if element >180:
              #      element = element - 50
            #print(np.max(frame))
            #frame1 = np.array(np.power(frame/(1.0*np.max(frame)),3.0)*255.0)#, dtype=np.uint8)
            #print(np.max(frame1))
            #frame3 = np.array((frame1 + frame)/2.0,dtype=np.uint8)
            frame_full_equalize = frame.copy()
            #frame_cv2_normalize = frame.copy()
            #frame_normalize = np.asarray(frame.copy(),dtype=float)
            #frame_partial_equalize = frame.copy()
            #frame_full_Clahe = frame.copy()

            '''
            mean = np.mean(frame,axis=(0,1))
            std = np.std(frame,axis=(0,1))
            #adjusted_std = np.max([std, 1.0/np.sqrt(num_element)])
            frame_normalize = ((frame-mean)/std)
            #frame_normalize2 = (frame_normalize-np.min(frame_normalize))/np.max(frame_normalize)
            cv2.imshow("preview", frame_normalize)
            cv2.waitKey(0)
            '''

            '''
            for c in range(0,3):
                mean = np.mean(frame[:,:,c])
                std = np.std(frame[:,:,c])
                frame_normalize[:,:,c]=(frame[:,:,c]-mean)/std
            cv2.imshow("preview", frame_normalize)
            cv2.waitKey(0)
            '''


            '''
            cv2.normalize(frame, frame_cv2_normalize, 50, 250, cv2.NORM_MINMAX)
            cv2.imshow("preview", frame_cv2_normalize)
            cv2.waitKey(0)
            '''

            '''
            std_image = tf.image.per_image_standardization(frame)
            with tf.Session() as sess:
                result = sess.run(std_image)
                cv2.imshow("preview", result)
                cv2.imshow("source", frame)
                cv2.waitKey(0)
            '''

            #frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            #frame_hsv2 = frame_hsv.copy()

            #frame_Lab = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)
            #frame_Lab_Clahe = frame_Lab.copy()

            
            
            #frame_hsv2[:,:,2] = clahe.apply(frame_hsv2[:,:,2])
            #frame_hsv_clahe_reverse = cv2.cvtColor(frame_hsv2, cv2.COLOR_HSV2BGR)

            #clahe = cv2.createCLAHE(clipLimit=3)
            
            #for c in range(0,3):
             #   frame_full_Clahe[:,:,c] = clahe.apply(frame[:,:,c])

            #for c in range(0,3):
             #   frame_full_equalize[:,:,c] = cv2.equalizeHist(frame_full_equalize[:,:,c])

            
            #frame_hsv[:,:,2] = cv2.equalizeHist(frame_hsv[:,:,2])

            #frame_hsv_reverse = cv2.cvtColor(frame_hsv, cv2.COLOR_HSV2BGR)    

            #frame_Lab[:,:,0] = cv2.equalizeHist(frame_Lab[:,:,0])

            #frame_Lab_reverse = cv2.cvtColor(frame_Lab, cv2.COLOR_Lab2BGR)

            #clahe2 = cv2.createCLAHE(clipLimit=3)

            #frame_Lab_Clahe[:,:,0] = clahe.apply(frame_Lab_Clahe[:,:,0])
            #frame_Lab_Clahe_reverse = cv2.cvtColor(frame_Lab_Clahe, cv2.COLOR_Lab2BGR)


            #for c in range(0,3):
             #   frame_partial_equalize[:,0:341,c] = clahe.apply(frame[:,0:341,c])

            #for c in range(0,3):
             #   frame_partial_equalize[:,682:-1,c] = clahe.apply(frame[:,682:-1,c])

            frame[:,341:682,:] = 0
            # Write the results back to output location.
            image_name= output_loc + filename +"_%#05d.jpg" % (count)
            cv2.imwrite(image_name, frame)

            #image_name= output_loc + filename +"_%#05d_gamma.jpg" % (count)
            #cv2.imwrite(image_name, frame)

            #image_name= output_loc + filename +"_%#05d_gamma_full_mean.jpg" % (count)
            #cv2.imwrite(image_name, frame3)

            #image_name= output_loc + filename +"_%#05d_full_equlize.jpg" % (count)
            #cv2.imwrite(image_name, frame_full_equalize)

            ## image_name= output_loc + filename +"_%#05d_full_Clahe.jpg" % (count)
            ## cv2.imwrite(image_name, frame_full_Clahe)

            #image_name= output_loc + filename +"_%#05d_hsv_equlize.jpg" % (count)
            #cv2.imwrite(image_name, frame_hsv_reverse)  

            #image_name= output_loc + filename +"_%#05d_hsv_Clahe.jpg" % (count)
            #cv2.imwrite(image_name, frame_Lab_Clahe_reverse)

            #image_name= output_loc + filename +"_%#05d_Lab_L_equlize.jpg" % (count)
            #cv2.imwrite(image_name, frame_Lab_reverse)

            #image_name= output_loc + filename +"_%#05d_lab_L_Clahe.jpg" % (count)
            #cv2.imwrite(image_name, frame_Lab_Clahe_reverse)



            #image_name= output_loc + filename +"_%#05d_partial_equalize.jpg" % (count)
            #cv2.imwrite(image_name, frame_partial_equalize)
            
            # If there are no more frames left
            if (count > (video_length-1)):
                # Log the time again
                time_end = time.time()
                # Release the feed
                cap.release()
                # Print stats
                print ("Done extracting frames.\n%d frames extracted" % count)
                print ("It took %d seconds forconversion." % (time_end-time_start))
                break

input_loc = '/home/mingming/Downloads/2019_May_01/0548207774494477-20190501-down-190640.mkv'
output_loc = './0548207774494477-20190501-down-190640/'
video_to_frames(input_loc, output_loc)
