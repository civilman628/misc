
import cv2
import time
import os
import glob

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
        print(video_file)
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
            if count % 30 != 0:
                continue

            
            #frame = cv2.resize(frame,(1224,1024),interpolation=cv2.INTER_CUBIC)
            frame = frame[:918,:,:]

            frame = cv2.resize(frame,(1024,768),interpolation=cv2.INTER_CUBIC)
            #frame[:,341:682,:] = 0
            # Write the results back to output location.
            image_name= output_loc + filename +"_%#05d.jpg" % (count)
            cv2.imwrite(image_name, frame)
            
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

input_loc = '/home/mingming/Downloads/April 2, 2019/flower/*.avi'
output_loc = './april2_flower/'
video_to_frames(input_loc, output_loc)
