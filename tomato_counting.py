#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import time
#import logging
import math
import os
import argparse
from darknet import performDetect

LINE_THICKNESS = 1

#fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#out = loc+'/outputs/'+camera+'_output.mp4'
#out = cv2.VideoWriter(out, fourcc, 20, (frame_w, frame_h))

#outblob = loc+'/outputs/'+camera+'_outblob.mp4'
#diffop = loc+'/outputs/'+camera+'_outdiff.mp4'
#outblob = cv2.VideoWriter(outblob, fourcc, 20, (frame_w, frame_h))
#diffop = cv2.VideoWriter(diffop, fourcc, 20, (frame_w, frame_h))

# ============================================================================

class Tomato(object):
    def __init__(self, id, contour, position):
        self.id = id
        #self.db_id = 0
        self.contour = contour
        self.positions = [position]
        self.unseen_frames = 0
        self.frames_seen = 0
        self.counted = False
        self.tomato_dir = 0

    @property
    def last_position(self):
        return self.positions[-1]
    @property
    def last_position2(self):
        return self.positions[-2]
    @property
    def first_position(self):
        return self.positions[0]        

    def add_position(self, new_position):
        self.positions.append(new_position)
        self.unseen_frames = 0
        #self.frames_seen += 1
    
    def update_contour(self, new_contour):
        self.contour = new_contour

    def draw_trace(self, output_image):
        #for point in self.positions:
         #   cv2.circle(output_image, point, 2, (0, 0, 255), -1)
        #x, y, w, h = self.contour
        #cv2.rectangle(output_image, (x, y), (x + w - 1, y + h - 1), (0, 0, 255), 1)
        if(len(self.positions)>5):
            cv2.polylines(output_image, [np.int32(self.positions[-5:-1])]
                , False, (0, 0, 255), 2)
        else:
            cv2.polylines(output_image, [np.int32(self.positions)]
                , False, (0, 0, 255), 2)

    def draw_bbox(self, output_image):
        x, y, w, h = self.contour
        cv2.rectangle(output_image, (x, y), (x + w - 1, y + h - 1), (0, 255, 0), 2)#LINE_THICKNESS)
        #for point in self.positions:
         #   cv2.circle(output_image, point, 2, (0, 0, 255), -1)
        #x, y, w, h = self.contour
        #cv2.rectangle(output_image, (x, y), (x + w - 1, y + h - 1), (0, 0, 255), 1)
        #if(len(self.positions)>10):
         #   cv2.polylines(output_image, [np.int32(self.positions[-10:-1])]
         #       , False, (0, 0, 255), 1)
        #else:
         #   cv2.polylines(output_image, [np.int32(self.positions)]
          #      , False, (0, 0, 255), 1)

# ============================================================================

class TomatoCounter(object):
    def __init__(self, shape, left_divider, right_divider, max_unseen_frames=10, min_seen_frame=5, distance_threshold=15, angle_deviation=45):
        #self.log = logging.getLogger("tomato_counter")

        self.height, self.width = shape
        self.left_divider = left_divider
        self.right_divider = right_divider

        self.tomatoes = []
        self.tracked =[]
        self.tomato_instances=[]
        self.next_tomato_id = 0
        self.tomato_count = 0
        self.tomato_LHS = 0
        self.tomato_RHS = 0
        self.error_tomato = 0
        self.left_flag = False
        self.right_flag = False
        self.max_unseen_frames = max_unseen_frames
        self.distance_threshold = distance_threshold
        self.angle_deviation = angle_deviation
        self.min_seen_frame = min_seen_frame
        self.db_id = 0


    @staticmethod
    def get_vector(self, a, b):
        """Calculate vector (distance, angle in degrees) from point a to point b.

        Angle ranges from -180 to 180 degrees.
        Vector with angle 0 points straight down on the image.
        Values decrease in clockwise direction.
        """
        dx = float(b[0] - a[0])
        dy = float(b[1] - a[1])

        if np.abs(dx) > self.distance_threshold or np.abs(dy) > self.distance_threshold:
            return None

        distance = math.sqrt(dx**2 + dy**2)

        if distance > self.distance_threshold:
            return None

        angle = 0 
        '''
        un-comment the code below to get the true angle, now we skip this part to improve performance
        
        if dy > 0:
            angle = math.degrees(math.atan(-dx/dy))
        elif dy == 0:
            if dx < 0:
                angle = 90.0
            elif dx > 0:
                angle = -90.0
            else:
                angle = 0.0
        else:
            if dx < 0:
                angle = 180 - math.degrees(math.atan(dx/dy))
            elif dx > 0:
                angle = -180 - math.degrees(math.atan(dx/dy))
            else:
                angle = 180.0        
        '''
        return distance, angle, dx, dy 

    @staticmethod
    def fast_check_dxdy(self, a, b):
        dx = np.abs(float(b[0] - a[0]))
        dy = np.abs(float(b[1] - a[1]))

        if dx > self.distance_threshold or dy > self.distance_threshold:
            return False
        else:
            return True


    @staticmethod
    def is_valid_vector(self, a, angleDev):
        # vector is only valid if threshold is less than distance_threshold
        # current method skip angle and angle deviation check, such as, angle should be is less than 170 or greater than 10 degs 
        # and angle_deviation should less then self.angle_deviation
        distance, angle, _, _ = a
        return (np.abs(angle) > 10 and np.abs(angle) < 170)
        #return (distance <= self.distance_threshold) #and (np.abs(angle) > 10 and np.abs(angle) < 170)  #and  angleDev < self.angle_deviation


    def update_tomato(self, tomato, boxes):
        # Find if any of the matches fits this tomato
        # [distance, index, (cx, cy), (x, y, w, h), dx]
        nearest_match = [100000,0,(0,0),(0,0,0,0),0] 
        #candidate_matches =[]
        for i, box in enumerate(boxes):
            cx, cy, w, h = box[2]

            x = int(cx - w/2)
            y = int(cy - h/2)
            w = int(w)
            h = int(h)

            centroid = (int(cx), int(cy))
            contour = (x, y, w, h)
            
            #if self.fast_check_dxdy(self, tomato.last_position, centroid) == False:
             #   continue

            # store the tomato data
            vector = self.get_vector(self, tomato.last_position, centroid)

            if vector is None :
                continue
            
            #candidate_matches.append([vector[0],i,centroid,contour,vector[2]])
            # only measure angle deviation if we have enough points
            #if tomato.frames_seen > 2:
             #   prevVector = self.get_vector(self, tomato.last_position2, tomato.last_position)
             #   angleDev = abs(prevVector[1]-vector[1])
            #else:
             #   angleDev = 0
            '''    
            b = dict(
                    id = tomato.id,
                    center_x = centroid[0],
                    center_y = centroid[1],
                    vector_x = vector[0],
                    vector_y = vector[1],
                    dx = vector[2],
                    dy = vector[3],
                    counted = tomato.counted,
                    frame_number = frame_no,
                    angle_dev = angleDev
                    )
            
            tracked_blobs.append(b)
            '''
            # check validity, skip for now
            #if self.is_valid_vector(self, vector, angleDev=0):
                #print(hex(tomato.id)[2:], vector[0],vector[1])

            if vector[0] < nearest_match[0]:
                nearest_match = [vector[0],i,centroid,contour,vector[2]]

            '''    
                tomato.add_position(centroid)
                tomato.update_contour(contour)
                tomato.frames_seen += 1
                # check tomato direction
                if vector[2] > 0:
                    # positive value means tomato is moving Right
                    tomato.tomato_dir = 1
                elif vector[2] < 0:
                    # negative value means tomato is moving Left
                    tomato.tomato_dir = -1
                #self.log.debug("Added match (%d, %d) to tomato #%d. vector=(%0.2f,%0.2f)"
                 #   , centroid[0], centroid[1], tomato.id, vector[0], vector[1])
                return i
            '''
        #if len(candidate_matches)==0:
         #   tomato.unseen_frames += 1
         #   return None

        #sorted_candidate_matches = sorted(candidate_matches, key=lambda x:x[0])
        #nearest_match =  sorted_candidate_matches[0]
        if nearest_match[0] < 100000:
            #sorted_candidate_matches = sorted(nearest_match, key=lambda x:x[0])
            #print("distance: ", nearest_match[0])
            tomato.add_position(nearest_match[2])
            tomato.update_contour(nearest_match[3])
            tomato.frames_seen += 1
        
            if nearest_match[4] > 0:
                # positive value means tomato is moving Right
                tomato.tomato_dir = 1
            elif nearest_match[4] < 0:
                # negative value means tomato is moving Left
                tomato.tomato_dir = -1    
            else:
                # video is stop, tomato has no motion.
                tomato.tomato_dir = 0

            return nearest_match[1]
            
        # No matches fit...        
        else:
            tomato.unseen_frames += 1
        #self.log.debug("No match for tomato #%d. unseen_frames=%d"
         #   , tomato.id, tomato.unseen_frames)
            return None


    def update_count(self, boxes, output_image = None, show=True, show_id = True):
        #self.log.debug("Updating count using %d matches...", len(matches))

        # First update all the existing tomatoes
        for tomato in self.tomatoes:
            i = self.update_tomato(tomato, boxes)
            if i is not None:
                #matches.pop(i)
                del boxes[i]

        # Add new tomatoes based on the remaining matches
        for box in boxes:
            cx, cy, w, h = box[2]

            x = int(cx - w/2)
            y = int(cy - h/2)
            w = int(w)
            h = int(h)

            centroid = (int(cx), int(cy))
            contour = (x, y, w, h)
            new_tomato = Tomato(self.next_tomato_id, contour, centroid)
            self.next_tomato_id += 1
            self.tomatoes.append(new_tomato)
            #self.log.debug("Created new tomato #%d from match (%d, %d)."
             #   , new_tomato.id, centroid[0], centroid[1])

        # Count any uncounted tomatoes that are past the divider
        for tomato in self.tomatoes:
            if show and show_id:
                cv2.putText(output_image, hex(tomato.id)[2:], tomato.last_position, cv2.FONT_HERSHEY_PLAIN, 1, (127,255, 255), 1)

            if not tomato.counted and tomato.frames_seen > self.min_seen_frame:
                if (((tomato.first_position[0] < self.right_divider) and (tomato.last_position[0] > self.right_divider) and (tomato.tomato_dir == 1)) or
                    ((tomato.first_position[0] > self.left_divider) and (tomato.last_position[0] < self.left_divider) and (tomato.tomato_dir == -1))):

                    tomato.counted = True
                    if show:
                        tomato.draw_bbox(output_image)
                        tomato.draw_trace(output_image)
                        #cv2.putText(output_image, ("%02d" % tomato.id), (tomato.contour[0],tomato.contour[1]), cv2.FONT_HERSHEY_PLAIN, 1, (127,255, 255), 1)
                    
                    x, y, w, h = tomato.contour
            
                    #x = 0 if x < 0 else x
                    #y = 0 if y < 0 else y
                    tomato_instance = output_image[y:y+h, x:x+w]
                    #print(tomato_instance.shape)
                    #print(y, x, y+h, x+w)
                    #tomato_CIE_lab = cv2.cvtColor(tomato_instance,cv2.COLOR_BGR2LAB)
                    #color = np.mean(tomato_CIE_lab,axis=(0,1))[1]
                 
                    try:
                        tomato_CIE_lab = cv2.cvtColor(tomato_instance,cv2.COLOR_BGR2LAB)
                        color = int(np.mean(tomato_CIE_lab,axis=(0,1))[1])
                    except:
                        #print("over flow value of x, y, w, h")
                        self.error_tomato += 1
                        continue
                    
                    # get the value of a channel from L*a*b, the value range set to 0 to 256
                    
                    #print(color)
                    #elf.tomato_instances.append((tomato.tomato_dir, tomato.id, color ,tomato_instance))
                    self.tomato_instances.append([self.db_id ,tomato.tomato_dir , color, x, y, w, h])
                    self.db_id += 1
                    #self.tracked.append(tomato.id)

                    if tomato.tomato_dir == 1:
                        self.tomato_RHS += 1
                        self.tomato_count += 1
                        #self.right_flag = True
                    elif tomato.tomato_dir == -1:
                        self.tomato_LHS += 1
                        self.tomato_count += 1
                        #self.left_flag = True

                    # update appropriate counter
                    '''
                    if ((tomato.last_position[0] > self.right_divider) and (tomato.tomato_dir == 1) and (tomato.last_position[0] >= (int(frame_w/2)-10))):
                        self.tomato_RHS += 1
                        self.tomato_count += 1
                        self.right_flag = True
                    elif ((tomato.last_position[0] < self.left_divider) and (tomato.tomato_dir == -1) and (tomato.last_position[0] <= (int(frame_w/2)+10))):
                        self.tomato_LHS += 1
                        self.tomato_count += 1
                        self.left_flag = True
                        
                    self.log.debug("Counted tomato #%d (total count=%d)."
                        , tomato.id, self.tomato_count)
                    '''
        # Optionally draw the tomatoes on an image

        if show:
            #if output_image is not None:
             #   for tomato in self.tomatoes:
              #      tomato.draw_trace(output_image)
                    
                # LHS
            cv2.putText(output_image, ("Left Row: %02d" % self.tomato_LHS), (12, 56)
                , cv2.FONT_HERSHEY_PLAIN, 1.2, (127,255, 255), 2)
                # RHS
            cv2.putText(output_image, ("Right Row: %02d" % self.tomato_RHS), (216, 56)
                , cv2.FONT_HERSHEY_PLAIN, 1.2, (127, 255, 255), 2)

        #Remove tomatoes that have not been seen long enough
        #removed = [ v.id for v in self.tomatoes
         #   if v.unseen_frames >= self.max_unseen_frames ]
         
        self.tomatoes[:] = [ v for v in self.tomatoes
            if v.unseen_frames < self.max_unseen_frames ]
        print("total tomatoes: ", len(self.tomatoes))

        #temp = []
        #for item in self.tomatoes:
         #   if item.unseen_frames < self.max_unseen_frames:
         #       temp.append(item)

        #self.tomatoes=temp

        #for id in removed:
         #   self.log.debug("Removed tomato #%d.", id)

        #self.log.debug("Count updated, tracking %d tomatoes.", len(self.tomatoes))

    def get_color_name(self, color):
        if color < 130:
            return "green"
        elif color >= 130 and color < 140:
            return "breaker"
        elif color >= 140 and color < 150:
            return "turning"
        elif color >= 150 and color < 160:
            return "pink"
        elif color >=160 and color < 170:
            return "light_red"
        else:
            return "red"        

    def save_tomato_instances(self, output_dir, show):
        
        for instance in self.tomato_instances:
            dir = "right" if instance[0]== 1 else "left"
            id = instance[1]
            color = instance[2]
            image = instance[3]

            if not show:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 

            color_name = self.get_color_name(color)
            filename = os.path.join(output_dir, str(id)+'_'+ dir + '_' + str(int(color)) + '_' +color_name +'.jpg')
            cv2.imwrite(filename,image)
        
# ============================================================================

def bbox_iou(box1, box2):
    """
    Returns the IoU of two bounding boxes
    """
    #Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,0], box1[:,1], box1[:,2], box1[:,3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,0], box2[:,1], box2[:,2], box2[:,3]
    
    #get the corrdinates of the intersection rectangle
    inter_rect_x1 =  np.max(b1_x1, b2_x1)
    inter_rect_y1 =  np.max(b1_y1, b2_y1)
    inter_rect_x2 =  np.min(b1_x2, b2_x2)
    inter_rect_y2 =  np.min(b1_y2, b2_y2)
    
    #Intersection area
    inter_area = np.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * np.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)
 
    #Union Area
    b1_area = (b1_x2 - b1_x1 + 1)*(b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1)*(b2_y2 - b2_y1 + 1)
    
    iou = inter_area / (b1_area + b2_area - inter_area)
    
    return iou

# ============================================================================

def main(config):
    
    if not os.path.exists(config.video_file):
        raise ValueError("Invalid video file path `"+os.path.abspath(config.video_file)+"`")

    if not os.path.isdir(config.output_path):
        raise ValueError("Invalid output dir `"+os.path.abspath(config.output_path)+"`")

    cap = cv2.VideoCapture(config.video_file)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    if total_frames == 0:
        raise ValueError("Invalid video file. no valid frames found.")

    frame_w = config.frame_width  
    frame_h = config.frame_height
    
    total_tomatoes = 0
    frame_no = 0
    tomato_counter = None
    bbox_count = 0

    clahe = cv2.createCLAHE(clipLimit=3)

    #global tomato_counter
    #global total_tomatoes

    fps=0
    start = time.time()
    while True:
        ret, frame = cap.read()
        if ret != True:
            break;
        t1 = time.time()
        
        frame_no = frame_no + 1
        
        if frame_no % config.frame_sample != 0:
            continue

        #frame = cv2.resize(frame,(frame_w,frame_h),interpolation=cv2.INTER_CUBIC)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        
        if config.equalize_hist:
            for c in range(0,3):
                frame[:,:,c] = clahe.apply(frame[:,:,c])

        #mask the middle 1/3 as black.
        
        #if frame_w == frame_h:
        #background = np.zeros((800,200,3),dtype=np.uint8)
        #background[:,0:100,:] = frame[:,109:209,:]
        #background[:,100:200,:] = frame[:,746:846,:]
        background = np.zeros((800,256,3),dtype=np.uint8)
        background[:,0:128,:] = frame[:,95:223,:]
        background[:,128:256,:] = frame[:,732:860,:]
        frame = background
        #mask_start = int(frame_w/4)
        #mask_end = int(3*frame_w/4)
        
        #frame[:,mask_start:mask_end,:] = 0
        
        #if ret and frame_no < total_frames:
        #t1 = time.time()
        boxes =  performDetect(image=frame, thresh=config.thresh, configPath=config.configPath, weightPath=config.weightPath, metaPath=config.metaPath ,showImage = False)
        #fps  = ( fps + (1./(time.time()-t1)) ) / 2
        #fps  = 1./(time.time()-t1)
        #print("fps= %f"%(fps))
        #print("Processing frame ",frame_no)
        bbox_count = len(boxes) + bbox_count
        # get returned time
        #frame_time = time.time()t1 = time.time()
        


        #frame = cv2.resize(frame,(960,540),interpolation=cv2.INTER_CUBIC)
        #frame = frame[0:512,:,:]
        #frame[:,320:640,:] = 0

        #image = Image.fromarray(frame)
        
        #blobs = yolo.detect_image(image)
        #frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if config.show:
            #convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)        
            for (i, match) in enumerate(boxes):
                cx, cy, w, h = match[2]

                x = int(cx - w/2)
                y = int(cy - h/2)
                w = int(w)
                h = int(h)

                centroid = (int(cx), int(cy))
                contour = (x, y, w, h)
                
                # store the contour data
                #c = dict(
                    #           frame_no = frame_no,
                    #           centre_x = x,
                    #           centre_y = y,
                    #           width = w,
                    #           height = h
                    #           )
                #tracked_conts.append(c)
                
                
                cv2.rectangle(frame, (x, y), (x + w - 1, y + h - 1), (0, 0, 255), LINE_THICKNESS)
                #cv2.circle(frame, centroid, 2, (0, 0, 255), -1)
                #cv2.circle(frame, centroid, 2, (0, 0, 255), -1)

        
        if tomato_counter is None:
            print("Creating tomato counter...")
            tomato_counter = TomatoCounter(frame.shape[:2], frame.shape[1] / 4, 3*frame.shape[1] / 4, config.max_unseen_frames, config.min_seen_frame, config.distance_threshold, config.angle_deviation)
            
        # get latest count
        tomato_counter.update_count(boxes, frame, config.show, config.show_id)
        current_count = tomato_counter.tomato_RHS + tomato_counter.tomato_LHS
        
        # print elapsed time to console
        #elapsed_time = time.time()-start_time
        #print("-- %s seconds --" % round(elapsed_time,2))
        
        # output videoabspath
        #frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # draw dividing line
        # flash green when new tomato counted
        total_tomatoes = current_count
        print("L:", tomato_counter.tomato_LHS, "R:", tomato_counter.tomato_RHS, "Total:", total_tomatoes, "error: ", tomato_counter.error_tomato)

        if config.show:
            cv2.line(frame, (int(frame_w/4), 0),(int(frame_w/4), int(frame_h)),
                (128,255,255), LINE_THICKNESS)
            cv2.line(frame, (int(3*frame_w/4), 0),(int(3*frame_w/4), int(frame_h)),
                (128,255,255), LINE_THICKNESS)
            
            cv2.imshow("preview", frame)
            #cv2.imwrite(("image_%#05d.jpg" % (frame_no)),frame)
            cv2.waitKey(1)
            #out.write(frame)
            
            #if cv2.waitKey(1) and 0xFF == ord('q'):
             #   break

        '''
        if tomato_counter.right_flag:
            cv2.line(frame, (int(frame_w/6), 0),(int(frame_w/6), frame_h),
                (0,255,0), 2*LINE_THICKNESS)  
            tomato_counter.right_flag = False

        elif tomato_counter.left_flag:
            cv2.line(frame, (int(5*frame_w/6), 0),(int(5*frame_w/6), frame_h),
                (0,255,0), 2*LINE_THICKNESS) 
            tomato_counter.left_flag = False

        else:
            cv2.line(frame, (int(frame_w/6), 0),(int(frame_w/6), int(frame_h)),
            (0,0,255), LINE_THICKNESS)
            cv2.line(frame, (int(5*frame_w/6), 0),(int(5*frame_w/6), int(frame_h)),
            (0,0,255), LINE_THICKNESS)
        '''

        #if current_count > total_tomatoes:
            #   cv2.line(frame, (0, int(2*frame_h/3)),(frame_w, int(2*frame_h/3)),
            #       (0,255,0), 2*LINE_THICKNESS)
        #else:
            #   cv2.line(frame, (0, int(2*frame_h/3)),(frame_w, int(2*frame_h/3)),
            #  (0,0,255), LINE_THICKNESS)
            
        # update with latest count

        # draw upper limit
        #cv2.line(frame, (0, 100),(frame_w, 100), (0,0,0), LINE_THICKNESS)
        #frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        '''
        if config.show:
            cv2.imshow("preview", frame)
            cv2.waitKey(0)
            #out.write(frame)
            
            if cv2.waitKey(27) and 0xFF == ord('q'):
                break
        '''
        fps  = ( fps + (1./(time.time()-t1)) ) / 2
        #fps  = 1./(time.time()-t1)
        print("fps= %f"%(fps))

        #else:
         #   break
    print("total boxes:", bbox_count)
    #print(tomato_counter.tracked)
    if config.show:
        #cv2.waitKey(0)
        cv2.destroyAllWindows()

    duration = time.time() - start
    print("duration of process: ", duration)
    #tomato_counter.save_tomato_instances(config.output_path, show=config.show)
    cap.release()
    #out.release()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    #counter's configuration.
    parser.add_argument('--max_unseen_frames', type=int, default=5, help='remove the object after max unseen frames')
    parser.add_argument('--distance_threshold', type=float, default=25, help='connect center positions as a line that smaller than this value')
    parser.add_argument('--angle_deviation', type=float, default=45, help='angle deviation between 2 continuouse angles for the same line')
    parser.add_argument('--min_seen_frame', type=int, default=2, help='the min seen frame before counting')
    parser.add_argument('--frame_sample', type=int, default=2, help='read every N frame')
    parser.add_argument('--frame_width', type=int, default=256, help='resize into this width')
    parser.add_argument('--frame_height', type=int, default=800, help='resize into this height')
    parser.add_argument('--equalize_hist', type=bool, default=False, help='equalize histgram to balance the intensity of the image, base on adaptive method')
    parser.add_argument('--show', type=bool, default=False,help='set to true to see the detection video; set false in production model to increase performance')
    parser.add_argument('--show_id', type=bool, default=False, help='show a hex id on tomato, this is for debug tomato counting method. this parameter is valid if "show" and "show_id" are set to true')
    parser.add_argument('--video_file', type=str, default='0548207774494477-20190501-down-190640.mkv', help='a video file')
    parser.add_argument('--output_path', type=str, default='../tomatoes', help='a folder for output tomato instances')

    #detection model's configuration.
    parser.add_argument('--thresh', type=float, default=0.3, help='min prob value for tomato prediction')
    parser.add_argument('--configPath', type=str, default='./model_data/yolov3_tomato_v5_one_eighth_384x384_2019_data.cfg', help='network configuration file *.cfg')
    parser.add_argument('--weightPath', type=str, default="./model_data/yolov3_tomato_v5_one_eighth_384x384_2019_data_25000.weights", help='network weight file')
    parser.add_argument('--metaPath', type=str, default='./model_data/tomato.data', help='network meta file')

    config = parser.parse_args()
    print(config)

    main(config)
