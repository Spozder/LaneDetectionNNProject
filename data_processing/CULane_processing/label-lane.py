#! /usr/local/bin/python
# test case: /media/jeoker/76A4-7403/dataset/driver_23_30frame/05160835_0469.MP4/04455.jpg
import cv2
import os, re
import numpy as np

root_dir = '/media/jeoker/76A4-7403/dataset'
# root_dir = '/mnt/f/76A4-7403/dataset'

def draw_lane(image, lanes):
    
    height, width, depth = image.shape
    # feature_points = np.array()

    for lane in lanes:
        lane_points = lane.split(' ')
        
        # draw lines
        for i in range( len(lane_points) - 3 ):
            
            if i % 2:
                cur_line = []
                cur_line.append( int(float(lane_points[i-1])) )
                cur_line.append( int(float(lane_points[i])) )
                cur_line.append( int(float(lane_points[i+1])) )
                cur_line.append( int(float(lane_points[i+2])) )
                cv2.line(image, (cur_line[0],cur_line[1]), (cur_line[2],cur_line[3]), 200, 1)
                
    # fill out the whole lane area
    for y in range(height):
        for x1 in range(width):
            if image[y,x1] != 0:
                for x2 in range(x1+1, width):
                    if image[y,x2] != 0:
                        cv2.line(image, (x1,y), (x2,y), 200, 1)
                break
    

    return image

def file_walk(file):
    count = 0
    image_paths = open(file).read().splitlines()
    for image_path in image_paths:
        work_file = root_dir + image_path
        line_file = work_file[:-4] + '.lines.txt'

        if os.path.isfile(work_file) and os.path.isfile(line_file):
            count += 1
            # load the image file & create the label image
            img = cv2.imread(work_file)
            height, width, depth = img.shape
        
            # load the lane points & create label files
            lanes = open(line_file).read().splitlines()
            label_file = root_dir + '/labels' + image_path

            mid_lane_x = width / 2
            # init output label.jpg file
            label = np.zeros((height,width,1), dtype = np.uint8)
            # we only care about the current lane we are in
            # TODO: use a better algorithm to extract the current lane from image
            # Now: assume the lane on the left to be the current lane
            if len(lanes) == 2:
                label = draw_lane(label, lanes)
                print 'image filled:', image_path, ' | ', count, 'files made'
                print 
            elif len(lanes) > 2:
                for i in range( 1, len(lanes) - 1 ):
                    lane1 = lanes[i-1].split(' ')
                    lane2 = lanes[i].split(' ')
                    if int( float(lane1[0]) ) < mid_lane_x \
                        and int( float(lane2[0]) ) > mid_lane_x:
                        label = draw_lane(label, lanes[i-1:i+1])
                        print 'image filled:', image_path, ' | ', count, 'files made'
                        print 
                        break

            # save label images into the target pathes
            label_dir = label_file[:-9]
            if not os.path.isdir(label_dir):
                os.system('mkdir -p ' + label_dir)

            cv2.imwrite(label_file, label)



if __name__ == '__main__':
    # try:
        index_file = root_dir + '/list/train.txt'
        file_walk(index_file)

    # except Exception as e:
    #     print e
    #     pass
        
    # finally:
    #     pass
    
