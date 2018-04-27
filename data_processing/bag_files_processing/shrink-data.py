#! /usr/local/bin/python

import os

def shrink_bag(in_folder, ou_folder):
    
    topic_arg = '\"topic==\'/camera_array/cam2/image_raw\'\"'

    for folderName, subfolders, filenames in os.walk(in_folder):
        for filename in filenames:
            
            if filename[-4:] == '.bag': # and filename[0] not in ['0', '1']:
                source_file = in_folder + '/' + filename
                target_file = ou_folder + '/' + filename[:-4] + 's.bag'
                
                os.system(  'rosbag filter ' 
                            + source_file + ' ' 
                            + target_file + ' ' 
                            + topic_arg)


if __name__ == '__main__':
    
    source_dir = '/media/jeoker/Media/ML-Data'
    target_dir = '/media/jeoker/76A4-7403/bags'
    
    try:
        shrink_bag(source_dir, target_dir)
        os.system('python extract-images.py')
    except Exception as e:
        print e
        pass
    else:
        pass
    finally:
        pass
    
