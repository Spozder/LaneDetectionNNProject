#! /usr/local/bin/python

import os
import roslaunch

# when the data is dealt in, depends on project folder hierarchy
root_dir = '/media/jeoker/76A4-7403'


def extract_images(source_dir, input_file):
    target_dir = root_dir + '/images'

    ouput_file = '%04d.jpg'
    ouput_file = input_file[:-4] + '_' + ouput_file

    uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
    roslaunch.configure_logging(uuid)

    ros_args = ['', # rospackage name
                'extract-images.launch', # launch file name
                'SOURCE_DIR:="' + source_dir + '"', # where to find the bag file
                'TARGET_DIR:="' + target_dir + '"', # where the images go
                'INPUT_FILE:="' + input_file + '"', # bag file name
                'OUPUT_FILE:="' + ouput_file + '"'] # images name regulation
    
    print '\t start working on: ' + input_file

    # run it in command line
    os.system(  'roslaunch ' + 
                ros_args[0] + ' ' + 
                ros_args[1] + ' ' + 
                ros_args[2] + ' ' + 
                ros_args[3] + ' ' + 
                ros_args[4] + ' ' + 
                ros_args[5])

# find bag files to extract images recursively
def recursive_walk(folder):
    for folderName, subfolders, filenames in os.walk(folder):
        if subfolders:
            for subfolder in subfolders:
                if subfolder not in (''):
                    print 'Now searching folder: ' + subfolder + '\n'
                    recursive_walk(subfolder)
        
        for filename in filenames:
            if filename[-4:] == '.bag':
                extract_images(folderName, filename)



if __name__ == '__main__':

    source_dir = root_dir + '/bags'
    try:
        recursive_walk(source_dir)

    except Exception as e:
        print e
        pass
    else:
        pass
    finally:
        pass
    
