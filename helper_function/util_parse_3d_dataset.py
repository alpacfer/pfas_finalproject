"""
Parse KITTI 3D object dataset into separate images for each object in each frame
and get associated labels
"""
from cmath import rect
import os
import sys
import time
import json
import matplotlib.pyplot as plt
import numpy as np
import copy

from scipy.datasets import face
import _pickle as pickle

import cv2
import PIL
from PIL import Image
from math import cos,sin
from util_load import get_coords_3d, plot_text, pil_to_cv,draw_prism
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms
from cv_lib import *
from matplotlib.patches import Rectangle

import re

def parse_calibration_file(file_path,raw=False):
    calibration_data = {}

    if raw == False:
    
        with open(file_path, 'r') as file:
            lines = file.readlines()
            
            for line in lines:
                if line.strip():  # Skip empty lines
                    key, values = line.split(':', 1)
                    values_list = [float(value) for value in values.split()]
                    calibration_data[key.strip()] = values_list

    else:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            
            for line in lines[1:]:
                if line.strip():  # Skip empty lines
                    key, values = line.split(':', 1)
                    values_list = [float(value) for value in values.split()]
                    calibration_data[key.strip()] = values_list

    
    return calibration_data



def parse_label_file2(label_list,idx):
    """parse label text file into a list of numpy arrays, one for each frame"""
    f = open(label_list[idx])
    line_list = []
    for line in f:
        line = line.split()
        line_list.append(line)
        
    # each line corresponds to one detection
    det_dict_list = []  
    for line in line_list:
        
        # det_dict holds info on one detection
        det_dict = {}
        det_dict['class']      = str(line[0])
        if det_dict['class'] == "DontCare":
            continue
        det_dict['truncation'] = float(line[1])
        det_dict['occlusion']  = int(line[2])
        det_dict['alpha']      = float(line[3]) # obs angle relative to straight in front of camera
        x_min = int(round(float(line[4])))
        y_min = int(round(float(line[5])))
        x_max = int(round(float(line[6])))
        y_max = int(round(float(line[7])))
        det_dict['bbox2d']     = np.array([x_min,y_min,x_max,y_max])
        length = float(line[10])
        width = float(line[9])
        height = float(line[8])
        det_dict['dim'] = np.array([length,width,height])
        x_pos = float(line[11])
        y_pos = float(line[12])
        z_pos = float(line[13])
        det_dict['pos'] = np.array([x_pos,y_pos,z_pos])
        det_dict['pos_rr'] = np.array([x_pos,z_pos,-y_pos])
        det_dict['rot_y'] = float(line[14])
        det_dict_list.append(det_dict)
    
    return det_dict_list



def parse_labelExt_file2(label_list,idx):
    """parse label text file into a list of numpy arrays, one for each frame"""
    f = open(label_list[idx])
    line_list = []
    for line in f:
        line = line.split()
        line_list.append(line)
        
    # each line corresponds to one detection
    det_dict_list = []  
    for line in line_list:
        # det_dict holds info on one detection
        det_dict = {}
        det_dict['class']      = str(line[2])
        det_dict['truncation'] = float(line[3])
        det_dict['occlusion']  = int(line[4])
        det_dict['alpha']      = float(line[5]) # obs angle relative to straight in front of camera
        x_min = int(round(float(line[6])))
        y_min = int(round(float(line[7])))
        x_max = int(round(float(line[8])))
        y_max = int(round(float(line[9])))
        det_dict['bbox2d']     = np.array([x_min,y_min,x_max,y_max])
        length = float(line[12])
        width = float(line[11])
        height = float(line[10])
        det_dict['dim'] = np.array([length,width,height])
        x_pos = float(line[13])
        y_pos = float(line[14])
        z_pos = float(line[15])
        det_dict['pos'] = np.array([x_pos,y_pos,z_pos])
        det_dict['rot_y'] = float(line[16])
        det_dict_list.append(det_dict)
    
    return det_dict_list


def parse_calib_file(calib_list,idx):
    """parse calib file to get  camera projection matrix"""
    f = open(calib_list[idx])
    line_list = []
    for line in f:
        line = line.split()
        line_list.append(line)
    line = line_list[2] # get line corresponding to left color camera
    vals = np.zeros([12])
    for i in range(0,12):
        vals[i] = float(line[i+1])
    calib = vals.reshape((3,4))
    return calib

    
def plot_bbox_2d2(im,det):
    """ Plots rectangular bbox on image and returns image
    im - cv2 or PIL style image (function converts to cv2 style image)
    label - for one det_dict, in the form output by parse_label_file 
    bbox_im -  cv2 im with bboxes and labels plotted
    """
    
    # check type and convert PIL im to cv2 im if necessary
    assert type(im) in [np.ndarray, PIL.Image.Image,PIL.PngImagePlugin.PngImageFile], "Invalid image format"
    
    if type(im) in [PIL.PngImagePlugin.PngImageFile,PIL.Image.Image]:
        im = pil_to_cv(im)
    cv_im = im.copy() 
    
    class_colors = {
            'Cyclist': (255,150,0),
            'Pedestrian':(255,100,0),
            'Person':(255,50,0),
            'Car': (0,255,150),
            '2': (0,255,100),
            'Truck': (0,255,50),
            'Tram': (0,100,255),
            'Misc': (0,50,255),
            'DontCare': (200,200,200)}
    
    bbox = det['bbox2d']
    cls = det['class']
    
    cv2.rectangle(cv_im,(bbox[0],bbox[1]),(bbox[2],bbox[3]), (255,0,0), 1)
    if cls != 'DontCare':
        # plot_text(cv_im,(bbox[0],bbox[1]),det['occlusion'],det['pos'][2])
        pass
    return cv_im 

def plot_bbox_3d2(im,det, style = "normal"):
    """ Plots rectangular prism bboxes on image and returns image
    im - cv2 or PIL style image (function converts to cv2 style image)
    label - for one frame, in the form output by parse_label_file
    P - camera calibration matrix
    bbox_im -  cv2 im with bboxes and labels plotted
    style - string, "ground_truth" or "normal"  ground_truth plots boxes as white
    """
        
    # check type and convert PIL im to cv2 im if necessary
    assert type(im) in [np.ndarray, PIL.Image.Image,PIL.PngImagePlugin.PngImageFile], "Invalid image format"
    
    if type(im) in [PIL.PngImagePlugin.PngImageFile,PIL.Image.Image]:
        im = pil_to_cv(im)
    cv_im = im.copy()
    
    class_colors = {
            'Cyclist': (255,150,0),
            'Pedestrian':(200,800,0),
            'Person':(160,30,0),
            '1': (0,255,150),
            '2': (0,255,100),
            'Truck': (0,255,50),
            'Tram': (0,100,255),
            'Misc': (0,50,255),
            'DontCare': (200,200,200)}
    
    cls = det['class']
    if cls != "DontCare":
        bbox_3d = det['bbox3d']
        if style == "ground_truth": # for plotting ground truth and predictions
            cv_im = draw_prism(cv_im,bbox_3d,(255,255,255))
        else:
            cv_im = draw_prism(cv_im,bbox_3d,(0,255,0))
            # plot_text(cv_im,(bbox_3d[0,4],bbox_3d[1,4]),cls,0,class_colors)
    return cv_im 

def plot_proj_point(K,det,center=True):
    """plot a 3D point on an image given the camera calibration matrix K"""
    det_copy = copy.deepcopy(det)
    point = det_copy['pos'].reshape((-1,1))
    # point = np.vstack((point, np.array([1])))
    if not center:
        point[1] -= (det_copy['dim'][2]/2)
    proj = np.dot(K,point)
    proj = proj/proj[2]
    # plt.scatter(proj[0],proj[1],c='g',s=5)

    proj = Pi(proj)

    return (int(round(proj[0].item())),int(round(proj[1].item())))

def construct_camera_matrix_from_file(file_path):
    # Read JSON data from the file
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    # Extract the intrinsic matrix K from the data
    K = np.array(data['Sensor']['Camera']['K'])
    
    return K

def plot_bev2(det,axs,color):

    for obj in det:
        if obj['class'] != "DontCare":
            angle = np.degrees(obj['rot_y'] + np.pi/2) # Rotation angle in degrees
            xy = (obj['pos'][0],obj['pos'][2])  # Center of the rectangle
            w = obj['pos'][1]  # Width of the rectangle
            h = obj['pos'][0]  # Length of the rectangle
            BL = (xy[0]-w/2, xy[1]-h/2)
            rect = Rectangle(BL, width=w, height=h, angle=angle,rotation_point=xy,edgecolor=color,facecolor='none')
            axs.add_patch(rect)



def plot_bev(det,axs,color):
    # Function to add a rectangle with a specified rotation
    def add_rotated_rectangle(ax, xy, width, height, angle, **kwargs):
        xy = (0,0)
        BL = (xy[0]-width/2, xy[1]-height/2)
        angle = 90
        rectangle = Rectangle(BL, width=width, height=height,angle=angle,rotation_point='center')
        ax.add_patch(rectangle)

    def add_fov_lines(ax, x_center, y_center, fov_angle_degrees, fov_length, **kwargs):
        fov_angle_radians = np.deg2rad(fov_angle_degrees / 2)
        
        # Left FOV line
        left_x = x_center - fov_length * np.sin(fov_angle_radians)
        left_y = y_center + fov_length * np.cos(fov_angle_radians)
        ax.plot([x_center, left_x], [y_center, left_y], **kwargs)
        
        # Right FOV line
        right_x = x_center + fov_length * np.sin(fov_angle_radians)
        right_y = y_center + fov_length * np.cos(fov_angle_radians)
        ax.plot([x_center, right_x], [y_center, right_y], **kwargs)

    # # Add rectangles with different rotations
    # # Set FOV parameters
    # x_center = 0
    # y_center = 0
    # fov_angle_degrees = 10  # FOV angle in degrees
    # fov_length = 999  # Length of the FOV lines

    # # Function to add FOV lines
    # add_fov_lines(axs, x_center, y_center, fov_angle_degrees, fov_length, linestyle='--', color='k')
    # add_fov_lines(axs, x_center, y_center, angle_degrees, fov_length, linestyle='--', color='k')
    # Add rectangles with different rotations
    pose = det['pos'].flatten()
    dim = det['dim'].flatten()
    rotation=-det['rot_y'] + np.pi/2
    l = dim[0]
    w = dim[1]
    add_rotated_rectangle(axs, (pose[0], pose[2]), width=w, height=l, angle=rotation, edgecolor=color, facecolor='none')

def cv2_plot_bev(dets,color=(255,0,0) ,shape = (500,500,3)):

    for obj in dets:
        # Define the rectangle parameters
        center = (obj['pos'][0],obj['pos'][2])  # Center of the rectangle
        width = obj['pos'][1]  # Width of the rectangle
        height = obj['pos'][0]  # Height of the rectangle
        angle = np.degrees(obj['rot_y']) # Rotation angle in degrees

        # Create an image
        img = np.ones(shape, dtype=np.uint8) * 255

        # Define the rectangle points
        rectangle = np.array([
            [center[0] - width // 2, center[1] - height // 2],
            [center[0] + width // 2, center[1] - height // 2],
            [center[0] + width // 2, center[1] + height // 2],
            [center[0] - width // 2, center[1] + height // 2]
        ])

        # Compute the rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Apply the rotation to each point
        rotated_rectangle = np.array([cv2.transform(np.array([[pt]]), M)[0][0] for pt in rectangle])

        # Draw the rotated rectangle
        cv2.polylines(img, [rotated_rectangle.astype(np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)
        return img

    
def write_label_file2(objs,path,idx,zfill=6):
    """write label file from list of numpy arrays"""
    f = open(f'{os.path.join(path,str(idx).zfill(zfill))}.txt','w')
    for obj in objs:
        if obj['class'] != "Pedestrian" and obj['occlusion']!=0:
            continue
        line = f"{obj['class']} {obj['truncation']:.2f} {obj['occlusion']} {obj['alpha']:.2f} {obj['bbox2d'][0]:.2f} {obj['bbox2d'][1]:.2f} {obj['bbox2d'][2]:.2f} {obj['bbox2d'][3]:.2f} {obj['dim'][2]:.2f} {obj['dim'][1]:.2f} {obj['dim'][0]:.2f} {obj['pos'][0]:.2f} {obj['pos'][1]:.2f} {obj['pos'][2]:.2f} {obj['rot_y']:.2f}"
        f.write(line + "\n")


def read_pfm(path):
    """Read pfm file.

    Args:
        path (str): path to file

    Returns:
        tuple: (data, scale)
    """
    with open(path, "rb") as file:

        color = None
        width = None
        height = None
        scale = None
        endian = None

        header = file.readline().rstrip()
        if header.decode("ascii") == "PF":
            color = True
        elif header.decode("ascii") == "Pf":
            color = False
        else:
            raise Exception("Not a PFM file: " + path)

        dim_match = re.match(r"^(\d+)\s(\d+)\s$", file.readline().decode("ascii"))
        if dim_match:
            width, height = list(map(int, dim_match.groups()))
        else:
            raise Exception("Malformed PFM header.")

        scale = float(file.readline().decode("ascii").rstrip())
        if scale < 0:
            # little-endian
            endian = "<"
            scale = -scale
        else:
            # big-endian
            endian = ">"

        data = np.fromfile(file, endian + "f")
        shape = (height, width, 3) if color else (height, width)

        data = np.reshape(data, shape)
        data = np.flipud(data)

        return data, scale
##############################################################################    


# if sys.platform == 'linux':
#     image_dir = "/media/worklab/data_HDD/cv_data/KITTI/3D_object/Images/training/image_2"
#     label_dir = "/media/worklab/data_HDD/cv_data/KITTI/3D_object/Detection_Labels/data_object_label_2/training/label_2"
#     calib_dir = "/media/worklab/data_HDD/cv_data/KITTI/3D_object/calib/training/calib"
#     new_dir =   "/media/worklab/data_HDD/cv_data/KITTI/3D_object_parsed"
# else:
#     # image_dir = "C:\\Users\\derek\\Desktop\\KITTI\\3D_object\\Images\\training\\image_2"
#     # label_dir = "C:\\Users\\derek\\Desktop\\KITTI\\3D_object\\Detection_Labels\\data_object_label_2\\training\\label_2"
#     # calib_dir = "C:\\Users\\derek\\Desktop\\KITTI\\3D_object\\data_object_calib\\training\\calib"
#     # new_dir =   "C:\\Users\\derek\\Desktop\\KITTI\\3D_object_parsed"
#     image_dir = "C:\\Users\\Hasan\\OneDrive\\Desktop\\Projects\\Depth-Anything\\metric_depth\\data\\Kitti\\raw_data\\2011_09_26\\2011_09_26_drive_0013_sync\\image_02\\data"
#     label_dir = "C:/Users/Hasan/Downloads/devkit_raw_data/devkit/data"
#     calib_dir = "C:/Users/Hasan/OneDrive/Desktop/Projects/TestKitti/2011_09_26_calib/2011_09_26"
#     new_dir =   "C:/Users/Hasan/OneDrive/Desktop/Projects/TestKitti"
# buffer = 25

# # create new directory for holding image crops
# if not os.path.exists(new_dir):
#     os.mkdir(new_dir)
#     os.mkdir(os.path.join(new_dir,"images"))

# # stores files for each set of images and each label
# dir_list = next(os.walk(image_dir))[1]
# image_list = [os.path.join(image_dir,item) for item in os.listdir(image_dir)]
# label_list = [os.path.join(label_dir,item) for item in os.listdir(label_dir)]
# calib_list = [os.path.join(calib_dir,item) for item in os.listdir(calib_dir)]
# image_list.sort()
# label_list.sort()
# calib_list.sort()


# out_labels = []
# # loop through images
# for i in range(len(image_list)):
#     if i %50 == 0:
#         print("On image {} of {}".format(i,len(image_list)))
        
#     # get label and cilbration matrix
#     det_dict_list = parse_label_file2(label_list,i)
#     calib = parse_calib_file(calib_list,i)
    
#     # open image
#     im = Image.open(image_list[i])
#     imsize = im.size
#     if False:
#         im.show()
    
#     # loop through objects in frame
#     obj_count = 0
#     for j in range(len(det_dict_list)):
#         det = det_dict_list[j]
#         if det['class'] not in ["dontcare", "DontCare"]:
#             crop = np.zeros(4)
#             crop[0] = max(det['bbox2d'][0]-buffer,0) #xmin, left
#             crop[1] = max(det['bbox2d'][1]-buffer,0) #ymin, top
#             crop[2] = min(det['bbox2d'][2]+buffer,imsize[0]-1) #xmax
#             crop[3] = min(det['bbox2d'][3]+buffer,imsize[1]-1) #ymax
#             det['offset'] = (crop[0],crop[1])    
    
            
#             crop_im = im.crop(crop)
            
#             det['bbox2d'][0] = det['bbox2d'][0] - crop[0]
#             det['bbox2d'][1] = det['bbox2d'][1] - crop[1]
#             det['bbox2d'][2] = det['bbox2d'][2] - crop[0]
#             det['bbox2d'][3] = det['bbox2d'][3] - crop[1]
            
            
#             # save image
#             im_name = "{:06d}-{:02d}.png".format(i,obj_count)
#             crop_im.save(os.path.join(new_dir,"images",im_name))
#             obj_count += 1
        
#             # create label - add calib and bbox3d to det_dict
#             det['calib'] = calib
#             det['bbox3d'],det['depths'],det['cam_space'] = get_coords_3d(det,calib)
            
#             # shift to cropped image coordinates
#             det['bbox3d'][0,:] = det['bbox3d'][0,:] - crop[0]
#             det['bbox3d'][1,:] = det['bbox3d'][1,:] - crop[1]
#             det['im_name'] = im_name
            
#             # replace old det_dict with det in det_dict_list
#             out_labels.append(det)
            
#             # show crop if true
#             if False:
#                 box_im = plot_bbox_3d2(crop_im,det)
#                 cv2.imshow("Frame",box_im)
#                 key = cv2.waitKey(0) & 0xff
#                 #time.sleep(1/30.0)
#                 if key == ord('q'):
#                     break
            
            
# # pickle out_labels
# with open(os.path.join(new_dir,"labels.cpkl"),'wb') as f:
#     pickle.dump(out_labels,f)
# print("Done parsing 3D detection dataset.")

# with open(os.path.join(new_dir,"labels.cpkl"),'rb') as f:
#     loaded_labels = pickle.load(f)