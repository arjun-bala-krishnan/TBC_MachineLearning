import numpy as np
import os
import cv2
import re
import xlrd
import math
import datetime
import sys
import time
import imutils

from matplotlib import pyplot as plt
import matplotlib.mlab as mlab
import pandas as pd

from shutil import copyfile, rmtree
from scipy import interpolate
from PIL import Image

# curr_dir = os.getcwd()
# image_dir = os.path.join(curr_dir, "Image_saved")
# img = os.path.join(image_dir, "#img.tif")
# img2 = os.path.join(image_dir, "#img2.tif")
# # read image
# img_orig = cv2.imread(img, cv2.IMREAD_UNCHANGED)
# img2_orig = cv2.imread(img2, cv2.IMREAD_UNCHANGED)
# # global threshold
# ret, img_th = cv2.threshold(img_orig, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# ret2, img2_th = cv2.threshold(img2_orig, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# # path to processed 500x image - img
# path_img_slice_orig = os.path.join(image_dir, "img_slice_orig.tif")
# path_img_slice_th = os.path.join(image_dir, "img_slice_th.tif")
# path_img_op0 = os.path.join(image_dir, "img_op0.tif")
# path_img_op1 = os.path.join(image_dir, "img_op1.tif")
# path_img_op2 = os.path.join(image_dir, "img_op2.tif")
# path_img_op3 = os.path.join(image_dir, "img_op3.tif")

# # path to processed 2000x image - img2
# path_img2_slice_orig = os.path.join(image_dir, "img2_slice_orig.tif")
# path_img2_slice_th = os.path.join(image_dir, "img2_slice_th.tif")
# path_img2_op0 = os.path.join(image_dir, "img2_op0.tif")
# path_img2_op1 = os.path.join(image_dir, "img2_op1.tif")
# path_img2_op2 = os.path.join(image_dir, "img2_op2.tif")
# path_img2_op3 = os.path.join(image_dir, "img2_op3.tif")


# def initialize_slice(img1, img2, x_img1 = 0, y_img1 = 0, x_img2 = 0, y_img2 = 0, slice_sq_dim = 224):
    
#     x1_img1 = x_img1 - int( slice_sq_dim / 2 )
#     y1_img1 = y_img1 - int( slice_sq_dim / 2 ) 
#     x2_img1 = x_img1 + int( slice_sq_dim / 2 )
#     y2_img1 = y_img1 + int( slice_sq_dim / 2 )
    
#     img1_slice = img1[y1_img1:y2_img1, x1_img1:x2_img1]
    
#     x1_img2 = x_img2 - int( slice_sq_dim / 2 )
#     y1_img2 = y_img2 - int( slice_sq_dim / 2 ) 
#     x2_img2 = x_img2 + int( slice_sq_dim / 2 )
#     y2_img2 = y_img2 + int( slice_sq_dim / 2 )
    
#     img2_slice = img2[y1_img2:y2_img2, x1_img2:x2_img2]
    
#     return img1_slice, img2_slice


def view1_image(imageA):
    fig = plt.figure(figsize=(18, 16), dpi= 80, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(imageA, cmap = plt.cm.gray)
    plt.axis("off")
    plt.show()

        
def view2_image(imageA, imageB):
    fig = plt.figure(figsize=(18, 16), dpi= 80, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(imageA, cmap = plt.cm.gray)
    plt.axis("off")
    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(imageB, cmap = plt.cm.gray)
    plt.axis("off")
    plt.show()

    
def get_nearest_neighbours(img, pos_i, pos_j, pix_block_size):
    top_NN_i = []
    top_NN_j = []
    left_NN_i = []
    left_NN_j = []
    bottom_NN_i = []
    bottom_NN_j = []
    right_NN_i = []
    right_NN_j = []
    
    for i in range(pix_block_size+2):
        top_NN_i.append(pos_i - 1)
        left_NN_j.append(pos_j - 1)
        bottom_NN_i.append(pos_i + pix_block_size)
        right_NN_j.append(pos_j + pix_block_size)
         
    for i in range(-1,pix_block_size+1,1):
        top_NN_j.append(pos_j + i)
        left_NN_i.append(pos_i + i)
        bottom_NN_j.append(pos_j + i)
        right_NN_i.append(pos_i + i)
        
    top_NN = []
    left_NN = []
    bottom_NN = []
    right_NN = []
    
    for i in range(pix_block_size+2):
        top_NN.append((top_NN_i[i],top_NN_j[i]))
        left_NN.append((left_NN_i[i],left_NN_j[i]))
        bottom_NN.append((bottom_NN_i[i],bottom_NN_j[i]))
        right_NN.append((right_NN_i[i],right_NN_j[i]))
        
    # convert all tuples into list to make it immutable     
    top_NN = list(top_NN)
    left_NN = list(left_NN)
    bottom_NN = list(bottom_NN)
    right_NN = list(right_NN)   
    
    coords = top_NN
    coords = [coord for coord in coords if not any(number < 0 for number in coord)]
    top_NN = coords
    
    coords = left_NN
    coords = [coord for coord in coords if not any(number < 0 for number in coord)]
    left_NN = coords
    
    coords = right_NN
    coords = [coord for coord in coords if not any(number < 0 for number in coord)]
    right_NN = coords
    
    coords = bottom_NN
    coords = [coord for coord in coords if not any(number < 0 for number in coord)]
    bottom_NN = coords   
    
    coords = top_NN
    top_NN = []
    for coord in coords:
        if not coord[0] >= img.shape[0] and not coord[1] >= img.shape[1]:
            top_NN.append(coord) 
            
    coords = left_NN
    left_NN = []
    for coord in coords:
        if not coord[0] >= img.shape[0] and not coord[1] >= img.shape[1]:
            left_NN.append(coord) 
            
    coords = right_NN
    right_NN = []
    for coord in coords:
        if not coord[0] >= img.shape[0] and not coord[1] >= img.shape[1]:
            right_NN.append(coord) 
            
    coords = bottom_NN
    bottom_NN = []
    for coord in coords:
        if not coord[0] >= img.shape[0] and not coord[1] >= img.shape[1]:
            bottom_NN.append(coord) 
    
    NN_List = top_NN + left_NN + bottom_NN + right_NN    
    NN_List = list( dict.fromkeys(NN_List) )

    return NN_List


#     def calc_min_distance(self, pos, max_row, max_col):
#         horiz_plus = 1
#         horiz_minus = 1
#         vert_plus = 1
#         vert_minus = 1
#         max_thickness = 3
        
#         while pos[1] + horiz_plus < max_col + 1:       
#             if pos[1] + horiz_plus > self.img.shape[1]:
#                 break
            
#             if self.img[pos[0], pos[1] + horiz_plus] == 255:
#                 break    
                
#             horiz_plus = horiz_plus + 1

#         while pos[1] - horiz_minus >= 0:
#             if self.img[pos[0], pos[1] - horiz_minus] == 255:
#                 break   
                
#             horiz_minus = horiz_minus + 1
            
#         while pos[0] - vert_minus >= 0: 
#             if self.img[pos[0] - vert_minus, pos[1]] == 255:
#                 break  
                
#             vert_minus = vert_minus + 1

#         while pos[0] + vert_plus < max_row + 1:
#             if pos[0] + vert_plus > self.img.shape[0]:
#                 break
            
#             if self.img[pos[0] + vert_plus, pos[1]] == 255:
#                 break
            
#             vert_plus = vert_plus + 1
                 
#         vert_thickness = vert_plus + vert_minus
#         horiz_thickness = horiz_plus + horiz_minus   
#         thickness = min(horiz_thickness, vert_thickness)
        
#         thickness_midpoint = False
        
#         if thickness == horiz_thickness:
#             if horiz_plus == horiz_minus and horiz_plus>1:
#                 thickness_midpoint = True
                
#         if thickness == vert_thickness:
#             if vert_plus == vert_minus and vert_plus > 1:
#                 thickness_midpoint = True
        
#         if thickness < max_thickness:
#             if thickness_midpoint is False:
#                 self.thread_cluster.append(pos)
#                 i = -1 * max_thickness
#                 while i < max_thickness:
#                     if pos[1] + i < max_col: 
#                         if self.img[pos[0], pos[1] + i] == 1:
#                             self.thread_cluster.append((pos[0], pos[1] + i))
#                     i += 1

#                 i = -1 * max_thickness
#                 while i < max_thickness:
#                     if pos[0] + i < max_row: 
#                         if self.img[pos[0] + i, pos[1]] == 1:
#                             self.thread_cluster.append((pos[0] + i, pos[1]))
#                     i += 1
    
#     def connected_cluster_seggragation(self, img):
#         self.thread_cluster = []
#         thread_cluster_temp = []
#         self.img = img
#         max_row = 0
#         max_col = 0
#         for coord in self.shape_cluster:
#             if self.img[coord] == 0:
#                 self.img[coord] = 1
            
#             if coord[0] > max_row:
#                 max_row = coord[0]
                
#             if coord[1] > max_col:
#                 max_col = coord[1]
        
#         for coord in self.shape_cluster:
#             self.calc_min_distance(coord, max_row, max_col)
            
#         for coord in self.thread_cluster:
#             mark_possibility_NN1 = False
#             mark_possibility_NN2 = False
#             possible_pixels = get_nearest_neighbours(self.img, coord[0], coord[1], 1)           
#             for possible_cord in possible_pixels:
#                 if possible_cord in self.thread_cluster:
#                     mark_possibility_NN1 = True
            
#             if coord[0] > 0 and coord[1] > 0:
#                 possible_pixels = get_nearest_neighbours(self.img, coord[0]-1, coord[1]-1, 2)           
#                 for possible_cord in possible_pixels:
#                     if possible_cord in self.thread_cluster:
#                         mark_possibility_NN2 = True
                        
#             else:
#                 # Since the pixel is in the edge
#                 mark_possibility_NN2 = False
                    
#             # To check if the thread is atleast 3 pixels long
#             if mark_possibility_NN1 is True and mark_possibility_NN2 is True:
#                 thread_cluster_temp.append(coord)
          
#         if thread_cluster_temp:
#             for elem in thread_cluster_temp:
#                 if self.img[elem] == 1:
#                     self.img[elem] = 2

#         self.thread_cluster = thread_cluster_temp
#         return self.img
                
    
#     def join_broken_threads(self, critical_area = 4):
#         # Current cluster in focus is not a noise        
#         if int(len(self.shape_cluster)) >= critical_area:
#             for coord in self.shape_cluster:
#                 if coord[0] - 1 >= 0 and coord[1] - 1 >= 0:
#                     pos_i = coord[0] - 1
#                     pos_j = coord[1] - 1
#                     possible_points = get_nearest_neighbours(self.img, pos_i, pos_j, 3)

#                     for possible_coord in possible_points:
#                         if (self.img[possible_coord] == 1 or self.img[possible_coord] == 0):
#                             if possible_coord not in self.shape_cluster:
#                                 self.img = cv2.line(self.img, (coord[1], coord[0]), (possible_coord[1], possible_coord[0]), (2, 2, 2), thickness = 1, lineType=8)


#             for coord in self.shape_cluster:
#                 if coord[0] - 2 >= 0 and coord[1] - 2 >= 0:
#                     pos_i = coord[0] - 2
#                     pos_j = coord[1] - 2
#                     possible_points = get_nearest_neighbours(self.img, pos_i, pos_j, 5)
#                     for possible_coord in possible_points:
#                         if (self.img[possible_coord] == 1 or self.img[possible_coord] == 0):
#                             if possible_coord not in self.shape_cluster:
#                                 self.img = cv2.line(self.img, (coord[1], coord[0]), (possible_coord[1], possible_coord[0]), (2, 2, 2), thickness = 1, lineType=8)

#         self.img[self.img == 2] = 1
#         return self.img 




def openCVcompare_contour(img, img_orig):
    
    def image_porosity(img):    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if i == 0 or i == img.shape[0]-1:
                img[i,j] = 255
            if j == 0 or i == img.shape[1]-1:
                img[i,j] = 255
            
    pores = cv2.Canny(img, 100, 200)
    view1_image(pores)

    # find contours in the binary image
    cnts, hierarchy = cv2.findContours(pores, cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    por_area = 0
    hierarchy = hierarchy[0]

    for i, c in enumerate(cnts):
        
        M = cv2.moments(c)
        a = cv2.contourArea(c)
        por_area = por_area+a
        cv2.drawContours(pores, cnts, i, (255, 255, 255), 2)
        cv2.fillPoly(pores, pts =[c], color=(255, 255, 255))
#         if hierarchy[i][2] < 0 and hierarchy[i][3] < 0:
#             cv2.drawContours(pores, cnts, i, (255, 255, 255), 2)
#             cv2.fillPoly(pores, pts =[c], color=(250, 250, 250))
#         else:
#             cv2.drawContours(pores, cnts, i, (255, 255, 255), 2)
    
    view1_image(pores)
    print(por_area)
    
    return pores
    
    img_trial = np.copy(img)
    img_porosity = image_porosity(img_trial)
    for i in range(0, img_trial.shape[0]):
        for j in range(0, img_trial.shape[1]):
            block = pixel_block()
            block.show_cluster = True
            block.orig_image = img_orig
            val = block.get_cluster_iteration(img_trial, i, j, True)
           
    img_trial[img_trial == 0] = 255
    img_trial[img_trial == 1] = 0  
    return img_trial, img_porosity



# To detect clusters, join points to closeby clusters/threads, count the remaining points (noises) and eliminate them
def porosity_operation1(img2_op0, img2_slice_orig):
    img_trial = np.copy(img2_op0) 
    area_noise = 0
    val = 0
    image_exist_bit = False
    
    for i in range(0, img_trial.shape[0]):
        for j in range(0, img_trial.shape[1]):
            image_exist_bit = True
            block = pixel_block()
            block.orig_image = img2_slice_orig
            val = block.get_cluster_iteration(img_trial, i, j, True)
            if val == 1:
                point_area_check, img_trial = block.repeat_points_area(img_trial, 6)
    
    if image_exist_bit:
        area_noise = block.count_pixel(img_trial, pixel_color = 0)
    else:
        area_noise = 0
           
    img_trial[img_trial == 0] = 255
    img_trial[img_trial == 1] = 0  
    return img_trial, area_noise


img_op1, area_noise = porosity_operation1(img_op0, img_slice_orig)    
print("img Noise area removed = ", area_noise)
img2_op1, area_noise = porosity_operation1(img2_op0, img2_slice_orig)    
print("img2 Noise area removed = ", area_noise)

# operation 2: Recalculate point area by giving in block size as arguement - also cleans the image    
def porosity_operation2(img_op1, img_orig):
    img_trial = np.copy(img_op1)
    area_points = 0
    val = 0
    for i in range(0, img_trial.shape[0]):
        for j in range(0, img_trial.shape[1]):  
            block = pixel_block()
            block.orig_image = img_orig
            val = block.get_cluster_iteration(img_trial, i, j, True)
            if val == 1:
                point_area_check, img_trial2 = block.repeat_points_area(img_trial, 4)
                if point_area_check is True:
                    area_points = area_points + 1
    img_trial[img_trial == 0] = 255
    img_trial[img_trial == 1] = 0
            
    return img_trial, area_points

img_op2, area_noise = porosity_operation2(img_op1, img_slice_orig)    
print("img Noise area removed = ", area_noise)
img2_op2, area_noise = porosity_operation2(img2_op1, img2_slice_orig)    
print("img2 Noise area removed = ", area_noise)

# operation 3: isolate threads from clusters and count them - Connected-component labeling
def porosity_operation3(img, img_orig):
    img_trial = np.copy(img)
    area_threads = 0
    val = 0
    for i in range(0, img_trial.shape[0]):
        for j in range(0, img_trial.shape[1]):
            block = pixel_block()
            block.orig_image = img_orig
            val = block.get_cluster_iteration(img_trial, i, j, True)
            if val == 1:
                img_trial = block.connected_cluster_seggragation(img_trial)

    img_trial[img_trial == 2] = 150
    area_threads = block.count_pixel(img_trial, 150)
    return img_trial, area_threads

img_op3, area_noise = porosity_operation3(img_op2, img_slice_orig)    
print("img Noise area removed = ", area_noise)
img2_op3, area_noise = porosity_operation3(img2_op2, img2_slice_orig)    
print("img2 Noise area removed = ", area_noise)