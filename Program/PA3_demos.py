# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 12:10:18 2021

@author: Ding,Li
"""
'''
This is DEMO file of our Programming Asignment 3 in CIS course, Fall 2021.
In this part, several works are included:
    Load the body surface file of the probe rigid body
    
    Work out d_{k} based on the frame relations
    
    By assuming F_{reg} = I, we construct 3 different searching method to search for nearest
    point in surface, including simple bounding search, simple sphere search 
    and KDTree searching method.
    
'''
import PA3_functions as PA3F
import PA3_solution as PA3S
import numpy as np
import time 
'''
Some general input of demos:
    data_load_path: The path of all PA3 data
    file_dict:the name of different data set, from A to J
    data_status: 'debug' and 'unknown'
    method: the included method for searching the nearest point, including
    'bounding_simple','bounding_sphere' and 'KDTree'
'''
######## DATA LOADING PART #############
data_load_path='.\2021_pa_3-5_student_data\2021 PA 3-5 Student Data\'
file_dict=['A','B','C','D','E','F']
data_status='Debug'
#file_dict=['J','H','G']
#data_status='Unknown'
method='KDTree'
result_save_path='..\output\'
######## SURFACE LOADING #############
'''
body_surface_loader function loads the body surface:
    Input: surface_file_path: The loading path of surface file
    
    Output:
        vertices: An Nx3 array, where each row is a vertice XYZ-coordinate
        
        triangle: An Nx3 array, where each row corresponding to the index of three
        vertices in variable 'vertices'
'''
vertices, triangles=PA3F.body_surface_loader(data_load_path+'Problem3Mesh.sur')
time_start=time.time()
##############SOLVING d_{K} ########################
N_A_marker=np.loadtxt(data_load_path+'Problem3-BodyA.txt',skiprows=1).shape[0]-1
N_B_marker=np.loadtxt(data_load_path+'Problem3-BodyB.txt',skiprows=1).shape[0]-1
N_point=16
for file_name in file_dict:
    samples=np.loadtxt(data_load_path+'PA3-'+file_name+'-'+data_status+'-SampleReadingsTest.txt',skiprows=1,delimiter=',')
    A_marker=np.loadtxt(data_load_path+'Problem3-BodyA.txt',skiprows=1)
    B_marker=np.loadtxt(data_load_path+'Problem3-BodyB.txt',skiprows=1)
    tip_save_path=result_save_path+file_name+'_tip_coordinate.npy'
    
    """
    step_1_rigid_registration function returns the position of pointer tip with respect to
    rigid body B
        Input:
            samples:  the SampleReadings file, in this file including the A
            body LED markers in tracker coordinates, B body LED markers in 
            tracker coordinates in different frames.
            
            A_marker:coordinates of marker LEDs in A body coordinates
            
            B_marker:coordinates of marker LEDs in B body coordinates
            
            N_frames: number of frames in samples, a default value is 15
            tip_save_path: the path to save d_{k}
            
        Output:
            tip_positions: the position of d_{k}
    """
    tip_positions=PA3S.step_1_rigid_registration(samples, A_marker,B_marker, N_frames= 15, tip_save_path=tip_save_path)
#################### SEARCHING FOR CLOSEST POINT PART ##########################    
    '''
    3 different method for searching nearest points in the surface and the general input
    for them:
        Input:
            tip_posiitons: the original positions of points to be searched with
            
            vertices: An Nx3 array, where each row is a vertice XYZ-coordinate
        
            triangle: An Nx3 array, where each row corresponding to the index of three
            vertices in variable 'vertices'
            
        Output:
           tip_positions_transformed: the found point positions in surface 
    '''
    if method == 'bounding_simple' : 
        tip_positions_transformed=PA3S.step_2_simple_search(tip_positions, vertices, triangles)
    if method == 'bounding_sphere':    
        centroid_path=result_save_path+file_name+'_centroid.npy'
        radius_path=result_save_path+file_name+'_radius.npy'    
        closest_point_path=result_save_path+file_name+'_closest_point_coordinate.npy'
        tip_positions_transformed=PA3S.step_2_simple_search_sphere(tip_positions, vertices, triangles,centroid_path,radius_path,closest_point_path)
    if method == 'KDTree':    
        tip_positions_transformed=PA3S.step_2_kd_tree_search(tip_positions, vertices, triangles)
        
    output_path=result_save_path+file_name+'_answer.txt'
    title=str(tip_positions_transformed.shape[0])+' '+ 'pa3-'+file_name+'-Output.txt'
    PA3S.output_generate(tip_positions,tip_positions_transformed,output_path,title)

time_end=time.time()

print(time_end-time_start)    
