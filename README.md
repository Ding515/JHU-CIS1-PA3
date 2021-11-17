# JHU-CIS1-PA3
This is work for CIS1 Fall 2021 PA3 program at JHU.
The user can directly use PA3_demos.py to operate the required operations with a change of input and save path.
In this PA, we assume 2 different types of dataset one is 'debug' and another is 'unknown'.

<1> Report is the experiment report

<2> The program part includes the PA3_demos.py as the main function and it depends on:

  <2.1> PA3_solution.py is the function to solve the practial problem in this PA, including a frame transformation to obtain d_{k}, three different searching methods
  (simple bounding box, simple sphere and KD-Tree structure) and a output file generation function. 
  
  <2.2>PA3_functions.py contains the functional function used in this program, including:
  
  body_surface_loader to load the surface file
  
  FindClosestPoint to find the closest point in a triangle region for a given point
  
  simple_search for matching in bounding box linear search manner
  
  bound_fix to find the bounding box in simple_search function
  
  bounding_sphere for matching in bounding sphere linear search manner
  
  bounding_sphere_compute to compute the centroid and radius for each triangle, used in bounding_sphere
  
  simple_search_kdtree use KD-Tree based structure for matching
  
# Aknowledgment

In this work, we use Python's package numpy, scipy and time and read the KD-Tree WIKI before work. 
