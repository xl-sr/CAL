"""Class used for operating the city map."""

import math
import os
import scipy.misc
import numpy as np
from scipy import spatial
import bcolz

def string_to_floats(string):
    vec = string.split(',')
    return (float(vec[0]), float(vec[1]), float(vec[2]))

def load_array(fname): 
    return bcolz.open(fname)[:]

class Centerlines(object):

    def __init__(self, city, pixel_density=16.43, node_density=50.0):
        dir_path = os.path.dirname(__file__)        
        self._pixel_density = pixel_density
        
        # Load the centerlines image and set straight as default
        self.load_centerlines(dir_path, city)
        self.set_centerlines('straight')
        # read in the directions map
        directions_path = os.path.join(dir_path, city + '/centerline_directions.png')
        self.centerline_directions = scipy.misc.imread(directions_path, flatten=False, mode='RGB')

    def get_directions(self, position):
        """
        everytime a colored deciscion border is crossed a decision is needed
        the possibilites are encoded in color
        red == 0 (straight)
        green == 1 (right)
        blue == -1 (left)
        If the function returns an empty set, no decision is needed
        """
        # default list of possible directions
        directions = set()

        # get the RGB values in the detected pixel
        r, g, b = self.centerline_directions[position[1], position[0],:]

        # set the directions according to the pixel values
        if r==255: directions.add(0)
        if g==255: directions.add(1)
        if b==255: directions.add(-1)

        return directions

    ### extra functions to get the centerline GT
    def load_centerlines(self, dir_path, city):
        """
        load the centerline images
        """
        # for straight streets
        file_lines = os.path.join(dir_path, city, 'centerlines_straight')
        self.straight, self.straight_grad = load_array(file_lines)

        # for curves of type 1
        file_lines = os.path.join(dir_path, city, 'centerlines_c1')
        self.c1, self.c1_grad = load_array(file_lines)

        # for curves of type 2
        file_lines = os.path.join(dir_path, city, 'centerlines_c2')
        self.c2, self.c2_grad = load_array(file_lines)
        
    def set_centerlines(self, street):
        """
        set the centerline image according to the given direction
        street is a string: either 'straight', 'c1' or 'c2'
        """
        if street == 'straight':
            self.centerlines, self.centerline_gradients = self.straight, self.straight_grad

        elif street == 'c1':
            self.centerlines, self.centerline_gradients = self.c1, self.c1_grad

        elif street == 'c2':
            self.centerlines, self.centerline_gradients = self.c2, self.c2_grad

        else:
            print('Street type {} does not exist'.format(street))

    def get_center_distance(self, pixel):
        """
        returns the absolute distance to the center line
        """
        # load the centerline image as array
        # switch x and y values, take only R values (of RGB image)
        centerlines_detected = np.copy(self.centerlines)
        # uncomment next line to plot the FOV
        # centerlines_detected = np.ones_like(self.centerlines[:,:,0])
        centerlines_detected = np.swapaxes(centerlines_detected, 0, 1)
        map_size = centerlines_detected.shape

        # params
        width = 80 # width of the detection rectangle
        length = 80 # length of the detection rectangle
        x_min, x_max = -1, map_size[0]
        y_min, y_max = -1, map_size[1]

        # defaults
        pixel = np.array([pixel])
        line_idcs = np.array([])
        distances = []
        closest_idx = 0
        dist = np.array([999.99])

        # correct center dimension
        center = tuple(np.squeeze(pixel))

        # get unrotated corner cordinates around origin in (0,0)
        x_start, x_end = center[0] - width, center[0] + width
        y_start, y_end = center[1] - length, center[1] + length

        # the mask is the FOV around the car
        X,Y = np.meshgrid(np.arange(x_start, x_end), np.arange(y_start, y_end))

        # the mask is the FOV around the car
        FOV_idcs = np.vstack([X.ravel(), Y.ravel()])

        # restrict to map size
        # limit x values
        FOV_idcs = FOV_idcs[:,FOV_idcs[0,:] > x_min]
        FOV_idcs = FOV_idcs[:,FOV_idcs[0,:] < x_max]
        # limit y values
        FOV_idcs = FOV_idcs[:,FOV_idcs[1,:] > y_min]
        FOV_idcs = FOV_idcs[:,FOV_idcs[1,:] < y_max]

        # only pixels inside of the FOV are interesting
        # get the values of the map inside of the FOV
        # is_line = vector, boolean for every point in FOV
        n_points = FOV_idcs.shape[1]
        # get the values of the center line in scope
        val_scope = centerlines_detected[FOV_idcs[0,:],FOV_idcs[1,:]].reshape(1,n_points)
        is_line = ~np.isnan(val_scope) #convert to boolean
        line_idcs = FOV_idcs[:, is_line[0,:]].transpose()

        # get the closest point to the ego vehicle as idx of line_idcs
        if line_idcs.shape[0]:
            distances = spatial.distance.cdist(line_idcs, pixel)
            closest_idx = np.argmin(distances)
            dist = distances[closest_idx]

        return dist[0]/100*self._pixel_density  # distance in m
