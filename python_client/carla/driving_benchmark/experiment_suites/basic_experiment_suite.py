# Copyright (c) 2017 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.


from __future__ import print_function

from carla.driving_benchmark.experiment import Experiment
from carla.sensor import Camera
from carla.settings import CarlaSettings

from .experiment_suite import ExperimentSuite


class BasicExperimentSuite(ExperimentSuite):

    @property
    def train_weathers(self):
        return [8]
        return [1,3,6,8]

    @property
    def test_weathers(self):
        return []
        return [4,14]

    def build_experiments(self):
        """
            Creates the whole set of experiment objects,
            The experiments created depends on the selected Town.

        """

        # We check the town, based on that we define the town related parameters
        # The size of the vector is related to the number of tasks, inside each
        # task there is also multiple poses ( start end, positions )
        if self._city_name == 'Town01':
            poses_tasks = [[[7, 3]], [[138, 17]], [[140, 134]], [[140, 134]]]
            vehicles_tasks = [0, 0, 0, 20]
            pedestrians_tasks = [0, 0, 0, 50]
        else:
            right_curves =  [[[1,56],[65,69],[78,51],[44,61],[40,17],[71,16],[74,38],[46,12],
                              [19,18],[26,74],[37,76],[11,44],[20,6],[10,22],[28,2],[5,15],
                              [14,33],[34,8]]]      
            left_curves =  [[[57,82],[72,43],[52,79],[70,66],[43,14],[11,47],[79,32],[37,75],
                             [75,16],[26,73],[39,5],[2,37],[34,13],[6,35],[10,19],[23,6],
                             [5,30],[16,2]]] 
            special_test = [[[19, 66], [79, 14],[42, 13], [31, 71], 
                             [54, 30], [10, 61], [66, 3], [27, 12],
                             [2, 29], [16, 14],[70, 73], [46, 67],
                             [51, 81], [56, 65], [43, 54]]]
            corl_task2 = [[[19, 66], [79, 14], [19, 57], [23, 1],
                    [53, 76], [42, 13], [31, 71], [33, 5],
                    [54, 30], [10, 61], [66, 3], [27, 12],
                    [79, 19], [2, 29], [16, 14], [5, 57],
                    [70, 73], [46, 67], [57, 50], [61, 49], [21, 12],
                    [51, 81], [77, 68], [56, 65], [43, 54]]]

            poses_tasks = corl_task2
            vehicles_tasks = [0]*len(poses_tasks[0])
            pedestrians_tasks = [0]*len(poses_tasks[0])

        # We set the camera
        # This single RGB camera is used on every experiment

        camera = Camera('CameraRGB')
        camera.set(FOV=90)
        camera.set_image_size(800, 600)
        camera.set_position(1.44, 0.0, 1.2)
        camera.set_rotation(0, 0, 0)

        # Based on the parameters, creates a vector with experiment objects.
        experiments_vector = []
        for weather in self.weathers:

            for iteration in range(len(poses_tasks)):
                poses = poses_tasks[iteration]
                vehicles = vehicles_tasks[iteration]
                pedestrians = pedestrians_tasks[iteration]

                conditions = CarlaSettings()
                conditions.set(
                    SendNonPlayerAgentsInfo=True,
                    NumberOfVehicles=vehicles,
                    NumberOfPedestrians=pedestrians,
                    WeatherId=weather

                )
                # Add all the cameras that were set for this experiments
                conditions.add_sensor(camera)
                experiment = Experiment()
                experiment.set(
                    Conditions=conditions,
                    Poses=poses,
                    Task=iteration,
                    Repetitions=1
                )
                experiments_vector.append(experiment)

        return experiments_vector
