from setuptools import setup
from glob import glob
import os

package_name = 'gesture_to_flight'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', 
         ['resource/' + package_name]),
        ('share/' + package_name, 
         ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
         glob(os.path.join('launch', '*.py'))),
    ],
    install_requires=['setuptools', 'rclpy', 'std_msgs', 'geometry_msgs'],
    zip_safe=True,
    maintainer='etienne',
    maintainer_email='etiennesasenarine@gmail.com',
    description='ROS 2 package for gesture-based drone flight control',
    license='TODO: License declaration',
    entry_points={
        'console_scripts': [
            'gesture_inference=gesture_to_flight.gesture_inference_ros_node:main',
            'flight_controller=gesture_to_flight.flightInstructions:main',
        ],
    },
)
