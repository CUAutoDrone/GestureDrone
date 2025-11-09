from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'gesture_cv'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'GestureData'),
            glob('GestureData/*.joblib')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='etienne',
    maintainer_email='etiennesasenarine@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'gesture_inference_node = gesture_cv.gesture_inference_node:main',
        ],
    },
)
