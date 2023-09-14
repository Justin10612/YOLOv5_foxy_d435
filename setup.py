from setuptools import setup
from glob import glob
import os

package_name = 'detection'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('./launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='sss0301',
    maintainer_email='atom.9031@gmail.com',
    description='The package with YOLOv5 and hook up with d435',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'yolov5_ros = '+package_name+'.main:detect_main',
        ],
    },
)
