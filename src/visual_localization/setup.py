from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'visual_localization'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools', 'opencv-python', 'numpy'],
    zip_safe=True,
    maintainer='AxeEffect',
    maintainer_email='eeeffect@gmail.com',
    description='GNSS-denied visual localization for drones using predefined maps',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'localization_node = visual_localization.localization_node:main'
        ],
    },
)
