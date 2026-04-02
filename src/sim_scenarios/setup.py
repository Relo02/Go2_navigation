from setuptools import setup
from glob import glob
import os

package_name = 'sim_scenarios'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'scenarios'),
            glob('scenarios/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Maintainer',
    maintainer_email='todo@example.com',
    description='Runtime obstacle management for Gazebo simulation',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'scenario_manager = sim_scenarios.scenario_manager_node:main',
            'spawn_obstacle = sim_scenarios.spawn_obstacle:main',
        ],
    },
)
