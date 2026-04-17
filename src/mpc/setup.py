import glob
from setuptools import setup

package_name = 'mpc'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', glob.glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Lorenzo Ortolani',
    maintainer_email='lorenzo@example.com',
    description='MPC trajectory tracking and control nodes for Go2 quadruped navigation',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'mpc_node = mpc.mpc_node:main',
            'mpc_path_optimizer_node = mpc.mpc_path_optimizer_node:main',
            'odom_to_pose_node = mpc.odom_to_pose_node:main',
            'setpoint_to_cmd_vel_node = mpc.setpoint_to_cmd_vel_node:main',
        ],
    },
)
