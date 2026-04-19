from setuptools import setup, find_packages

package_name = 'graphnav_mpc_planner'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', ['config/graphnav_mpc_params.yaml']),
        ('share/' + package_name + '/launch', ['launch/graphnav_mpc_planner.launch.py']),
        ('share/' + package_name + '/rviz',   ['rviz/graphnav_mpc.rviz']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Lorenzo Ortolani',
    maintainer_email='ortolore@gmail.com',
    description='Graph-nav Dijkstra planner with CasADi MPC tracker',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'planner_node = graphnav_mpc_planner.planner_node:main',
            'mpc_node = graphnav_mpc_planner.mpc_node:main',
            'setpoint_to_cmd_vel_node = graphnav_mpc_planner.setpoint_to_cmd_vel_node:main',
            'odom_to_pose_node = graphnav_mpc_planner.odom_to_pose_node:main',
            'mock_nav_graph_publisher = graphnav_mpc_planner.mock_nav_graph_publisher:main',
        ],
    },
)
