from setuptools import find_packages, setup

package_name = 'arena_perception_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='graduate',
    maintainer_email='dimethylcadmium100@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'camera_processing_node = arena_perception_pkg.perception_node:main',
            'lidar_processing_node = arena_perception_pkg.lidar_compression:main',
            'get_rl_state_node = arena_perception_pkg.combine_lidar_camera_output_node:main',
        ],
    },
)
