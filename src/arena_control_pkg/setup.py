from setuptools import find_packages, setup

package_name = 'arena_control_pkg'

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
            'deterministic_planner = arena_control_pkg.deterministic_planner:main',
            'multi_robot_deterministic_planner = arena_control_pkg.multi_robot_deterministic_planner:main',
            'even_better_deterministic_planner = arena_control_pkg.even_better_deterministic_planner:main'
        ],
    },
)
