from setuptools import find_packages, setup

package_name = "tomato_pipeline"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(include=["pipeline_app", "pipeline_app.*"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="robot",
    maintainer_email="robot@todo.todo",
    description="Tomato harvest capstone v3 pipeline node",
    license="TODO: License declaration",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "pipeline_node = pipeline_app.ros2.pipeline_node:main",
            "get_cam_intrinsics = pipeline_app.non_ros2.tools.get_cam_intrinsics:main",
            "pipeline_trigger_keyboard = pipeline_app.ros2.pipeline_trigger_keyboard:main",
        ],
    },
)
