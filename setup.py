import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="polyroi",                     # This is the name of the package
    version="0.0.3",                        # The initial release version
    author="Taha Bouhsine",                     # Full name of the author
    author_email="skywolf.mo@gmail.com",

    description="This tool help in extracting the region of interest in a given image.",
    # Long description read from the the readme file
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/skywolfmo/polyroi",
    project_urls={
        "Bug Tracker": "https://github.com/skywolfmo/polyroi/issues",
    },

    # List of all python modules to be installed
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],                                      # Information to filter the project on PyPi website
    python_requires='>=3.6',                # Minimum version requirement of the package
    py_modules=["polyroi"],             # Name of the python package
    # Directory of the source code of the package
    package_dir={'': 'src'},
    packages=setuptools.find_packages(where="src"),

    install_requires=[
        'numpy',
        'opencv-python'
    ]                     # Install other dependencies if any
)
