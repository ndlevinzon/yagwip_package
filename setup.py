import setuptools

setuptools.setup(
    name="yagwip",
    version="1.6.9",
    author="Gregor Patof, NDL",
    description="Yet Another Gromacs Wrapper In Python",
    packages=setuptools.find_packages('src'),
    package_dir={'':'src'},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        "numpy",
    ],
    entry_points={
        "console_scripts": [
            "yagwip = yagwip.yagwip:main",
        ],
    },
    include_package_data=True,
    package_data={
        "yagwip": [
            "assets/*.txt",
            "templates/*.mdp",
            "templates/*.slurm",
            "templates/amber14sb.ff/",
        ],
    }
)
