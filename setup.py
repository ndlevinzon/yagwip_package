import setuptools

setuptools.setup(
    name="yagwip",
    version="2.0.1",
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
        "numpy<2",
        "pandas"
    ],
    entry_points={
        "console_scripts": [
            "yagwip = yagwip.yagwip:main",
            "yagtraj = yagwip.yagtraj:main",
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
