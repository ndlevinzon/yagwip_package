import setuptools

setuptools.setup(
    name="yagwip",
    version="2.0.4",
    author="Gregor Patof, NDL",
    description="Yet Another Gromacs Wrapper In Python",
    packages=setuptools.find_packages('src'),
    package_dir={'':'src'},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 only (GPLv3)",
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
            "fep_prep = yagwip.fep_prep:main",
            "tremd_prep = yagwip.tremd_prep:main"
        ],
    },
    include_package_data=True,
    package_data={
        "yagwip": [
            "assets/*.txt",
            "templates/*.mdp",
            "templates/*.slurm",
            "templates/amber14sb.ff/*",
        ],
        "utils": [
            "*.py",
        ],
        "templates": [
            "*.mdp",
            "*.slurm",
            "amber14sb.ff/*",
        ],
        "assets": [
            "*.txt",
        ],
    }
)
