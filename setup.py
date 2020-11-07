import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="3dtrees-nbingo",
    version="0.1.5",
    author="Noam Ringach, Justus Kebschull",
    author_email="nomir@cs.stanford.edu",
    description="A package for creating 3D phylogenetic trees with two axes of variation.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nbingo/3dtrees",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Science/Research',
        'Natural Language :: English',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Typing :: Typed'
    ],
    python_requires='>=3.7',
)