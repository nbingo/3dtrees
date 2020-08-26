import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="3dtrees-nbingo",
    version="0.0.1",
    author="Noam Ringach, Justus Kebschull",
    author_email="nomir@cs.stanford.edu",
    description="A package for creating 3D phylogenetic trees with two axes of variation.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nbingo/3dtrees",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)