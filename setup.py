from setuptools import find_packages, setup

about = {}
with open("flight_maneuvers/__about__.py") as fp:
    exec(fp.read(), about)

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

download_url = "https://github.com/clay-curry/Flight-Maneuvers/archive/v{}.tar.gz".format(
    about["__version__"]
)

setup(
    name=about["__title__"],
    version=about["__version__"],
    description=about["__summary__"],
    author=about["__author__"],
    author_email=about["__email__"],
    url=about["__url__"],
    download_url=download_url,
    license=about["__license__"],
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=["test", "test.*"]),
    python_requires=(">=3.7,<3.8"),
    keywords=[
        "pytorch",
        "cnn",
        "convolutional-networks" "equivariant",
        "flight_manuevers",
    ],
    install_requires= [
        "torch>=1.3",
        "numpy",
        "pandas",
        "plotly",
        "escnn",
        "matplotlib",
    ],
    setup_requires=[''],
    tests_require=[''],
)
