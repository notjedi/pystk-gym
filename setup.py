import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pystk_gym",
    version="1.0.0",
    author="Krithic Kumar",
    author_email="temp.terrafly@slmail.me",
    description="",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/notjedi/pystk-gym",
    packages=setuptools.find_packages(),
    classifiers=[
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy",
        "sympy",
        "PyQt5",
        "pygame",
        "gymnasium",
        "matplotlib",
        "pettingzoo",
        "PySuperTuxKart",
    ],
    extras_require={
        "dev": ["mypy", "black", "isort", "flake8", "pylint", "pyright", "pytest"]
    },
)
