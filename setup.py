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
        "Programming Language :: Python :: 3",
    ],
    python_requires=">=3.6",
    install_requires=[
        "numpy",
        "sympy",
        "PyQt5",
        "pygame",
        "gymnasium",
        "matplotlib",
        # "PySuperTuxKart",
        "pettingzoo",
    ],
    extras_require={
        "dev": ["mypy", "black", "isort", "flake8", "pytest", "pre-commit"]
    },
)
