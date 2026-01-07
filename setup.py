from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="airline-satisfaction",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A machine learning package for predicting airline passenger satisfaction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/airline-satisfaction-prediction",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.23.0",
        "matplotlib>=3.6.0",
        "seaborn>=0.12.0",
        "scikit-learn>=1.2.0",
        "xgboost>=1.7.0",
        "kagglehub>=0.1.0",
    ],
    entry_points={
        'console_scripts': [
            'airline-satisfaction=airline_satisfaction.main:main',
        ],
    },
)
