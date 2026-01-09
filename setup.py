from setuptools import setup, find_packages

setup(
    name="backtesting-framework",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.21.0",
        "xlsxwriter>=3.0.0",
        "yfinance>=0.2.0",
        "ta>=0.10.0",
        "matplotlib>=3.5.0",
    ],
    python_requires=">=3.8",
    description="A backtesting framework with parameter optimization",
    author="Your Name",
)
