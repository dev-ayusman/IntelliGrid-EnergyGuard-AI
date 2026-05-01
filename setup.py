"""
Setup script for IntelliGrid EnergyGuard AI
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="intelligen-energyguard-ai",
    version="1.0.0",
    author="Ayusman Choudhury",
    author_email="dev.ayusman@example.com",
    description="AI-Powered Building Energy Anomaly Detection System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dev-ayusman/IntelliGrid-EnergyGuard-AI",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.9",
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "plotly>=5.14.0",
        "scipy>=1.10.0",
        "streamlit>=1.28.0",
        "jupyter>=1.0.0",
        "statsmodels>=0.14.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "energyguard=app:main",
        ],
    },
)
