"""
Setup script for Decision Agent platform.
"""
from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text() if readme_path.exists() else ""

setup(
    name="decision-agent",
    version="0.1.0",
    description="Principal Data Science Decision Agent Platform",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Data Science Team",
    python_requires=">=3.10",

    # Package discovery
    package_dir={"": "src"},
    packages=find_packages(where="src"),

    # Dependencies
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "pyyaml>=6.0.0",
        "jsonschema>=4.17.0",
        "mlflow>=2.8.0",
    ],

    # Optional dependencies
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "pylint>=2.17.0",
            "flake8>=6.0.0",
        ],
        "databricks": [
            "pyspark>=3.4.0",
            "delta-spark>=2.4.0",
            "databricks-feature-store>=0.15.0",
        ],
    },

    # Entry points
    entry_points={
        "console_scripts": [
            "decision-agent=decision_agent.cli:main",
        ],
    },

    # Classifiers
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
