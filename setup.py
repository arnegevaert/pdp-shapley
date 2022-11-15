from setuptools import setup

setup(
    name="pddshap",
    version="0.0.1",
    author="Arne Gevaert",
    python_requires=">=3.6.8, <4",
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
    ],
    extras_require={
        "dev": [
            "matplotlib",
            "shap",
            "seaborn"
        ]
    }
)