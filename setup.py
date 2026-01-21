from setuptools import setup, find_packages

setup(
    name="Topsis-Sneha-101234",
    version="1.0.0",
    packages=find_packages(),
    install_requires=["numpy","pandas"],
    entry_points={
        "console_scripts": [
            "topsis=topsis.topsis:main"
        ]
    },
)
