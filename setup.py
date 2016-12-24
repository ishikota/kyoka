from setuptools import setup, find_packages

setup(
    name = "kyoka",
    version = "0.2.1",
    author = "ishikota",
    author_email = "ishikota086@gmail.com",
    description = ("Simple Reinforcement Learning Library"),
    license = "MIT",
    keywords = "reinforcement learning RL",
    url = "https://github.com/ishikota/kyoka",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
    ],
)
