from setuptools import setup, find_packages

setup(
    name="glaucoma-detection",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.19.5",
        "pandas>=1.3.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "opencv-python>=4.5.3",
        "albumentations>=1.3.0",
        "pillow>=8.2.0",
        "scikit-learn>=0.24.2",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "wandb>=0.15.0",
        "tqdm>=4.61.2",
    ],
    author="Vaibhav Talekar",
    author_email="vaibhavtalekar87@gmail.com",
    description="A modular machine learning pipeline for glaucoma detection",
    keywords="glaucoma, medical imaging, deep learning",
    python_requires=">=3.7",
)