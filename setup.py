from setuptools import find_packages, setup


setup(
    name="netra4d-verified",
    version="2.0.0",
    description="Verified Drive video ingestion and real Ultralytics inference",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "torch>=2.0.0",
        "ultralytics>=8.4.0",
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "pyyaml>=6.0",
        "requests>=2.31.0",
        "google-api-python-client>=2.140.0",
        "google-auth-oauthlib>=1.2.1",
        "google-auth-httplib2>=0.2.0",
        "kaggle==1.6.17",
        "pydantic>=2.0.0",
        "fastapi>=0.110.0",
        "uvicorn>=0.29.0",
        "lap>=0.5.12",
        "httpx>=0.27.0",
        "Pillow>=10.0.0",
    ],
    python_requires=">=3.10",
)
