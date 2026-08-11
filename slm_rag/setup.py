from setuptools import setup

setup(
    name="slm-rag",
    version="0.1.3",
    description="A lightweight, CPU-optimized RAG library powered by a local Small Language Model (SLM)",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Suryaprakash CV",
    author_email="suryaprakash.c.v@gmail.com",
    url="https://github.com/SuryaprakashCV/SLM-RAG",
    packages=["slm_rag"],
    include_package_data=True,
    install_requires=[
        "onnxruntime-genai>=0.2.0",
        "huggingface_hub>=0.10.0",
        "pyyaml"
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ]
)
