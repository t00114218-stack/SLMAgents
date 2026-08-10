from setuptools import setup

setup(
    name="slm-text-to-sql",
    version="0.1.0",
    description="A lightweight, CPU-optimized Text-to-SQL translation agent powered by a local Small Language Model (SLM)",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Suryaprakash CV",
    author_email="suryaprakash.c.v@gmail.com",
    packages=["slm_text_to_sql"],
    include_package_data=True,
    install_requires=[
        "onnxruntime-genai>=0.2.0",
        "huggingface_hub>=0.10.0",
        "pyyaml"
    ]
)
