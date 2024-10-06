from setuptools import setup, find_packages

setup(
    name="gensh",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "openai>=1.0.0",
        "anthropic>=0.35.0",
        "PyYAML>=6.0",
        "ollama>=0.1.0",
        "python-dotenv>=0.19.0",
        "markdown>=3.3.0",
        "python-docx>=0.8.10",
        "requests>=2.25.0",
        "beautifulsoup4>=4.9.3",
    ],
    extras_require={
        "dev": [
            "pytest>=6.2",
        ],
    },
    entry_points={
        'console_scripts': [
            'gensh=gensh.processor:main',
        ],
    },
)
