from setuptools import setup, find_packages

setup(
    name="gru_chatbot",
    version="1.0.0",
    description="A GRU-based chatbot with Flask and Streamlit support",
    author="Your Name",
    packages=find_packages(exclude=("tests", "docs")),
    install_requires=[
        "tensorflow>=2.0",
        "numpy",
        "pandas",
        "scikit-learn",
        "flask",
        "streamlit"
    ],
    entry_points={
        "console_scripts": [
            "chatbot-train=train_gru_chatbot:main",
            "chatbot-flask=chatbot_flask:main",
            "chatbot-ui=chatbot_streamlit:main"
        ]
    },
    include_package_data=True,
    python_requires=">=3.8",
)
