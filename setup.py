from setuptools import setup, find_packages

setup(
    name='sentiment-analysis',
    version='0.1',
    packages=find_packages(),  # Isso inclui todos os pacotes no diretório sentiment_analysis
    install_requires=[
        'Flask==3.1.0',
        'nltk==3.9.1',
        'pandas==2.2.3',
        'langdetect==1.0.9',
        'googletrans==4.0.0-rc1',
        'matplotlib==3.9.2',
        'wordcloud==1.9.4',
        'scikit-learn==1.5.2',
        'numpy==2.0.2',
        'tqdm==4.67.0',
    ],
)
