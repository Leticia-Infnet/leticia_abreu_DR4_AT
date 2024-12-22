
# Dashboard Câmara dos Deputados com LLM

Projeto como parte do Assesment de Engenharia de Prompts para Ciência de Dados, essa aplicação consome os dados coletados e tratados da API da Câmara dos Deputados, disponíveis [aqui](https://dadosabertos.camara.leg.br/swagger/api.html), e disponibiliza ao usuário interagir com esses dados de maneira dinâmica, incluindo: sumarizações com LLM, gráficos, dataframes e um RAG powered ChatBot!  

## Linguagens, Frameworks e Ferramentas usadas

![Python](https://img.shields.io/badge/Python-3.11.9-blue?style=for-the-badge&logo=python&logoColor=yellow) ![Streamlit](https://img.shields.io/badge/streamlit-1.41.0-red?style=for-the-badge&logo=streamlit&logoColor=red) ![Gemini](https://img.shields.io/badge/gemini-1.5-%234796E3?style=for-the-badge&logo=googlegemini&logoColor=%234796E3) ![Llama-Index](https://img.shields.io/badge/llama--index-0.12.8-purple?style=for-the-badge)

## Acessado a Aplicação

Você pode acessar a aplicação clicando neste [link](https://leticia-abreu-dr4-at.streamlit.app/). Obs: A aplicação pode demorar um pouco para iniciar após ficar um tempo em stand-by.

## Rodando localmente

Clone o projeto

```
  git clone https://github.com/Leticia-Infnet/leticia_abreu_DR4_AT.git
```

Entre no diretório do projeto

```
  cd leticia_abreu_DR4_AT
```

Instale as dependências

```
  pip install -r requirements.txt
```

Na raíz do projeto, crie um arquivo .env contendo sua chave da API do Gemini, no formato abaixo

GEMINI_API_KEY = SUA_API_KEY

Da raíz do diretório, rode a aplicação streamlit

```
streamlit run dashboard.py
```
