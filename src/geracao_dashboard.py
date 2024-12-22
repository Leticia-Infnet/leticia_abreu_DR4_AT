from google import genai
from dotenv import load_dotenv
from google import genai
from google.genai import types
import os

DATA_PATH = os.path.abspath(os.path.join('.', 'data'))

DOCS_PATH = os.path.abspath(os.path.join('.', 'docs'))

ENV_PATH = os.path.abspath(os.path.join('.', '.env'))

if os.path.exists(ENV_PATH):
    load_dotenv(ENV_PATH,
                override=True)

client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))

CODELINES = ''

# Funções para geração do código inicial do dashboard através das técnicas de Prompting
# Chain-of-Thoughts e Batch-prompting. Em meu caso, o código gerado não funcionou corretamente
# e necessitou de algumas modificações para funcionar corretamente.


def geracao_codigo_chain():
    '''
    Função que gera o código inicial do dashboard através da técnica de Prompting Chain-of-Thoughts.
    Args:
        None
    Returns:
        None
    '''
    system_instruction = '''You are an expert software developer and a helpful coding assistant. 
    You are able to generate high-quality code in any programming language.'''

    chat = client.chats.create(
        model='gemini-1.5-pro',
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.4,
        ),
    )

    response = chat.send_message(f'''Write a python code that create an Streamlit app with three tabs, named: Overview, Despesas and Proposições.
                                 The Overview tab must have the title: "Análise da Câmara dos Deputados" and following text description: 
                                 "Análises geradas com os dados públicos da API da Câmara dos Deputados com ajuda de diferentes técnicas de Prompting."
                                 The app must read an yaml file located in relative path: './data/config.yaml' and display the text in it. The key
                                 to the text in the yaml file is overview_summary. Generate only the python code, no explanations or comments needed.
                                 ''')

    response = chat.send_message(f'''Modify the previous code so now the Overview tab display a .png image file located in the in relative path: './docs/distribuicao_deputados.png'.
                                 Before the image, create a subtitle with the text: "Gráfico gerado por LLM da distribuição de deputados por partido" Generate only the python code,
                                 no explanations or comments needed.''')

    response = chat.send_message(f'''Modify the code so now the Overview tab display the text in a JSON file located in relative path: './data/insights_distribuicao_deputados.json.'
                                 The JSON has the structure described bellow:
                                 - The JSON is a main object containing three primary fields.
                                 - Each of the three primary fields is a list of objects, representing different insights within each topic.
                                 - Each field has a list of objects, each containing a single key identifying the insight (e.g., primeiro_insight, segundo_insight, etc.) with a string value describing the insight.
                                 - Each object in the list contains: A key indicating the insight number (primeiro_insight, segundo_insight, etc. and a value that is a string detailing the insight.
                                 The summary of the structure:
                                 {{
                                    "main_field": [
                                        {{
                                        "insight_name": "Insight description as a string"
                                        }},
                                        ...
                                    ],
                                    ...
                                    }}
                                 You must only display the insights description texts. Before the first group of texts, the subtitle "Insights destribuição dos deputados: Governabilidade" must be displayed.
                                 Before the second group of texts, the subtitle "Insights destribuição dos deputados: Pauta Legislativa" must be displayed.
                                 Finally, before the last group of texts, the subtitle: "Insights destribuição dos deputados: Representatividade Popular" must be displayed.
                                 The texts must not be displayed as an JSON, but like normal string text.
                                 The name of the subtitles don't reflect the main_field names.
                                 The encoding of the JSON is utf-8.
                                 ''')

    codelines = response.text.replace("```python\n", '').replace("\n```", '')
    global CODELINES
    CODELINES = codelines
    with open('dashboard.py', 'w', encoding='utf-8') as fid:
        fid.write(codelines)


def geracao_codigo_batch():
    '''
    Função que gera o código inicial do dashboard através da técnica de Prompting Batch-prompting.
    Args:
        None
    Returns:
        None
    '''
    system_instruction = '''You are an expert software developer and a helpful coding assistant. 
    You are able to generate high-quality code in any programming language.'''

    prompt_batch = f'''
    You are working on a Streamlit app that displays information about the brazilian federal deputies. Part of the python code has already been written {CODELINES}.
    Keep in mind that the enconding of all the JSON files is utf-8. 
    Now you must do the following additions:
    - The Despesas tab must show the insights about the deputies expenses. The insights are located in the JSON file in the relative path:
    './data/insights_despesas_deputados.json'. This JSON file has a root object containing three main keys. Each of these keys contains an array of exactly 4 strings. 
    Each string represents an analytical insight or observation about parliamentary expenses in Brazil, written in Portuguese. Here's the pattern:
    {{
    "insights_type": [
        // Array of 4 strings
        String1,
        String2, 
        String3, 
        String4  
    ], ...
    }}
    The texts in the four strings of each key must desplayed. For the first key, the subtitle "Insights por tipo de despesa" must be displayed before the text. 
    For the second key, the subtitle "Insights despesas por partido". For the third key, the subtitle "Insights despesas por UF".

    ######

    - The Despesas tab must show a bar graph with the temporal series of the selected deputy. The data about each deputy is available at the relative path: './data/serie_despesas_diárias_deputados.parquet'.
    It's a parquet file that has each deputy expenses grouped by dataDocumento, id and tipoDespesa. It has the following columns:
    tipoDespesa: Type of expense made. String representing the type of expense.
    dataDocumento: Document date. Datetime object representing the day when the expense was made.
    id: Deputy identifier. Integer representing the identifier of the deputy who made the expense.
    nome: Deputy name. String representing the name of the deputy who made the expense.
    siglaPartido: Party acronym. String representing the acronym of the deputy's party.
    siglaUf: State acronym. String representing the acronym of the state that elected the deputy.
    valorLiquido: Net value. Float or integer representing the net value of the expense.
    valorGlosa: Deduction value. Float or integer representing the deducted(unauthorized or discounted) value of the expense.
    The user must be able to use a selectbox to select a deputy by name and see the temporal series bar graph of the total Net value spent by month by the selected deputy.
    The subtitle before the graph must be: "Despesas por deputado por mês (2024)".

    ######

    - The Proposições tab must show a dataframe with the data about the deputies propositions. The .parquet file with this data is located at the relative path: './data/proposicoes_deputados.parquet'
    The subtitle before the Dataframe must be: "Proposições dos deputados sobre os temas: Economia; Educação; Ciência, Tecnologia e Inovação."

    - The Proposições must display the text of each proposition. The data is available at the relative path: './data/sumarizacao_proposicoes.json'. The JSON contains an array of 
    objects, where each object represents a legislative proposal or action. Each object has a consistent structure with two key properties:
    [
    {{
        "id": Number,        // Unique numerical identifier
        "resumo": String     // Text summary in Portuguese of the legislative item
    }},
    // ... more items with the same structure
    ]

    You must display the texts in the app in the following pattern:
    "Proposição ID XXXXX: YYYYY"
    Where XXXXX is the id Number of the proposition and YYYY the text summary of the legislative item. 
    Before all the propositions texts, you must display the subtitle: "Resumo das proposições dos deputados".
    Generate me only the python code, no comments or explanations needed.
    '''

    response = client.models.generate_content(
        model='gemini-1.5-pro',
        contents=prompt_batch,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.4,
        ))

    codelines = response.text.replace("```python\n", '').replace("\n```", '')
    with open('dashboard.py', 'w', encoding='utf-8') as fid:
        fid.write(codelines)


if __name__ == '__main__':
    geracao_codigo_chain()
    geracao_codigo_batch()
