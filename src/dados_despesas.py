from google import genai
from dotenv import load_dotenv
from tqdm import tqdm
from google import genai
from google.genai import types
import requests
import json
import pandas as pd
import os

DATA_PATH = os.path.abspath(os.path.join('.', 'data'))

DOCS_PATH = os.path.abspath(os.path.join('.', 'docs'))

ENV_PATH = os.path.abspath(os.path.join('.', '.env'))

if os.path.exists(ENV_PATH):
    load_dotenv(ENV_PATH,
                override=True)

client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))

# Classe para recuperar e processar os dados das despesas dos deputados


class ProcessDespesas():
    def get_despesas(self):
        '''
        Função que recupera os dados das despesas dos deputados do endpoint /deputados/{id}/despesas da API da Câmara dos Deputados,
        agrupa os dados por dia, id do deputado e tipo de despesa e salva os dados em um arquivo parquet no diretório data.
        Args:
            None
        Returns:
            None
        '''
        # Recuperando os ids dos deputados do endpoint /deputados

        url_base = 'https://dadosabertos.camara.leg.br/api/v2'
        params = {
            'dataInicio': '2024-08-01',
            'dataFim': '2024-08-30',
        }
        response = requests.get(url=f'{url_base}/deputados',
                                params=params)
        if not response.ok:
            raise Exception('Não foi possivel recuperar os dados')
        df_deputados = pd.DataFrame().from_dict(
            json.loads(response.text)['dados'])

        anoDespesa = 2024

        legislaturaAtual = 57  # Número da legislatura atual

        maxItens = 100  # Número máximo de itens por página

        list_expenses = []

        for id in tqdm(df_deputados.id.unique()):
            url = f'{url_base}/deputados/{id}/despesas'
            params = {
                'ano': anoDespesa,
                'itens': maxItens,
                'idLegislatura': legislaturaAtual
            }

            response = requests.get(url, params)
            df_resp = pd.DataFrame().from_dict(
                json.loads(response.text)['dados'])
            df_resp['id'] = id
            list_expenses.append(df_resp)

            df_links = pd.DataFrame().from_dict(
                json.loads(response.text)['links'])
            df_links = df_links.set_index('rel').href

            while 'next' in df_links.index:
                response = requests.get(df_links['next'])
                df_resp = pd.DataFrame().from_dict(
                    json.loads(response.text)['dados'])
                df_resp['id'] = id
                list_expenses.append(df_resp)

                df_links = pd.DataFrame().from_dict(
                    json.loads(response.text)['links'])
                df_links = df_links.set_index('rel').href

        df_expenses = pd.concat(list_expenses)

        df_expenses = df_expenses.merge(df_deputados, on=['id'])

        # Transformando o campo dataDocumento do dataframe de string no formato de
        # de data ISO 8601 para objeto datetime

        df_expenses['dataDocumento'] = pd.to_datetime(
            df_expenses['dataDocumento'],
            format='ISO8601')

        # Dropando colunas que não serão relevantes para a análise

        df_expenses.drop(columns=['ano',
                                  'mes',
                                  'codDocumento',
                                  'tipoDocumento',
                                  'codTipoDocumento',
                                  'numDocumento',
                                  'urlDocumento',
                                  'codLote',
                                  'parcela',
                                  'uri',
                                  'uriPartido',
                                  'idLegislatura',
                                  'urlFoto',
                                  'email',
                                  'valorDocumento',
                                  'nomeFornecedor',
                                  'cnpjCpfFornecedor',
                                  'numRessarcimento'], inplace=True)

        # Criando df com a soma dos valores líquidos e glosados

        df_sum = df_expenses.groupby(['dataDocumento', 'id', 'tipoDespesa'])[
            ['valorLiquido', 'valorGlosa']].sum().reset_index()

        # Fazendo merge dos dataframes

        df_expenses_grouped = df_expenses.drop(columns=['valorLiquido', 'valorGlosa']).drop_duplicates(
            subset=['dataDocumento', 'id', 'tipoDespesa'])
        df_expenses_grouped = df_expenses_grouped.merge(
            df_sum, on=['dataDocumento', 'id', 'tipoDespesa'])

        df_expenses_grouped.to_parquet(os.path.join(
            DATA_PATH, 'serie_despesas_diárias_deputados.parquet'))

    def generate_analises(self):
        '''
        Função que gera um prompt para criar um código python que lê um arquivo parquet 
        com os dados das despesas dos deputados e realiza 3 análises. O resultado das análises
        é salvo em um arquivo JSON no diretório data.
        Args:
            None
        Returns:
            None
        '''
        prompt_analises = f'''
        Você é um cientista de dados trabalhando para a Câmara dos Deputados. Sua tarefa é analisar padrões e tendências nas despesas dos deputados.
        O arquivo parquet contendo os dados das despesas dos deputados agrupados por dataDocumento, id e tipoDespesa possui as seguintes colunas e informações:
        - tipoDespesa: Tipo de despesa realizada. String que representa o tipo de despesa realizada.
        - dataDocumento: Data do documento. Objeto datetime que representa o dia em que a despesa foi realizada.
        - id: Identificador do deputado. Inteiro que representa o identificador do deputado que realizou a despesa.
        - nome: Nome do deputado. String que representa o nome do deputado que realizou a despesa.
        - siglaPartido: Sigla do partido. String que representa a sigla do partido do deputado que realizou a despesa.
        - siglaUf: Sigla da unidade federativa. String que representa a sigla da unidade federativa que elegeu o deputado.
        - valorLiquido: Valor líquido. Float ou inteiro que representa o valor líquido da despesa realizada.
        - valorGlosa: Valor glosa. Float ou inteiro que representa o valor glosado (não autorizado ou descontado) da despesa realizada.

        O arquivo parquet está localizado em {DATA_PATH+'/serie_despesas_diárias_deputados.parquet'}.
        Com base nos dados disponíveis, você deve criar e realizar 3 análises e retornar um código Python que possa ser executado dinamicamente através da função exec do Python.
        Cada análise deve conter pelo menos três resultados relevantes diferentes.
        Não é necessário realizar análises com o valorGlosa.
        As análises devem ser realizadas com a biblioteca pandas.
        O código não deve conter comentários, somente o código Python.
        O resultado da execução do código Python deve gerar um arquivo JSON (utf-8), com ensure_ascii setado para False e indent=4, em {DATA_PATH+'/analises_despesas.json'} e que siga o seguinte formato de exemplo:

        ### EXEMPLO
        {{
        'analise_deputados' : [
        'top_5_deputados_mais_gastaram': 'Deputado 1 (Partido): R$ XXX', 'Deputado 2 (Partido): R$ XXX', 'Deputado 3 (Partido): R$ XXX', 'Deputado 4 (Partido): R$ XXX', 'Deputado 5 (Partido): R$ XXX'],
        'top_5_deputados_que_menos_gastaram': 'Deputado 1 (Partido): R$ XXX', 'Deputado 2 (Partido): R$ XXX', 'Deputado 3 (Partido): R$ XXX', 'Deputado 4 (Partido): R$ XXX', 'Deputado 5 (Partido): R$ XXX'],
        'media_gastos_deputados': 'R$ XXX'
        ]
        }}
        '''

        response = client.models.generate_content(
            model='gemini-1.5-pro',
            contents=prompt_analises,
            config=types.GenerateContentConfig(temperature=0.3
                                               )
        )

        response = response.text.replace('```python', '').replace('```', '')

        exec(response)

    def generate_insights(self):
        '''
        Função que gera um prompt que orienta a criação de insights a partir dos resultados das
        análises das despesas dos deputados e salva os insights em um arquivo JSON no diretório data.
        Args:
            None
        Returns:
            None
        '''
        file_path = os.path.join(DATA_PATH, 'analises_despesas.json')

        with open(file_path, 'r', encoding='utf8') as file:
            data = json.load(file)

        json_string = json.dumps(data, ensure_ascii=False, indent=4)

        prompt_insights = f'''
        Você é um cientista de dados trabalhando para a Câmara dos Deputados. Sua tarefa é gerar insights a partir dos dados das despesas dos deputados.
        O arquivo parquet contendo os dados das despesas dos deputados agrupados por dataDocumento, id e tipoDespesa possui as seguintes colunas e informações:
        - tipoDespesa: Tipo de despesa realizada. String que representa o tipo de despesa realizada.
        - dataDocumento: Data do documento. Objeto datetime que representa o dia em que a despesa foi realizada.
        - id: Identificador do deputado. Inteiro que representa o identificador do deputado que realizou a despesa.
        - nome: Nome do deputado. String que representa o nome do deputado que realizou a despesa.
        - siglaPartido: Sigla do partido. String que representa a sigla do partido do deputado que realizou a despesa.
        - siglaUf: Sigla da unidade federativa. String que representa a sigla da unidade federativa que elegeu o deputado.
        - valorLiquido: Valor líquido. Float ou inteiro que representa o valor líquido da despesa realizada.
        - valorGlosa: Valor glosa. Float ou inteiro que representa o valor glosado (não autorizado ou descontado) da despesa realizada.
        Você realizou as análises das despesas dos deputados e gerou os seguintes resultados:
        {json_string}
        Baseado nesses resultados, você deve criar pelo menos dois insights para cada tipo de análise realizada. Os insights devem ser organizados como um JSON
        seguindo o formato de exemplo abaixo:
        ### EXEMPLO
        {{
        'insights_analise_tipo_despesa': [
        'O tipo de despesa X é o que mais gera gastos entre os deputados. Com isso, é possível inferir que...',
        'O tipo de despesa Y é o que menos gera gastos entre os deputados. Com isso, é possível inferir que...'
        ]
        }}

        ### OBSERVAÇÃO
        Lembre-se que você é um cientista de dados e deve gerar insights relevantes e coerentes com os dados analisados.
        '''

        response = client.models.generate_content(
            model='gemini-1.5-pro',
            contents=prompt_insights,
            config=types.GenerateContentConfig(temperature=0.3
                                               )
        )

        response = response.text.replace("```json\n", '').replace(
            "\n```", '').replace("\n", "")

        data = json.loads(response)

        file_path = DATA_PATH+'/insights_despesas_deputados.json'

        with open(file=file_path, mode='w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
