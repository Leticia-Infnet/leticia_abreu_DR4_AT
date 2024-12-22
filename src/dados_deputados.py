from google import genai
from google.genai import types
from dotenv import load_dotenv
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

# Classe para recuperar e processar os dados dos deputados


class ProcessDadosDeputados():
    def get_deputados(self):
        '''Função que recupera os dados dos deputados do endpoint /deputados da API da Câmara dos Deputados
        e salva os dados em um arquivo parquet no diretório data.
        Args:
            None
        Returns:
            None
        '''
        url = 'https://dadosabertos.camara.leg.br/api/v2'
        params = {
            'dataInicio': '2024-08-01',
            'dataFim': '2024-08-30',
        }
        response = requests.get(url=f'{url}/deputados',
                                params=params)
        if not response.ok:
            raise Exception('Nao foi possivel recuperar os dados')
        df_deputados = pd.DataFrame().from_dict(
            json.loads(response.text)['dados'])
        df_deputados.to_parquet(os.path.join(DATA_PATH, 'deputados.parquet'))

    def distribuicao_partido(self):
        '''
        Função que gera um prompt para criar um código python que lê um arquivo parquet
        com os dados dos deputados e gera um gráfico de pizza com a distribuição dos deputados
        por partido.
        Args:
            None
        Returns:
            None
        '''
        prompt = f'''Você deve criar um código python que leia um arquivo parquet e 
            faça um gráfico de pizza com a distribuição dos deputados por partido. O arquivo
            parquet deve ser lido do diretório {DATA_PATH+'/deputados.parquet'}. O arquivo 
            parquet possui as seguintes colunas: id, uri, nome, siglaPartido, uriPartido, siglaUf,
            idLegislatura, urlFoto e email. O gráfico deve ser salvo no diretório {DOCS_PATH+'/distribuicao_deputados.png'}.
            O código gerado não deve conter comentários e deve ser construído de forma a ser executado dinamicamente através
            da função exec do python. O gráfico deve conter título e legendas. Tenha em mente que são 21 partidos diferentes,
            e alguns partidos tem representação pequena, então a legenda, título e nomes dos partidos devem ser ajustados
            de forma a melhorar a visibilidade. Os nomes dos partidos devem ser rotacionados para melhorar a visualização.
            Os quatro menores partidos devem ser agrupados em um único setor do gráfico chamado "Outros".'''
        response = client.models.generate_content(
            model='gemini-1.5-pro',
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.3
                                               )
        )
        response = response.text.replace('```python', '').replace('```', '')
        exec(response)

    def parquet_to_json_string(self, file_path):
        '''Função que lê um arquivo parquet e converte os dados para uma string JSON.
        Args:
            file_path (str): Caminho do arquivo parquet.
        Returns:
            json_string (str): String JSON com os dados do arquivo parquet.'''
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f'O arquivo {file_path} não foi encontrado.')
        df = pd.read_parquet(file_path)

        json_string = df.to_json(orient='records', force_ascii=False)

        return json_string

    def distribuicao_insights(self):
        '''
        Função que gera um prompt para criar insights sobre a distribuição dos deputados por partido e salva
        os insights em um arquivo JSON.
        Args:
            None
        Returns:
            None
        '''
        file_path = os.path.join(DATA_PATH, 'deputados.parquet')
        try:
            deputados_json = self.parquet_to_json_string(file_path)
        except Exception as e:
            raise RuntimeError(f"Erro ao processar o arquivo Parquet: {e}")

        data = json.loads(deputados_json)

        df = pd.DataFrame(data)

        distribuicao_partido = df['siglaPartido'].value_counts().to_dict()

        distribuicao_partido = {k: v for k, v in sorted(
            distribuicao_partido.items(), key=lambda item: item[1], reverse=True)}

        distribuicao_partido = json.dumps(
            distribuicao_partido, ensure_ascii=False)

        prompt = f'''Você é um jurista com profundo conhecimento sobre a Câmara dos Deputados. Seu objetivo é analisar a distribuição
        dos deputados por partido e gerar insights sobre os dados. Aqui estão os dados disponíveis no formato de JSON string: {distribuicao_partido}.
        A JSON string está organizada da seguinte maneira: "PARTIDO": n° deputados. Seus insights devem ser estruturados no formato de uma JSON String. Você deve
        gerar três insights sobre três tópicos diferentes que são afetados pela composição da câmara dos deputados. O JSON gerado não deve ter nada além da string JSON,
        como por exemplo comentários. Veja abaixo o exemplo de como estruturar sua resposta JSON:
        ### EXEMPLO
        {{
  "politica_externa": [
    {{ "primeiro_insight": "texto" }},
    {{ "segundo_insight": "texto" }},
    {{ "terceiro_insight": "texto" }}
  ],
  "sociedade_civil": [
    {{ "primeiro_insight": "texto" }},
    {{ "segundo_insight": "texto" }},
    {{ "terceiro_insight": "texto" }}
  ]
        }}
        '''
        response = client.models.generate_content(
            model='gemini-1.5-pro',
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.3
                                               )
        )
        response = response.text.replace("```json\n", '').replace(
            "\n```", '').replace("\n", "")
        data = json.loads(response)
        file_path = DATA_PATH+'/insights_distribuicao_deputados.json'
        with open(file=file_path, mode='w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
