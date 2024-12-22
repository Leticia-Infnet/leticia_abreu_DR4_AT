from google import genai
from dotenv import load_dotenv
from tqdm import tqdm
from google import genai
from google.genai import types
import time
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

# Classe para recuperar e processar os dados das proposições


class ProcessProposicoes():
    def get_proposicoes(self):
        '''
        Função que recupera 10 proposições de cada um dos temas economia, educação e tecnologia
        do endpoint /proposicoes da API da Câmara dos Deputados e salva os dados em um arquivo parquet
        no diretório data.
        Args:
            None
        Returns:
            None
        '''
        url_base = 'https://dadosabertos.camara.leg.br/api/v2'

        params = {
            'codTema': 40,
            'itens': 10,
            'dataInicio': '2024-08-01',
            'dataFim': '2024-08-30',
            'ano': 2024
        }

        response = requests.get(url=f'{url_base}/proposicoes',
                                params=params)

        if not response.ok:
            raise Exception('Nao foi possivel recuperar os dados')

        df_proposicoes_economia = pd.DataFrame().from_dict(
            json.loads(response.text)['dados'])

        params = {
            'codTema': 46,
            'itens': 10,
            'dataInicio': '2024-08-01',
            'dataFim': '2024-08-30',
            'ano': 2024
        }

        response = requests.get(url=f'{url_base}/proposicoes',
                                params=params)

        if not response.ok:
            raise Exception('Nao foi possivel recuperar os dados')

        df_proposicoes_educacao = pd.DataFrame().from_dict(
            json.loads(response.text)['dados'])

        params = {
            'codTema': 62,
            'itens': 10,
            'dataInicio': '2024-08-01',
            'dataFim': '2024-08-30',
            'ano': 2024
        }

        response = requests.get(url=f'{url_base}/proposicoes',
                                params=params)

        if not response.ok:
            raise Exception('Nao foi possivel recuperar os dados')

        df_proposicoes_tecnologia = pd.DataFrame().from_dict(
            json.loads(response.text)['dados'])

        df_proposicoes_concat = pd.concat(
            [df_proposicoes_economia, df_proposicoes_educacao,
                df_proposicoes_tecnologia],
            ignore_index=True
        )

        list_proposicoes = []

        # Recupera as informações detalhadas de cada proposição do endpoint /proposicoes/{id}

        for id in tqdm(df_proposicoes_concat.id.unique()):
            response = requests.get(
                url=f'{url_base}/proposicoes/{id}')

            if not response.ok:
                raise Exception('Nao foi possivel recuperar os dados')

            list_proposicoes.append(json.loads(response.text)['dados'])

        df_proposicoes = pd.DataFrame().from_dict(list_proposicoes)

        # Merge dos dados das proposições com os dados das proposições detalhadas
        # Os dataframes possuem algumas colunas em comum, que são renomeadas para
        # depois serem removidas

        df_proposicoes_extended = df_proposicoes_concat.merge(
            df_proposicoes,
            on='id',
            suffixes=('', '_duplicated')
        )

        # Remoção das colunas duplicadas

        df_proposicoes_extended = df_proposicoes_extended.loc[
            :, ~df_proposicoes_extended.columns.str.endswith('_duplicated')
        ]

        df_proposicoes_extended.drop(
            columns=['uri',
                     'codTipo',
                     'numero',
                     'ano',
                     'uriOrgaoNumerador',
                     'uriAutores',
                     'uriPropPrincipal',
                     'uriPropAnterior',
                     'uriPropPosterior',
                     'urnFinal'],
            inplace=True)

        df_proposicoes_extended.to_parquet(
            os.path.join(DATA_PATH, 'proposicoes_deputados.parquet'))

# Classe para gerar a sumarização das proposições


class GenerateSumarizacao():
    def __init__(self):
        self.DATA_PATH = DATA_PATH

    def generate_sumarizacao(self):
        '''
        Função que gera um resumo para cada proposição no arquivo parquet e 
        salva os resumos em um arquivo json no diretório data.
        Args:
            None
        Returns:
            None
        '''
        input_parquet = os.path.join(
            self.DATA_PATH, 'proposicoes_deputados.parquet')
        output_json = os.path.join(
            self.DATA_PATH, 'sumarizacao_proposicoes.json')

        df = pd.read_parquet(input_parquet)

        # Junção das colunas do dataframe em uma única coluna de texto

        df['statusProposicao'] = df['statusProposicao'].fillna({}).apply(
            lambda x: x if isinstance(x, dict) else {}
        )

        df['siglaTipo'] = df['siglaTipo'].fillna('')
        df['ementa'] = df['ementa'].fillna('')
        df['dataApresentacao'] = df['dataApresentacao'].fillna('')
        df['descricaoTipo'] = df['descricaoTipo'].fillna('')
        df['ementaDetalhada'] = df['ementaDetalhada'].fillna('')
        df['keywords'] = df['keywords'].fillna('')
        df['justificativa'] = df['justificativa'].fillna('')

        df['texto'] = (
            'Sigla tipo de proposição: ' + df['siglaTipo'] + '\n' +
            'Ementa: ' + df['ementa'] + '\n' +
            'Data de Apresentação: ' + df['dataApresentacao'] + '\n' +
            'Descrição do Tipo: ' + df['descricaoTipo'] + '\n' +
            'Ementa Detalhada: ' + df['ementaDetalhada'] + '\n' +
            'Keywords: ' + df['keywords'] + '\n' +
            'Justificativa: ' + df['justificativa'] + '\n' +
            df['statusProposicao'].apply(lambda x: (
                'Ambito: ' + str(x.get('ambito', '')) + '\n' +
                'Apreciacao: ' + str(x.get('apreciacao', '')) + '\n' +
                'Situacao: ' + str(x.get('descricaoSituacao', '')) + '\n' +
                'Tramitacao: ' + str(x.get('descricaoTramitacao', '')) + '\n' +
                'Despacho: ' + str(x.get('despacho', '')) + '\n' +
                'Regime: ' + str(x.get('regime', '')) + '\n' +
                'Orgao: ' + str(x.get('siglaOrgao', ''))
            )))

        resumos = []

        total = len(df)

        # Delay entre as requisições para evitar exceder o limite de requisições por minuto da API do Gemini

        delay = 15

        # Loop para sumarizar cada proposição (linha do dataframe) da coluna de texto

        for index, row in df.iterrows():
            print(
                f"Sumarizando proposição {index + 1}/{total} - ID {row['id']}...")

            resumo = self.__process_text(row['texto'])

            resumos.append({
                'id': row['id'],
                'resumo': resumo
            })

            # Salvar os resultados parciais a cada 10 proposições sumarizadas

            if (index + 1) % 10 == 0:
                with open(output_json, 'w', encoding='utf-8') as f:
                    json.dump(resumos, f, ensure_ascii=False, indent=4)
                print(
                    f'Resultados parciais salvos após {index + 1} proposições')

            # Aguardar o delay entre as requisições, com exceção da última proposição do dataframe

            if index < total - 1:
                print(
                    f'Aguardando {delay} segundos antes da próxima requisição...')
                time.sleep(delay)

        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(resumos, f, ensure_ascii=False, indent=4)

    def __process_text(self, text, window_size=200, overlap_size=50):
        '''
        Classe privada para fazer chunking do texto e sumarizar cada chunk
        separadamente. Ao final, gera um resumo final combinando os resumos
        parciais.
        Args:
            text (str): Texto a ser sumarizado
            window_size (int): Tamanho da janela de chunking
            overlap_size (int): Tamanho da sobreposição entre as janelas
        Returns:
            str: Resumo final
        '''

        if isinstance(text, str):
            text = [text]

        chunks = [text[i:i+window_size]
                  for i in range(0, len(text), window_size-overlap_size)]

        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            print(f'Sumarizando chunk {i+1} de {len(chunks)}')

            text_lines = '\n'.join(chunk)
            chunk_prompt = f'''
            #proposições#
            {text_lines}
            ######
            Crie um resumo claro e objetivo.
            '''

            response = self.__generate_content(chunk_prompt)
            chunk_summaries.append(response)

        # Gerar resumo final
        summaries = '- ' + '\n- '.join(chunk_summaries)
        final_prompt = f'''
        Você é um assistente especializado em sumarização de textos legislativos.
        Abaixo estão os resumos parciais de uma proposição:
        {summaries}
        ######
        Combine os resumos e crie uma sumarização final. O resumo deve ser objetivo e claro,
        destacando o objetivo principal da proposição.
        '''

        return self.__generate_content(final_prompt)

    def __generate_content(self, prompt):
        '''
        Função privada para chamar o LLM.
        Args:
            prompt (str): Prompt para o LLM
        Returns:
            str: Resposta do LLM
        '''
        response = client.models.generate_content(
            model='gemini-1.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.3)
        )
        return response.text
