# Configurações de ambiente
import os
os.environ['PYTORCH_JIT'] = '0'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Importações da biblioteca padrão Python
import json
import yaml
from pathlib import Path

# 3. Terceiro: Bibliotecas básicas de terceiros
import streamlit as st
import pandas as pd
import plotly.express as px
from dotenv import load_dotenv

# PyTorch e relacionados
import torch
import transformers
transformers.logging.set_verbosity_error()

# LlamaIndex e componentes
from llama_index.llms.gemini import Gemini
from llama_index.core import VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.readers.json import JSONReader

DEVICE = 'cpu'

DATA_PATH = os.path.abspath('data')

MEMORY_TOKEN_LIMIT = 3900

ENV_PATH = os.path.abspath('.env')

load_dotenv(dotenv_path=ENV_PATH,
            override=True)

if 'chat_engine' not in st.session_state:
    st.session_state.chat_engine = None

@st.cache_resource
def initialize_chat_engine():
    '''Inicializa a engine de chat com contexto dos documentos e memória conversacional.'''
    try:
        Settings.embed_model = initialize_embeddings()

        json_reader = JSONReader(ensure_ascii='False')

        directory = Path(DATA_PATH)

        docs = []

        for json_file in directory.glob('*.json'):
            doc = json_reader.load_data(json_file)
            docs.extend(doc)
    
        index = VectorStoreIndex.from_documents(
            docs,
            transformations=[SentenceSplitter(
                chunk_size=2000, chunk_overlap=300)],
            show_progress=True  
        )

        memory = ChatMemoryBuffer.from_defaults(token_limit=MEMORY_TOKEN_LIMIT)

        return index.as_chat_engine(
            chat_mode='condense_plus_context',
            memory=memory,
            llm=initialize_llm(),
            context_prompt=('''Você é um chatbot especialista na câmara dos deputados, capaz de ter conversas normais e prover informações sobre o contexto abaixo:
                {context_str}
                Aqui estão as instruções para responder as perguntas dos usuários:
                Para cada pergunta recebida, você deve seguir este processo de auto-questionamento:
                1. Primeira pergunta: "Que tipo de dados da Câmara eu preciso consultar para responder esta questão?"
                Resposta: [Identifique se precisa consultar dados sobre deputados, despesas ou proposições]

                2. Segunda pergunta: "Quais informações específicas devo procurar nesses dados?"
                Resposta: [Liste os campos e informações relevantes necessários]

                3. Terceira pergunta: "Como devo analisar e processar essas informações?"
                Resposta: [Descreva o método de análise e processamento]

                4. Quarta pergunta: "Existem contextos ou considerações adicionais importantes?"
                Resposta: [Identifique fatores contextuais relevantes]

                Após responder estas perguntas para si mesmo, forneça uma resposta completa e estruturada para o usuário que:
                - Apresente os dados e análises solicitados
                - Explique quaisquer contextos importantes
                - Inclua estatísticas relevantes quando apropriado
                - Mencione limitações ou ressalvas quando necessário 

                Mantenha suas respostas:
                - Objetivas e factuais
                - Baseadas estritamente nos dados disponíveis
                - Imparciais
                - Claras e diretas

                Observação: Não exiba para o usuário os passos de auto-questionamento acima. Use-os apenas para orientar suas respostas.
                            A resposta deve ser com uma linguagem natural e direta, sem referências explícitas a esses passos.           
                '''),
            verbose=False,
        )
    except Exception as e:
        st.error(f"Erro ao inicializar chat engine: {str(e)}")
        raise


if 'messages' not in st.session_state:
    st.session_state.messages = []


@st.cache_data
def initialize_llm() -> Gemini:
    '''Inicializa o modelo Gemini com configurações otimizadas para RAG.'''
    generation_config = {
        'temperature': 0.4,
        'top_p': 1,
        'top_k': 40,
    }
    safety_settings = [
        {'category': 'HARM_CATEGORY_HARASSMENT',
            'threshold': 'BLOCK_MEDIUM_AND_ABOVE'},
        {'category': 'HARM_CATEGORY_HATE_SPEECH',
            'threshold': 'BLOCK_MEDIUM_AND_ABOVE'},
        {'category': 'HARM_CATEGORY_SEXUALLY_EXPLICIT',
            'threshold': 'BLOCK_MEDIUM_AND_ABOVE'},
        {'category': 'HARM_CATEGORY_DANGEROUS_CONTENT',
            'threshold': 'BLOCK_MEDIUM_AND_ABOVE'},
    ]
    return Gemini(
        model='models/gemini-1.5-flash',
        generation_config=generation_config,
        safety_settings=safety_settings,
        transport='rest',
        api_key=os.getenv('GEMINI_API_KEY')
    )


@st.cache_resource  
def initialize_embeddings() -> HuggingFaceEmbedding:
    '''Inicializa HuggingFaceEmbedding com configurações otimizadas para RAG.'''
    try:
        return HuggingFaceEmbedding(
            model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
            query_instruction='Represent the question for retrieving supporting documents: ',
            text_instruction='Represent the document for retrieval: ',
            normalize=True,
            embed_batch_size=32,
            device='cpu',  
            model_kwargs={
                'trust_remote_code': True,
                'torch_dtype': torch.float32,  
                'low_cpu_mem_usage': True,     
            },
        )
    except Exception as e:
        st.error(f"Erro ao inicializar embeddings: {str(e)}")
        raise


@st.cache_data
def load_parquet_data(filepath):
    try:
        return pd.read_parquet(filepath)
    except Exception as e:
        st.error(f'Error loading parquet file {filepath}: {e}')
        return None


@st.cache_data
def load_json_data(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        st.error(f'Error loading JSON file {filepath}: {e}')
        return None


@st.cache_data
def load_yaml_data(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        st.error(f'Error loading YAML file {filepath}: {e}')
        return None


st.set_page_config(page_title='Câmara dos Deputados',
                   page_icon='🏛️', layout='centered')


tab1, tab2, tab3, tab4 = st.tabs(
    ['Overview', 'Despesas', 'Proposições', 'Q&A'])


with tab1:
    st.title('Análise da Câmara dos Deputados')
    st.write('Análises geradas com os dados públicos da API da Câmara dos Deputados com ajuda de diferentes técnicas de Prompting.')

    with st.spinner('Carregando configurações...'):
        config = load_yaml_data('./data/config.yaml')
        if config:
            st.write(config['overview_summary'])

    st.subheader(
        'Gráfico gerado por LLM da distribuição de deputados por partido')
    try:
        st.image('./docs/distribuicao_deputados.png')
    except Exception as e:
        st.error(f'Error loading image: {e}')

    with st.spinner('Carregando insights de distribuição...'):
        insights_data = load_json_data(
            './data/insights_distribuicao_deputados.json')
        if insights_data:
            st.subheader(
                "Insights distribuição dos deputados: Governabilidade")
            st.write(insights_data['governabilidade'][0]['primeiro_insight'])
            st.write(insights_data['governabilidade'][1]['segundo_insight'])
            st.write(insights_data['governabilidade'][2]['terceiro_insight'])

            st.subheader(
                'Insights distribuição dos deputados: Pauta Legislativa')
            st.write(insights_data['pauta_legislativa'][0]['primeiro_insight'])
            st.write(insights_data['pauta_legislativa'][1]['segundo_insight'])
            st.write(insights_data['pauta_legislativa'][2]['terceiro_insight'])

            st.subheader(
                'Insights distribuição dos deputados: Representatividade Popular')
            st.write(
                insights_data['representatividade_popular'][0]['primeiro_insight'])
            st.write(
                insights_data['representatividade_popular'][1]['segundo_insight'])
            st.write(
                insights_data['representatividade_popular'][2]['terceiro_insight'])


with tab2:
    st.title('Despesas')

    with st.spinner('Carregando insights de despesas...'):
        insights_despesas = load_json_data(
            './data/insights_despesas_deputados.json')
        if insights_despesas:
            st.subheader('Insights por tipo de despesa')
            for insight in insights_despesas['insights_analise_tipo_despesa']:
                st.write(insight)

            st.subheader('Insights despesas por partido')
            for insight in insights_despesas['insights_analise_partido']:
                st.write(insight)

            st.subheader('Insights despesas por UF')
            for insight in insights_despesas['insights_analise_uf']:
                st.write(insight)

    with st.spinner('Carregando dados de despesas...'):
        df_despesas = load_parquet_data(
            './data/serie_despesas_diárias_deputados.parquet')

        if df_despesas is not None:
            deputados = sorted(df_despesas['nome'].unique())
            selected_deputy = st.selectbox('Selecione um deputado:', deputados)

            df_filtered = df_despesas[df_despesas['nome']
                                      == selected_deputy].copy()

            df_filtered['dataDocumento'] = pd.to_datetime(
                df_filtered['dataDocumento'])
            df_filtered['Mês'] = df_filtered['dataDocumento'].dt.to_period(
                'M')
            df_filtered['Valor'] = df_filtered['valorLiquido']

            df_monthly = df_filtered.groupby(
                'Mês')['Valor'].sum().reset_index()
            df_monthly['Mês'] = df_monthly['Mês'].dt.to_timestamp()

            st.subheader("Despesas por deputado por mês (2024)")
            fig = px.bar(
                df_monthly,
                x='Mês',
                y='Valor',
                title=f'Despesas de {selected_deputy} por mês'
            )
            st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.title('Proposições')
    st.subheader(
        'Proposições dos deputados sobre os temas: Economia; Educação; Ciência, Tecnologia e Inovação.')

    with st.spinner('Carregando dados de proposições...'):
        df_proposicoes = load_parquet_data(
            './data/proposicoes_deputados.parquet')
        if df_proposicoes is not None:
            st.dataframe(df_proposicoes, use_container_width=True)

    with st.spinner('Carregando resumos de proposições...'):
        proposicoes_data = load_json_data(
            './data/sumarizacao_proposicoes.json')
        if proposicoes_data:
            st.subheader("Resumo das proposições dos deputados")
            for proposicao in proposicoes_data:
                st.write(
                    f"Proposição ID {proposicao['id']}: {proposicao['resumo']}")

with tab4:
    st.title('🤖Converse com o ChatJurídico, seu jurista especialista na Câmara dos Deputados!')
    st.info(
    'Esse chatbot se lembra da conversa e produz respostas de acordo com o contexto')

    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])  

    if user_input := st.chat_input('Faça uma pergunta'):
        st.session_state.messages.append(
            {'role': 'user', 'content': user_input})

        with st.chat_message('user'):
            st.markdown(user_input)

        with st.chat_message('assistant'):
            try:
                with st.spinner('Pensando...'):
                    chat_engine = initialize_chat_engine()
                    response = chat_engine.chat(user_input)

                    
                    response_text = str(response).strip()
                    if not response_text:
                        response_text = "Desculpe, não consegui gerar uma resposta. Pode reformular sua pergunta?"

                    st.markdown(response_text)
                    st.session_state.messages.append(
                        {'role': 'assistant', 'content': response_text})

            except Exception as e:
                st.error(f'Erro: {str(e)}')
                st.error('Tentando reinicializar o sistema...')

                
                st.cache_resource.clear()
                try:
                    chat_engine = initialize_chat_engine()
                    response = chat_engine.chat(user_input)
                    st.markdown(str(response))
                except Exception as e2:
                    st.error(f'Erro persistente: {str(e2)}')
                    st.session_state.messages.append(
                        {'role': 'assistant', 'content': 'Desculpe, ocorreu um erro ao processar sua pergunta. Por favor, tente novamente mais tarde.'})
