from .dados_deputados import ProcessDadosDeputados
from .dados_despesas import ProcessDespesas
from .dados_proposicoes import ProcessProposicoes, GenerateSumarizacao

# Script para organizar as classes e funções dentro das classes
# de forma que nenhuma função que necessite do output de outra função
# seja chamada antes da função que gera o output


def deputados():
    dados_deputados = ProcessDadosDeputados()
    dados_deputados.get_deputados()
    dados_deputados.distribuicao_partido()
    dados_deputados.distribuicao_insights()


def despesas():
    despesas_deputados = ProcessDespesas()
    despesas_deputados.get_despesas()
    despesas_deputados.generate_analises()
    despesas_deputados.generate_insights()


def proposicoes():
    proposicoes = ProcessProposicoes()
    proposicoes.get_proposicoes()
    sumarizacao = GenerateSumarizacao()
    sumarizacao.generate_sumarizacao()


def main():
    deputados()
    despesas()
    proposicoes()


if __name__ == '__main__':
    main()
