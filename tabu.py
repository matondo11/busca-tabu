"""Demonstração de Busca Tabu para o Problema do Caixeiro-Viajante.

O programa gera cidades em um plano, procura uma rota curta que visite cada
cidade uma vez e retorne ao ponto inicial, e apresenta a evolução da busca em
gráficos. Ao final, compara a solução heurística com uma solução exata obtida
por enumeração de todas as rotas possíveis.

A Busca Tabu é uma heurística: ela explora vizinhos de uma rota e mantém uma
lista de movimentos recentes proibidos temporariamente. Isso ajuda a evitar
ciclos e a escapar de ótimos locais, mas não garante, por si só, a melhor rota
possível.

Dependências:
    matplotlib: exibição dos gráficos.

Execução:
    python tabu.py
"""

import itertools
import math
import random

import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Parâmetros do problema
# ---------------------------------------------------------------------------

SEMENTE_ALEATORIA = 42
"""int: Semente fixa para que cidades e rota inicial sejam reproduzíveis."""

NUMERO_DE_CIDADES = 10
"""int: Quantidade de cidades criadas para o problema."""

LIMITE_INFERIOR_COORDENADA = 10.0
"""float: Menor coordenada X ou Y permitida para uma cidade."""

LIMITE_SUPERIOR_COORDENADA = 90.0
"""float: Maior coordenada X ou Y permitida para uma cidade."""

TAMANHO_MAXIMO_LISTA_TABU = 5
"""int: Número máximo de movimentos recentes mantidos como proibidos."""

NUMERO_MAXIMO_ITERACOES = 100
"""int: Número máximo de atualizações da Busca Tabu."""

TOLERANCIA_COMPARACAO = 1e-12
"""float: Margem numérica para comparar distâncias de ponto flutuante."""

INTERVALO_ANIMACAO_SEGUNDOS = 0.05
"""float: Pausa entre atualizações dos gráficos, em segundos."""


# ---------------------------------------------------------------------------
# Preparação dos dados e cálculo da rota
# ---------------------------------------------------------------------------

def gerar_cidades(numero_de_cidades, gerador_aleatorio):
    """Cria cidades com coordenadas aleatórias dentro da área definida.

    Args:
        numero_de_cidades (int): Quantidade de pontos a criar.
        gerador_aleatorio (random.Random): Gerador usado para obter
            coordenadas reproduzíveis.

    Returns:
        list[tuple[float, float]]: Coordenadas das cidades. A posição de cada
        tupla na lista é o identificador da cidade; por exemplo, a tupla no
        índice 0 representa a cidade C0.
    """
    return [
        (
            gerador_aleatorio.uniform(
                LIMITE_INFERIOR_COORDENADA,
                LIMITE_SUPERIOR_COORDENADA,
            ),
            gerador_aleatorio.uniform(
                LIMITE_INFERIOR_COORDENADA,
                LIMITE_SUPERIOR_COORDENADA,
            ),
        )
        for _ in range(numero_de_cidades)
    ]


def calcular_distancia_total(rota, cidades):
    """Calcula o comprimento de uma rota circular.

    A rota é representada por uma lista de índices de cidades. A última cidade
    também é ligada à primeira, completando o ciclo do Caixeiro-Viajante.

    Args:
        rota (list[int]): Ordem em que as cidades são visitadas.
        cidades (list[tuple[float, float]]): Coordenadas indexadas pelo
            identificador de cada cidade.

    Returns:
        float: Soma das distâncias euclidianas entre cidades consecutivas,
        incluindo o trecho de retorno ao início.
    """
    return sum(
        math.dist(cidades[rota[indice]], cidades[rota[(indice + 1) % len(rota)]])
        for indice in range(len(rota))
    )


def encontrar_melhor_rota_exata(cidades):
    """Encontra a rota ótima avaliando todas as permutações possíveis.

    A primeira cidade é fixada como início para não avaliar várias versões da
    mesma rota que diferem apenas pelo ponto de partida. Este método é útil
    para comparação em problemas pequenos, mas seu custo cresce
    fatorialmente com o número de cidades.

    Args:
        cidades (list[tuple[float, float]]): Coordenadas das cidades.

    Returns:
        float: Menor distância total entre as rotas avaliadas.
    """
    indices_das_demais_cidades = range(1, len(cidades))
    return min(
        calcular_distancia_total([0, *permutacao], cidades)
        for permutacao in itertools.permutations(indices_das_demais_cidades)
    )


# ---------------------------------------------------------------------------
# Visualização da execução
# ---------------------------------------------------------------------------

def preparar_graficos(cidades):
    """Cria a janela e retorna os elementos necessários para desenhar os dados.

    Args:
        cidades (list[tuple[float, float]]): Coordenadas que serão mostradas.

    Returns:
        tuple: Figura, eixos dos gráficos e listas separadas de coordenadas X
        e Y, nesta ordem.
    """
    plt.ion()
    figura, (eixo_rota, eixo_evolucao) = plt.subplots(1, 2, figsize=(14, 6))
    coordenadas_x = [coordenada_x for coordenada_x, _ in cidades]
    coordenadas_y = [coordenada_y for _, coordenada_y in cidades]
    return figura, eixo_rota, eixo_evolucao, coordenadas_x, coordenadas_y


def atualizar_graficos(
    numero_iteracao,
    cidades,
    melhor_rota,
    historico_rota_atual,
    historico_melhor_rota,
    eixo_rota,
    eixo_evolucao,
    coordenadas_x,
    coordenadas_y,
):
    """Atualiza o desenho da rota e os gráficos de evolução da busca.

    Args:
        numero_iteracao (int): Iteração que está sendo exibida.
        cidades (list[tuple[float, float]]): Coordenadas das cidades.
        melhor_rota (list[int]): Melhor ordem de visita encontrada até agora.
        historico_rota_atual (list[float]): Distância da rota escolhida em
            cada iteração concluída.
        historico_melhor_rota (list[float]): Melhor distância global registrada
            em cada iteração.
        eixo_rota: Eixo Matplotlib usado para desenhar o mapa das cidades.
        eixo_evolucao: Eixo Matplotlib usado para desenhar as distâncias.
        coordenadas_x (list[float]): Coordenadas horizontais das cidades.
        coordenadas_y (list[float]): Coordenadas verticais das cidades.
    """
    eixo_rota.clear()
    eixo_evolucao.clear()

    rota_com_retorno = [*melhor_rota, melhor_rota[0]]
    rota_x = [cidades[indice][0] for indice in rota_com_retorno]
    rota_y = [cidades[indice][1] for indice in rota_com_retorno]

    eixo_rota.plot(
        rota_x,
        rota_y,
        "o-",
        color="crimson",
        linewidth=2,
        markersize=8,
        label="Melhor rota",
    )
    eixo_rota.scatter(coordenadas_x, coordenadas_y, color="blue", s=100, zorder=5)

    for indice_cidade, (coordenada_x, coordenada_y) in enumerate(cidades):
        eixo_rota.text(
            coordenada_x + 1,
            coordenada_y + 1,
            f"C{indice_cidade}",
            fontsize=11,
            fontweight="bold",
        )

    distancia_melhor_rota = historico_melhor_rota[-1]
    eixo_rota.set_title(
        f"Iteração {numero_iteracao} / {NUMERO_MAXIMO_ITERACOES}\n"
        f"Melhor distância: {distancia_melhor_rota:.2f}",
        fontsize=12,
    )
    eixo_rota.set_xlim(0, 100)
    eixo_rota.set_ylim(0, 100)
    eixo_rota.grid(True, linestyle="--", alpha=0.5)

    eixo_evolucao.plot(
        historico_rota_atual,
        label="Distância da rota atual",
        color="gray",
        alpha=0.6,
        linestyle=":",
    )
    eixo_evolucao.plot(
        historico_melhor_rota,
        label="Melhor distância encontrada",
        color="green",
        linewidth=2,
    )
    eixo_evolucao.set_title("Evolução da distância por iteração", fontsize=12)
    eixo_evolucao.set_xlabel("Iteração")
    eixo_evolucao.set_ylabel("Distância total")
    eixo_evolucao.legend()
    eixo_evolucao.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.pause(INTERVALO_ANIMACAO_SEGUNDOS)


# ---------------------------------------------------------------------------
# Busca Tabu
# ---------------------------------------------------------------------------

def executar_busca_tabu(cidades, gerador_aleatorio):
    """Procura uma rota curta usando trocas de posição e memória tabu.

    Em cada iteração, são avaliadas rotas vizinhas, criadas pela troca de duas
    cidades na rota atual. O melhor vizinho permitido é escolhido. Um movimento
    tabu só pode ser aceito quando satisfaz o critério de aspiração, isto é,
    quando produz uma solução melhor do que a melhor conhecida globalmente.

    Args:
        cidades (list[tuple[float, float]]): Coordenadas das cidades.
        gerador_aleatorio (random.Random): Gerador usado para embaralhar a rota
            inicial.

    Returns:
        tuple: (rota_inicial, melhor_rota, historico_rota_atual,
        historico_melhor_rota), em que as rotas são listas de índices e os
        históricos são listas de distâncias por iteração.
    """
    numero_de_cidades = len(cidades)

    # Cada índice representa uma cidade e a lista define sua ordem de visita.
    rota_inicial = list(range(numero_de_cidades))
    gerador_aleatorio.shuffle(rota_inicial)

    rota_atual = rota_inicial.copy()
    melhor_rota = rota_atual.copy()

    # Cada movimento tabu é um par ordenado com os índices das cidades
    # trocadas. A lista funciona como uma fila de tamanho limitado.
    lista_tabu = []

    distancia_inicial = calcular_distancia_total(rota_inicial, cidades)
    historico_rota_atual = [distancia_inicial]
    historico_melhor_rota = [distancia_inicial]

    (
        _,
        eixo_rota,
        eixo_evolucao,
        coordenadas_x,
        coordenadas_y,
    ) = preparar_graficos(cidades)

    for numero_iteracao in range(1, NUMERO_MAXIMO_ITERACOES + 1):
        melhor_rota_candidata = None
        menor_distancia_candidata = float("inf")
        melhor_movimento_candidato = None
        distancia_melhor_rota = calcular_distancia_total(melhor_rota, cidades)

        # Gera a vizinhança: cada vizinho troca duas posições da rota atual.
        for primeira_posicao in range(numero_de_cidades):
            for segunda_posicao in range(primeira_posicao + 1, numero_de_cidades):
                rota_vizinha = rota_atual.copy()
                rota_vizinha[primeira_posicao], rota_vizinha[segunda_posicao] = (
                    rota_vizinha[segunda_posicao],
                    rota_vizinha[primeira_posicao],
                )

                movimento = tuple(
                    sorted(
                        (
                            rota_atual[primeira_posicao],
                            rota_atual[segunda_posicao],
                        )
                    )
                )
                distancia_vizinha = calcular_distancia_total(rota_vizinha, cidades)

                # Aspiração permite ignorar a proibição tabu se houver melhora
                # sobre a melhor rota global conhecida.
                atende_aspiracao = (
                    distancia_vizinha
                    < distancia_melhor_rota - TOLERANCIA_COMPARACAO
                )
                movimento_eh_permitido = movimento not in lista_tabu or atende_aspiracao

                if (
                    movimento_eh_permitido
                    and distancia_vizinha < menor_distancia_candidata
                ):
                    melhor_rota_candidata = rota_vizinha
                    menor_distancia_candidata = distancia_vizinha
                    melhor_movimento_candidato = movimento

        if melhor_rota_candidata is None:
            print("Busca encerrada: nenhum vizinho admissível foi encontrado.")
            break

        # A rota escolhida passa a ser o estado atual, e o movimento é
        # temporariamente proibido para reduzir a chance de ciclos.
        rota_atual = melhor_rota_candidata
        lista_tabu.append(melhor_movimento_candidato)
        if len(lista_tabu) > TAMANHO_MAXIMO_LISTA_TABU:
            lista_tabu.pop(0)

        distancia_atual = calcular_distancia_total(rota_atual, cidades)
        distancia_melhor_rota = calcular_distancia_total(melhor_rota, cidades)
        if distancia_atual < distancia_melhor_rota - TOLERANCIA_COMPARACAO:
            melhor_rota = rota_atual.copy()
            distancia_melhor_rota = distancia_atual

        historico_rota_atual.append(distancia_atual)
        historico_melhor_rota.append(distancia_melhor_rota)

        atualizar_graficos(
            numero_iteracao=numero_iteracao,
            cidades=cidades,
            melhor_rota=melhor_rota,
            historico_rota_atual=historico_rota_atual,
            historico_melhor_rota=historico_melhor_rota,
            eixo_rota=eixo_rota,
            eixo_evolucao=eixo_evolucao,
            coordenadas_x=coordenadas_x,
            coordenadas_y=coordenadas_y,
        )

    plt.ioff()
    return (
        rota_inicial,
        melhor_rota,
        historico_rota_atual,
        historico_melhor_rota,
    )


def exibir_resultados(
    rota_inicial,
    melhor_rota,
    historico_rota_atual,
    cidades,
):
    """Imprime um resumo da execução e mantém a janela dos gráficos aberta.

    Args:
        rota_inicial (list[int]): Ordem aleatória usada como ponto de partida.
        melhor_rota (list[int]): Melhor ordem de visita encontrada pela busca.
        historico_rota_atual (list[float]): Distâncias da rota atual, incluindo
            o valor inicial na primeira posição.
        cidades (list[tuple[float, float]]): Coordenadas das cidades.
    """
    distancia_inicial = historico_rota_atual[0]
    distancia_melhor_rota = calcular_distancia_total(melhor_rota, cidades)
    distancia_otima_exata = encontrar_melhor_rota_exata(cidades)

    if distancia_inicial > 0:
        percentual_melhoria = (
            100 * (distancia_inicial - distancia_melhor_rota) / distancia_inicial
        )
    else:
        percentual_melhoria = 0.0

    print("\n--- RESULTADOS DA EXECUÇÃO ---")
    print(f"Rota inicial: {rota_inicial}")
    print(f"Distância inicial: {distancia_inicial:.2f}")
    print(f"Melhor rota encontrada pela Busca Tabu: {melhor_rota}")
    print(f"Melhor distância encontrada (Busca Tabu): {distancia_melhor_rota:.2f}")
    print(f"Ótimo exato (enumeração): {distancia_otima_exata:.2f}")
    print(f"Melhoria em relação à rota inicial: {percentual_melhoria:.2f}%")

    plt.show()


def main():
    """Coordena a criação do problema, a busca e a apresentação dos resultados."""
    gerador_aleatorio = random.Random(SEMENTE_ALEATORIA)
    cidades = gerar_cidades(NUMERO_DE_CIDADES, gerador_aleatorio)
    (
        rota_inicial,
        melhor_rota,
        historico_rota_atual,
        _,
    ) = executar_busca_tabu(cidades, gerador_aleatorio)

    exibir_resultados(
        rota_inicial=rota_inicial,
        melhor_rota=melhor_rota,
        historico_rota_atual=historico_rota_atual,
        cidades=cidades,
    )


if __name__ == "__main__":
    main()
