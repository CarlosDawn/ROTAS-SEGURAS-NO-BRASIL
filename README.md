# ROTAS-SEGURAS

Projeto de disciplina extensionista do Centro Universitário Estácio, realizado pelos estudantes Carlos Yuri e Francisco Neto do curso de Ciências da Computação.

O projeto foi realizado com o auxílio do Claude para criação e documentação do código principal, posteriormente revisado e alterado pelos estudantes. O objetivo é criar um script que seja possível realizar os mesmos tratamentos de dados e resultados com arquivos futuros. 

O projeto análisa os dados de acidentes nas rodovias federarias ao redor do Brasil para apontar as principas causas de acidentes, os tipos mais comuns, auxiliando onde é necessário aumentar a atenção e prioridade para resolução. Por se tratar de dados apenas de janeiro a agosto de 2026, ainda é necessário um arquivo com mais dados que pode ser obtido posteriormente e rodado no código normalmente.

## Estrutura do repositório
<code>
projeto-acidentes-2026/
├── main.py                              # script principal (Partes 1 a 7)
├── requirements.txt                     # pandas, numpy, duckdb, matplotlib
├── README.md
├── ambiente/
│   └── acidentes2026_causas_tipos.csv   # dados de entrada (não versionar)
└── graficos_resultantes/                # criada automaticamente na execução
    ├── graficos/
    │   ├── 01_qualidade_ausentes.png
    │   ├── 02_panorama_geral.png
    │   ├── 04_ranking_hotspots.png
    │   ├── 10_fatores_ambientais.png
    │   ├── 11_tipo_e_causa.png
    │   ├── 13_cramers_v.png
    │   ├── 16_perfil_idade_sexo.png
    │   └── 17_perfil_usuario_veiculo.png
    └── resumo_executivo.txt
</code>

## Mapa detalhado: o que cada parte faz e produz
<code>
main.py
├── Parte 1 - Configuração, carga e qualidade
│   ├── 1.1 Bibliotecas e parâmetros ... JANELA_KM, MIN_ACID_HOTSPOT, TOP_N, MIN_N_FATOR, PESOS_UPS
│   │                                    funções: display(), salvar()
│   ├── 1.2 Carga com DuckDB ........... CSV -> tabela "bruto" (tudo como texto)
│   └── 1.3 Valores ausentes ........... gráfico 01_qualidade_ausentes
│
├── Parte 2 - Limpeza e modelagem
│   ├── 2.1 Tabela pessoas (SQL) ....... tipos, datas, coordenadas, deduplicação
│   ├── 2.2 Variáveis por pessoa ....... gravidade_pessoa, faixa_etaria, grupo_veiculo, idade_veiculo
│   └── 2.3 Tabela acidentes ........... grave, fatal, ups, mês, faixa_horaria, flags (moto, caminhão...)
│
├── Parte 3 - Visão geral (EDA) ........ gráfico 02_panorama_geral
│
├── Parte 4 - Hotspots (usa: acidentes)
│   ├── 4.1 Segmentação ................ seg_km, janelas_moveis(), selecionar_sem_sobreposicao()
│   ├── 4.2 Teste estatístico .......... poisson_sf(), razão observado/esperado, Bonferroni
│   └── 4.3 Gráfico .................... 04_ranking_hotspots
│
├── Parte 5 - Fatores de gravidade (usa: acidentes)
│   ├── 5.1 Funções .................... wilson(), taxa_por_fator(), grafico_taxas()
│   ├── 5.2 Fatores ambientais ......... 10_fatores_ambientais
│   ├── 5.3 Tipo e causa ............... 11_tipo_e_causa
│   └── 5.4 V de Cramér ................ cramers_v(), 13_cramers_v
│
├── Parte 6 - Perfis de risco (usa: pessoas)
│   ├── 6.1 Base de pessoas válidas .... pv, BASE_P
│   ├── 6.2 Faixa etária e sexo ........ 16_perfil_idade_sexo
│   └── 6.3 Usuário e veículo .......... 17_perfil_usuario_veiculo, tabela de letalidade
│
└── Parte 7 - Resumo executivo ......... melhores_categorias(), linhas_ou_aviso() -> resumo_executivo.txt
</code>

FONTE DE DADOS: https://www.gov.br/prf/pt-br/acesso-a-informacao/dados-abertos/dados-abertos-da-prf - arquivo utilizado: Documento CSV de Acidentes 2026 (Agrupados por pessoa - Todas as causas e tipos de acidentes)
