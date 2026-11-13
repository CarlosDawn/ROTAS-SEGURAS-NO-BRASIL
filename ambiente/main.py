"""
ANÁLISE DE ACIDENTES RODOVIÁRIOS 2026
Hotspots, fatores associados à gravidade e perfis de risco
=====================================================================
Bibliotecas: pandas, numpy, duckdb, matplotlib  (folium é opcional)

COMO RODAR NO PYCHARM
  1. Coloque este arquivo e o CSV (acidentes2026.csv) na mesma pasta do projeto.
  2. File > Settings > Project > Python Interpreter > instale os pacotes de requirements.txt
     (ou no terminal do PyCharm:  pip install -r requirements.txt).
  3. Ajuste a seção CONFIGURAÇÃO (nome do arquivo, separador, encoding).
  4. Run > Run 'analise_acidentes_2026'  (o script inteiro)
     OU execute seção por seção: o PyCharm reconhece os marcadores "# %%" como células
     (botão ▶ ao lado de cada célula; os gráficos aparecem na aba "Plots"/SciView).
  5. Também aceita o caminho do CSV como argumento:  python analise_acidentes_2026.py meu_arquivo.csv

SAÍDAS (criadas na pasta "saida/" ao lado do script)
  graficos/*.png · tabelas/*.csv · *_silver.parquet · mapa_hotspots.html · resumo_executivo.txt

DICA: se abrir janelas demais, defina MOSTRAR_GRAFICOS = False (os PNG continuam sendo salvos).
"""


# %% [markdown]
# # Análise de Acidentes Rodoviários 2026
# ## Hotspots, fatores associados à gravidade e perfis de risco
#
# **Objetivo:** transformar o arquivo de ocorrências de 2026 (uma linha por pessoa envolvida) em evidências para orientar intervenções de segurança viária: **onde** agir (hotspots), **o que** aumenta a gravidade (fatores) e **para quem** agir (perfis de risco).
#
# **Como este notebook está organizado**
#
# | Parte | Conteúdo | Gráficos |
# |---|---|---|
# | 0 | Tecnologias e arquitetura recomendadas | – |
# | 1 | Configuração, carga (DuckDB) e qualidade dos dados | valores ausentes |
# | 2 | Limpeza e modelagem: tabela de **pessoas** e tabela de **acidentes** | – |
# | 3 | Visão geral (EDA) | mês, dia/hora, classificação, causas |
# | 4 | **Hotspots**: trechos km, concentração, mapa, recorrência, municípios, delegacias | 8 gráficos |
# | 5 | **Fatores associados à gravidade**: taxas, risco relativo, Cramér V, regressão logística | 6 gráficos |
# | 6 | **Perfis de risco**: pessoa, veículo, combinações e perfil dos hotspots | 6 gráficos |
# | 7 | **Matriz de intervenções** por hotspot | 1 gráfico |
# | 8 | Exportação, resumo executivo e limitações | – |
#
# > **Como usar:** ajuste apenas a célula de *Configuração* (caminho do arquivo, separador, encoding) e execute as células em ordem (`Kernel > Restart & Run All`). Todos os gráficos são salvos em `saida/graficos/` e as tabelas finais em `saida/tabelas/`.

# %% [markdown]
# ---
# # Parte 0 — Tecnologias e linguagens recomendadas
#
# ### Núcleo do projeto (o que este notebook usa)
# | Camada | Ferramenta | Por que usar |
# |---|---|---|
# | Linguagem principal | **Python 3.10+** | Simples de ler, enorme ecossistema de dados, padrão de mercado |
# | Linguagem de consulta | **SQL (via DuckDB)** | Agregações pesadas em milhões de linhas, direto no CSV/Parquet, sem servidor |
# | Manipulação | **pandas** | Limpeza, criação de variáveis, tabelas dinâmicas |
# | Cálculo numérico | **numpy** | Estatística vetorizada (IC de Wilson, regressão logística, Poisson) |
# | Gráficos | **matplotlib** | Controle total e gráficos reproduzíveis |
# | Ambiente | **Jupyter Notebook / JupyterLab / VS Code** | Código + texto + gráficos no mesmo documento |
#
# ### Extensões profissionais (opcionais)
# | Necessidade | Ferramenta |
# |---|---|
# | Mapas interativos | **folium** (usado aqui se instalado), **kepler.gl**, **QGIS** |
# | Geoprocessamento (snap em rodovia, buffers, Getis-Ord / KDE) | **geopandas**, **shapely**, **pysal/esda** |
# | Modelos estatísticos com p-valor e diagnósticos | **statsmodels** |
# | Modelos preditivos (gradient boosting, SHAP) | **scikit-learn**, **LightGBM**, **shap** |
# | Painel para gestores | **Streamlit**, **Power BI**, **Metabase**, **Apache Superset** |
# | Armazenamento eficiente | **Parquet** (colunar, 5 a 10 vezes menor que CSV) |
# | Reprodutibilidade | **Git**, `requirements.txt`/**uv**, **pytest** (testes de qualidade de dados) |
# | Orquestração (quando virar rotina mensal) | **Prefect**, **Airflow**, **dbt** (transformações SQL versionadas) |
#
# ### Arquitetura sugerida (camadas)
# ```
#  CSV bruto  ──►  BRONZE (cópia fiel, Parquet)
#                    │   tipagem, limpeza, deduplicação
#                    ▼
#                  SILVER  (tabela `pessoas` limpa)
#                    │   agregação por acidente / trecho / município
#                    ▼
#                  GOLD    (tabelas analíticas: hotspots, fatores, perfis)
#                    │
#                    ▼
#         Notebook (análise)  ►  Painel (Streamlit/Power BI)  ►  Relatório para gestores
# ```
#
# ### Instalação
# ```bash
# python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
# pip install pandas numpy duckdb matplotlib jupyterlab pyarrow
# pip install folium            # opcional (mapa interativo)
# ```

# %% [markdown]
# ---
# # Parte 1 — Configuração, carga e qualidade dos dados
#
# ### 1.1 Bibliotecas e parâmetros
# Todos os parâmetros ajustáveis ficam em um só lugar. Os **pesos UPS** (Unidade Padrão de Severidade) transformam acidentes com consequências diferentes em um único índice comparável: um acidente fatal "vale" mais que vários acidentes sem vítimas. Os valores 1/4/13 são os usuais em metodologias de rodovias federais; **confirme o padrão adotado pelo seu órgão** e ajuste se necessário.

# %%
import sys, warnings, math, itertools
from pathlib import Path

import numpy as np
import pandas as pd
import duckdb
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

warnings.filterwarnings("ignore")

# ------------------------- CONFIGURAÇÃO -------------------------
ARQUIVO   = "acidentes2026_todas_causas_tipos.csv"   # caminho do seu arquivo (relativo à pasta do script ou absoluto)
SEPARADOR = ";"                    # arquivos da PRF costumam usar ";"
ENCODING  = "latin-1"              # use "utf-8" se o arquivo estiver em UTF-8
ANO_REF   = 2026                   # ano de referência para calcular idade do veículo

JANELA_KM          = 5     # tamanho do trecho (km) para detectar hotspots
MIN_ACID_HOTSPOT   = 8     # mínimo de acidentes para um trecho ser candidato
TOP_N              = 15    # quantos itens mostrar nos rankings
MIN_N_FATOR        = 30    # mínimo de observações para exibir uma categoria
PESOS_UPS = {"sem_vitimas": 1, "feridos": 4, "fatal": 13}   # ajuste conforme a metodologia do órgão
# ----------------------------------------------------------------

MOSTRAR_GRAFICOS = True            # False = só salva os PNG (não abre janelas)

# Pasta do script (funciona no PyCharm e no Jupyter). Um caminho passado na linha de comando tem prioridade.
BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
if len(sys.argv) > 1 and not sys.argv[1].startswith("-") and sys.argv[1].lower().endswith((".csv", ".txt")):
    ARQUIVO = sys.argv[1]
ARQUIVO = (Path(ARQUIVO) if Path(ARQUIVO).is_absolute() else BASE_DIR / ARQUIVO).as_posix()

PASTA = BASE_DIR / "saida"
(PASTA / "graficos").mkdir(parents=True, exist_ok=True)
(PASTA / "tabelas").mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 100, "savefig.dpi": 150, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": .25, "axes.titleweight": "bold", "axes.titlesize": 12,
    "font.size": 10, "figure.autolayout": False,
})
COR_ALERTA, COR_NEUTRA, COR_DESTAQUE = "#c0392b", "#2e86c1", "#f39c12"
pd.set_option("display.max_columns", 60, "display.width", 200, "display.float_format", lambda x: f"{x:,.0f}" if float(x).is_integer() else f"{x:,.3f}")

def display(obj):
    """Mostra tabelas (DataFrame/Series) no console do PyCharm."""
    print(obj)

def salvar(fig, nome):
    """Salva o gráfico em PNG (alta resolução) e exibe."""
    fig.savefig(PASTA / "graficos" / f"{nome}.png", bbox_inches="tight")
    if MOSTRAR_GRAFICOS: plt.show()
    else: plt.close(fig)

# %% [markdown]
# ### 1.2 Carga do arquivo com DuckDB
# Lemos **tudo como texto** (`all_varchar`) e só depois convertemos tipos com `TRY_CAST`. Isso evita falhas por vírgula decimal (`"386,4"`), campos vazios ou valores inesperados. O DuckDB lê o CSV em paralelo e consegue processar arquivos muito maiores que a memória RAM.

# %%
con = duckdb.connect()   # banco em memória (use duckdb.connect("acidentes.duckdb") para persistir)

try:
    con.execute(f"""
        CREATE OR REPLACE TABLE bruto AS
        SELECT * FROM read_csv('{ARQUIVO}', delim='{SEPARADOR}', header=true, all_varchar=true,
                               encoding='{ENCODING}', normalize_names=true, ignore_errors=true)
    """)
except Exception as e:                       # plano B: pandas lê e entrega ao DuckDB
    print("DuckDB não conseguiu ler direto; usando pandas. Motivo:", str(e)[:120])
    _tmp = pd.read_csv(ARQUIVO, sep=SEPARADOR, encoding=ENCODING, dtype=str)
    _tmp.columns = [c.strip().lower() for c in _tmp.columns]
    con.register("_tmp", _tmp)
    con.execute("CREATE OR REPLACE TABLE bruto AS SELECT * FROM _tmp")

n_linhas = con.execute("SELECT COUNT(*) FROM bruto").fetchone()[0]
print(f"Linhas lidas: {n_linhas:,}")

ESPERADAS = ["id","pesid","data_inversa","dia_semana","horario","uf","br","km","municipio","causa_principal",
             "causa_acidente","ordem_tipo_acidente","tipo_acidente","classificacao_acidente","fase_dia","sentido_via",
             "condicao_metereologica","tipo_pista","tracado_via","uso_solo","id_veiculo","tipo_veiculo","marca",
             "ano_fabricacao_veiculo","tipo_envolvido","estado_fisico","idade","sexo","ilesos","feridos_leves",
             "feridos_graves","mortos","latitude","longitude","regional","delegacia","uop"]
presentes = [r[0] for r in con.execute("DESCRIBE bruto").fetchall()]
faltando = [c for c in ESPERADAS if c not in presentes]
print("Colunas faltando:", faltando if faltando else "nenhuma ✔")
display(con.execute("SELECT * FROM bruto LIMIT 5").df())

# %% [markdown]
# ### 1.3 Diagnóstico de qualidade (valores ausentes)
# Antes de analisar, medimos o quanto cada coluna está preenchida. Campos como `latitude`, `longitude`, `idade` e `ano_fabricacao_veiculo` costumam ter ausências, e isso afeta quais análises são confiáveis (por exemplo, mapas só usam registros com coordenadas).

# %%
# % de vazios por coluna, calculado em SQL (uma única passagem pelos dados)
partes = ",\n".join(
    f"100.0 * SUM(CASE WHEN {c} IS NULL OR TRIM({c}) = '' THEN 1 ELSE 0 END) / COUNT(*) AS {c}"
    for c in presentes
)
ausentes = con.execute(f"SELECT {partes} FROM bruto").df().T.rename(columns={0: "pct_vazio"}).sort_values("pct_vazio", ascending=False)

fig, ax = plt.subplots(figsize=(8, 7))
top = ausentes.head(20).iloc[::-1]
ax.barh(top.index, top.pct_vazio, color=np.where(top.pct_vazio > 5, COR_ALERTA, COR_NEUTRA))
ax.set_xlabel("% de registros vazios"); ax.set_title("Qualidade dos dados: colunas com mais valores ausentes")
ax.xaxis.set_major_formatter(mtick.PercentFormatter())
salvar(fig, "01_qualidade_ausentes")
display(ausentes.head(10))

# %% [markdown]
# ---
# # Parte 2 — Limpeza e modelagem dos dados
#
# ### 2.1 Tabela `pessoas` (silver)
# O arquivo tem **uma linha por pessoa**; por isso o mesmo `id` de acidente se repete. Regras aplicadas em SQL:
#
# 1. Converter tipos (`TRY_CAST`), trocando vírgula decimal por ponto em `km`, `latitude` e `longitude`.
# 2. Aceitar datas em `AAAA-MM-DD` ou `DD/MM/AAAA`.
# 3. Remover espaços e padronizar `uf` e `municipio` em maiúsculas.
# 4. Invalidar valores impossíveis: idade fora de 0–110, ano de fabricação fora de 1950–2027, coordenadas fora do Brasil.
# 5. Remover duplicatas exatas (`id` + `pesid` + `id_veiculo`).

# %%
COLS_TEXTO = ["causa_principal","causa_acidente","tipo_acidente","classificacao_acidente","fase_dia","sentido_via",
              "condicao_metereologica","tipo_pista","tracado_via","uso_solo","tipo_veiculo","marca","tipo_envolvido",
              "estado_fisico","sexo","regional","delegacia","uop"]
sel_texto = ",\n        ".join(f"NULLIF(TRIM({c}), '') AS {c}" for c in COLS_TEXTO)

con.execute(f"""
CREATE OR REPLACE TABLE pessoas_sql AS
WITH t AS (
    SELECT
        TRY_CAST(id AS BIGINT)                                          AS id,
        TRY_CAST(pesid AS BIGINT)                                       AS pesid,
        COALESCE(TRY_STRPTIME(SUBSTR(data_inversa, 1, 10), '%Y-%m-%d'),
                 TRY_STRPTIME(SUBSTR(data_inversa, 1, 10), '%d/%m/%Y'))::DATE AS data,
        TRY_CAST(SPLIT_PART(horario, ':', 1) AS INTEGER)                AS hora,
        UPPER(NULLIF(TRIM(uf), ''))                                     AS uf,
        TRY_CAST(TRY_CAST(REPLACE(br, ',', '.') AS DOUBLE) AS INTEGER)  AS br,
        TRY_CAST(REPLACE(km, ',', '.') AS DOUBLE)                       AS km,
        UPPER(NULLIF(TRIM(municipio), ''))                              AS municipio,
        TRY_CAST(ordem_tipo_acidente AS INTEGER)                        AS ordem_tipo,
        TRY_CAST(id_veiculo AS BIGINT)                                  AS id_veiculo,
        TRY_CAST(REPLACE(ano_fabricacao_veiculo, ',', '.') AS DOUBLE)   AS ano_fab,
        TRY_CAST(TRY_CAST(REPLACE(idade, ',', '.') AS DOUBLE) AS INTEGER) AS idade,
        COALESCE(TRY_CAST(ilesos AS INTEGER), 0)                        AS ilesos,
        COALESCE(TRY_CAST(feridos_leves AS INTEGER), 0)                 AS feridos_leves,
        COALESCE(TRY_CAST(feridos_graves AS INTEGER), 0)                AS feridos_graves,
        COALESCE(TRY_CAST(mortos AS INTEGER), 0)                        AS mortos,
        TRY_CAST(REPLACE(latitude, ',', '.') AS DOUBLE)                 AS lat_raw,
        TRY_CAST(REPLACE(longitude, ',', '.') AS DOUBLE)                AS lon_raw,
        {sel_texto}
    FROM bruto
)
SELECT * EXCLUDE (lat_raw, lon_raw, ano_fab, idade),
       CASE WHEN lat_raw BETWEEN -34 AND 6 AND lon_raw BETWEEN -74 AND -28 THEN lat_raw END AS latitude,
       CASE WHEN lat_raw BETWEEN -34 AND 6 AND lon_raw BETWEEN -74 AND -28 THEN lon_raw END AS longitude,
       CASE WHEN ano_fab BETWEEN 1950 AND 2027 THEN CAST(ano_fab AS INTEGER) END AS ano_fab,
       CASE WHEN idade BETWEEN 0 AND 110 THEN idade END AS idade
FROM t
WHERE id IS NOT NULL
QUALIFY ROW_NUMBER() OVER (PARTITION BY id, pesid, id_veiculo) = 1
""")

pessoas = con.execute("SELECT * FROM pessoas_sql").df()
print(f"Pessoas: {len(pessoas):,} | Acidentes distintos: {pessoas['id'].nunique():,} | "
      f"Período: {pessoas['data'].min().date()} a {pessoas['data'].max().date()}")
display(pessoas.head())

# %% [markdown]
# ### 2.2 Variáveis derivadas por pessoa
# Criamos grupos legíveis para análise: **gravidade individual**, **faixa etária**, **grupo do veículo** (moto, caminhão…) e **idade do veículo**. A classificação do veículo usa expressões regulares porque o texto original varia ("Motocicleta", "Motoneta", "Caminhão-trator"…).

# %%
p = pessoas  # alias curto

# Gravidade individual (hierarquia: óbito > grave > leve > ileso)
p["gravidade_pessoa"] = np.select(
    [p.mortos > 0, p.feridos_graves > 0, p.feridos_leves > 0, p.ilesos > 0],
    ["Óbito", "Ferido grave", "Ferido leve", "Ileso"], default="Ignorado")
p["pessoa_fatal"] = (p.mortos > 0)
p["pessoa_grave"] = (p.mortos + p.feridos_graves) > 0     # alvo: óbito OU lesão grave

# Faixa etária
p["faixa_etaria"] = pd.cut(p.idade, bins=[-1, 11, 17, 24, 34, 44, 54, 64, 74, 200],
                           labels=["0-11", "12-17", "18-24", "25-34", "35-44", "45-54", "55-64", "65-74", "75+"])

# Grupo do veículo (a ordem das regras importa: caminhonete/camioneta ANTES de caminhão)
tv = p.tipo_veiculo.fillna("").str.lower()
p["grupo_veiculo"] = np.select(
    [tv.str.contains("motocicleta|motoneta|ciclomotor|triciclo", regex=True),
     tv.str.contains("bicicleta"),
     tv.str.contains("caminhonete|camioneta|autom[óo]vel|utilit[áa]rio", regex=True),
     tv.str.contains("caminh[ãa]o|trator|cavalo|reboque|semi", regex=True),
     tv.str.contains("[ôo]nibus", regex=True),
     tv.str.contains("pedestre|^$", regex=True)],
    ["Motocicleta", "Bicicleta", "Automóvel/Utilitário", "Caminhão/Carga", "Ônibus", "Sem veículo"],
    default="Outros")

# Idade do veículo
p["idade_veiculo"] = (ANO_REF - p.ano_fab).clip(lower=0)
p["faixa_idade_veiculo"] = pd.cut(p.idade_veiculo, bins=[-1, 2, 5, 10, 15, 20, 100],
                                  labels=["0-2", "3-5", "6-10", "11-15", "16-20", "21+"])

display(p.gravidade_pessoa.value_counts().to_frame("pessoas"))
display(p.grupo_veiculo.value_counts().to_frame("pessoas"))

# %% [markdown]
# ### 2.3 Tabela `acidentes` (uma linha por ocorrência)
# Hotspots e fatores ambientais devem ser analisados **por acidente**, não por pessoa; caso contrário, um acidente com 5 ocupantes contaria 5 vezes. O DuckDB consulta o DataFrame `pessoas` diretamente e agrega em SQL:
#
# * `mode()` → causa mais frequente do acidente; `arg_min(tipo, ordem)` → tipo de acidente de menor ordem (o principal);
# * contagens de pessoas e veículos, soma de mortos/feridos e *flags* de composição (há moto? caminhão? pedestre?).

# %%
con.register("pessoas_df", pessoas)

acidentes = con.execute("""
SELECT
    id,
    ANY_VALUE(data) AS data, ANY_VALUE(hora) AS hora,
    ANY_VALUE(uf) AS uf, ANY_VALUE(br) AS br, ANY_VALUE(km) AS km, ANY_VALUE(municipio) AS municipio,
    MODE(causa_acidente)                                              AS causa_acidente,
    COALESCE(ARG_MIN(tipo_acidente, ordem_tipo), ANY_VALUE(tipo_acidente)) AS tipo_acidente,
    ANY_VALUE(classificacao_acidente) AS classificacao_acidente,
    ANY_VALUE(fase_dia) AS fase_dia, ANY_VALUE(sentido_via) AS sentido_via,
    ANY_VALUE(condicao_metereologica) AS condicao_metereologica,
    ANY_VALUE(tipo_pista) AS tipo_pista, ANY_VALUE(tracado_via) AS tracado_via, ANY_VALUE(uso_solo) AS uso_solo,
    ANY_VALUE(regional) AS regional, ANY_VALUE(delegacia) AS delegacia, ANY_VALUE(uop) AS uop,
    AVG(latitude) AS latitude, AVG(longitude) AS longitude,
    COUNT(*)                         AS n_pessoas,
    COUNT(DISTINCT id_veiculo)       AS n_veiculos,
    SUM(mortos)                      AS mortos,
    SUM(feridos_graves)              AS feridos_graves,
    SUM(feridos_leves)               AS feridos_leves,
    SUM(ilesos)                      AS ilesos,
    BOOL_OR(grupo_veiculo = 'Motocicleta')          AS tem_moto,
    BOOL_OR(grupo_veiculo = 'Caminhão/Carga')       AS tem_caminhao,
    BOOL_OR(grupo_veiculo = 'Bicicleta')            AS tem_bicicleta,
    BOOL_OR(tipo_envolvido ILIKE '%pedestre%')      AS tem_pedestre
FROM pessoas_df
GROUP BY id
""").df()

for c in ["n_pessoas", "n_veiculos", "mortos", "feridos_graves", "feridos_leves", "ilesos"]:
    acidentes[c] = acidentes[c].astype(int)          # SUM() do DuckDB pode voltar como float

a = acidentes
a["grave"] = (a.mortos + a.feridos_graves) > 0      # alvo principal: óbito OU lesão grave
a["fatal"] = a.mortos > 0
a["ups"]   = np.select([a.fatal, (a.feridos_graves + a.feridos_leves) > 0],
                       [PESOS_UPS["fatal"], PESOS_UPS["feridos"]], default=PESOS_UPS["sem_vitimas"])

# Variáveis temporais (dia da semana calculado a partir da data, independente do idioma do arquivo)
DIAS = ["Segunda", "Terça", "Quarta", "Quinta", "Sexta", "Sábado", "Domingo"]
a["dia_semana"] = pd.Categorical(a.data.dt.dayofweek.map(dict(enumerate(DIAS))), categories=DIAS, ordered=True)
a["mes"] = a.data.dt.month
a["fim_de_semana"] = a.data.dt.dayofweek >= 5
a["faixa_horaria"] = pd.cut(a.hora, bins=[-1, 5, 11, 17, 23], labels=["Madrugada (0-5h)", "Manhã (6-11h)", "Tarde (12-17h)", "Noite (18-23h)"])
a["n_veiculos_cat"] = a.n_veiculos.clip(upper=3).map({0: "0", 1: "1", 2: "2", 3: "3+"})

print(f"Acidentes: {len(a):,} | com óbito: {a.fatal.sum():,} ({a.fatal.mean():.1%}) | "
      f"graves (óbito ou lesão grave): {a.grave.sum():,} ({a.grave.mean():.1%})")
print(f"Total de mortos: {a.mortos.sum():,} | feridos graves: {a.feridos_graves.sum():,} | feridos leves: {a.feridos_leves.sum():,}")
display(a.head())

# %% [markdown]
# ---
# # Parte 3 — Visão geral (EDA)
# Antes de procurar padrões específicos, entendemos o panorama: **quando** os acidentes acontecem, **quão graves** são e **quais causas** predominam. Isso calibra as análises seguintes e revela erros grosseiros nos dados.
#
# ### 3.1 Painel geral (4 gráficos)

# %%
fig, axes = plt.subplots(2, 2, figsize=(15, 9))

# (a) Acidentes e óbitos por mês
mes = a.groupby("mes").agg(acidentes=("id", "size"), mortos=("mortos", "sum"))
ax = axes[0, 0]; ax.bar(mes.index, mes.acidentes, color=COR_NEUTRA, label="Acidentes")
ax.set_xlabel("Mês"); ax.set_ylabel("Acidentes"); ax.set_title("Evolução mensal")
ax2 = ax.twinx(); ax2.plot(mes.index, mes.mortos, color=COR_ALERTA, marker="o", label="Mortos"); ax2.set_ylabel("Mortos", color=COR_ALERTA)
ax2.grid(False); ax2.spines["right"].set_visible(True)

# (b) Classificação do acidente
cl = a.classificacao_acidente.value_counts()
ax = axes[0, 1]; ax.barh(cl.index[::-1], cl.values[::-1], color=[COR_ALERTA if "Fatais" in i else COR_NEUTRA for i in cl.index[::-1]])
for i, v in enumerate(cl.values[::-1]): ax.text(v, i, f" {v:,} ({v/len(a):.1%})", va="center")
ax.set_title("Classificação dos acidentes"); ax.set_xlim(0, cl.max() * 1.25)

# (c) Acidentes por hora do dia + % graves
h = a.groupby("hora").agg(n=("id", "size"), pct_grave=("grave", "mean"))
ax = axes[1, 0]; ax.bar(h.index, h.n, color=COR_NEUTRA); ax.set_xlabel("Hora do dia"); ax.set_ylabel("Acidentes")
ax.set_title("Acidentes por hora e % graves")
ax2 = ax.twinx(); ax2.plot(h.index, h.pct_grave * 100, color=COR_ALERTA, marker="o"); ax2.set_ylabel("% graves", color=COR_ALERTA); ax2.grid(False)
ax2.spines["right"].set_visible(True)

# (d) Top causas
c = a.causa_acidente.value_counts().head(10).iloc[::-1]
ax = axes[1, 1]; ax.barh(c.index, c.values, color=COR_NEUTRA); ax.set_title("Top 10 causas dos acidentes")
plt.tight_layout(); salvar(fig, "02_panorama_geral")

# %% [markdown]
# ---
# # Parte 4 — Hotspots (onde agir)
#
# Um **hotspot** é um local onde acidentes (e principalmente os graves) se concentram acima do que seria esperado ao acaso. Usaremos cinco lentes complementares:
#
# 1. **Trechos de rodovia** (BR + UF + km) com janela móvel e **índice UPS** (prioriza gravidade, não só quantidade).
# 2. **Concentração** (curva de Pareto): quanto da rodovia concentra a maior parte dos acidentes.
# 3. **Mapa** de densidade com coordenadas.
# 4. **Recorrência** mensal: o problema é crônico ou foi um evento isolado?
# 5. **Territórios administrativos**: municípios, delegacias e UOPs, para direcionar equipes.
#
# ### 4.1 Segmentação por km e janela móvel (SQL + pandas)
# Primeiro agregamos por quilômetro em SQL. Depois aplicamos uma **janela móvel** de `JANELA_KM` km: um acidente no km 215,9 e outro no 216,1 pertencem ao mesmo problema, mesmo caindo em "caixinhas" de 1 km diferentes. Por fim selecionamos os trechos de maior UPS **sem sobreposição**.

# %%
# 1) Agregação por UF + BR + km inteiro (DuckDB)
con.register("acidentes_df", a)
seg_km = con.execute("""
    SELECT uf, br, CAST(FLOOR(km) AS INTEGER) AS km_ini,
           COUNT(*) AS n_acid, SUM(mortos) AS mortos, SUM(feridos_graves) AS graves,
           SUM(feridos_leves) AS leves, SUM(ups) AS ups
    FROM acidentes_df
    WHERE br IS NOT NULL AND km IS NOT NULL AND uf IS NOT NULL
    GROUP BY uf, br, km_ini
""").df()
seg_km[["n_acid", "mortos", "graves", "leves", "ups"]] = seg_km[["n_acid", "mortos", "graves", "leves", "ups"]].astype(int)
print(f"Segmentos de 1 km com pelo menos 1 acidente: {len(seg_km):,}")

# 2) Janela móvel por rodovia
def janelas_moveis(seg, w):
    saidas = []
    for (uf, br), g in seg.groupby(["uf", "br"]):
        s = g.set_index("km_ini")[["n_acid", "mortos", "graves", "leves", "ups"]]
        s = s.reindex(np.arange(0, int(s.index.max()) + 1), fill_value=0)      # km sem acidente = 0
        r = s.rolling(w).sum().dropna()                                         # janela termina no km do índice
        r["km_ini"] = r.index - (w - 1); r["km_fim"] = r.index + 1
        r["uf"], r["br"] = uf, br
        saidas.append(r[r.n_acid > 0])
    return pd.concat(saidas, ignore_index=True)

jan = janelas_moveis(seg_km, JANELA_KM)

# 3) Seleção gulosa de trechos sem sobreposição, em ordem de gravidade (UPS)
def selecionar_sem_sobreposicao(jan, top, min_acid):
    jan = jan[jan.n_acid >= min_acid].sort_values(["ups", "n_acid"], ascending=False)
    esc = []
    for r in jan.itertuples():
        if all(not (r.uf == e.uf and r.br == e.br and r.km_ini < e.km_fim and e.km_ini < r.km_fim) for e in esc):
            esc.append(r)
            if len(esc) == top: break
    return pd.DataFrame(esc).drop(columns="Index")

hot = selecionar_sem_sobreposicao(jan, top=50, min_acid=MIN_ACID_HOTSPOT)
hot["trecho"] = hot.apply(lambda r: f"BR-{int(r.br):03d}/{r.uf} km {int(r.km_ini)}–{int(r.km_fim)}", axis=1)
print(f"Hotspots identificados: {len(hot)}")
display(hot.head(TOP_N)[["trecho", "n_acid", "mortos", "graves", "leves", "ups"]])

# %% [markdown]
# ### 4.2 Teste estatístico: o trecho é realmente anômalo?
# Um trecho pode parecer "campeão" só por estar em rodovia muito longa ou movimentada. Comparamos o observado com o **esperado sob distribuição uniforme** ao longo da mesma BR/UF (Poisson): se a rodovia tem 400 acidentes em 400 km, esperamos ≈ 1 por km, ou ≈ 5 em uma janela de 5 km. A razão **Observado/Esperado (O/E)** e o p-valor indicam o quão fora da curva o trecho está.
#
# > ⚠️ Como testamos muitos trechos, use o p-valor como **critério de triagem**, não como prova (problema de comparações múltiplas). A correção de Bonferroni simples está incluída.

# %%
def poisson_sf(k, lam):
    """P(X >= k) para X ~ Poisson(lam), calculado só com numpy/math (sem scipy)."""
    k = int(k)
    if k <= 0: return 1.0
    termos = [math.exp(-lam + i * math.log(lam) - math.lgamma(i + 1)) for i in range(k)]
    return max(0.0, 1.0 - sum(termos))

# Extensão observada de cada rodovia/UF e total de acidentes nela
ext = seg_km.groupby(["uf", "br"]).agg(km_max=("km_ini", "max"), acid_total=("n_acid", "sum")).reset_index()
ext["km_extensao"] = ext.km_max + 1
hot = hot.merge(ext[["uf", "br", "km_extensao", "acid_total"]], on=["uf", "br"], how="left")
hot["esperado"] = hot.acid_total * JANELA_KM / hot.km_extensao
hot["razao_O_E"] = hot.n_acid / hot.esperado
hot["p_valor"] = [poisson_sf(o, e) for o, e in zip(hot.n_acid, hot.esperado)]
n_testes = len(jan[jan.n_acid >= MIN_ACID_HOTSPOT])
hot["significativo_bonferroni"] = hot.p_valor < (0.05 / max(n_testes, 1))
hot["pct_acid_total"] = hot.n_acid / len(a) * 100

display(hot.head(TOP_N)[["trecho", "n_acid", "esperado", "razao_O_E", "p_valor", "significativo_bonferroni", "ups"]])
print(f"{hot.significativo_bonferroni.sum()} de {len(hot)} hotspots permanecem significativos após Bonferroni.")

# %% [markdown]
# ### 4.3 Gráfico: ranking dos hotspots (UPS e composição de vítimas)
# Mostramos os trechos com maior UPS. A barra empilhada separa **mortos, feridos graves e feridos leves**, para que o gestor veja a natureza do problema, e não apenas a pontuação.

# %%
top = hot.head(TOP_N).iloc[::-1]
fig, axes = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={"width_ratios": [1.2, 1]}, sharey=True)

ax = axes[0]
ax.barh(top.trecho, top.mortos, color=COR_ALERTA, label="Mortos")
ax.barh(top.trecho, top.graves, left=top.mortos, color=COR_DESTAQUE, label="Feridos graves")
ax.barh(top.trecho, top.leves, left=top.mortos + top.graves, color="#aab7b8", label="Feridos leves")
ax.set_xlabel("Vítimas"); ax.set_title("Composição das vítimas"); ax.legend(loc="lower right")

ax = axes[1]
cores = np.where(top.significativo_bonferroni, COR_ALERTA, COR_NEUTRA)
ax.barh(top.trecho, top.ups, color=cores)
for i, (u, n) in enumerate(zip(top.ups, top.n_acid)): ax.text(u, i, f"  {int(u)} UPS | {int(n)} acid.", va="center", fontsize=8)
ax.set_xlim(0, top.ups.max() * 1.35); ax.set_xlabel("UPS"); ax.set_title("Severidade (vermelho = significativo após Bonferroni)")
plt.suptitle(f"Top {TOP_N} hotspots (janela de {JANELA_KM} km)", fontweight="bold", y=1.0)
plt.tight_layout(); salvar(fig, "04_ranking_hotspots")

# %% [markdown]
# ---
# # Parte 5 — Fatores associados à gravidade (o que agravar ou proteger)
#
# **Variável-alvo:** `grave` = o acidente teve **pelo menos um óbito ou ferido grave**. É binária, estável e menos sensível a diferenças de registro entre leve/sem vítima.
#
# Seguimos uma progressão de rigor crescente:
#
# | Etapa | Técnica | Pergunta respondida |
# |---|---|---|
# | 5.1 | Taxa de gravidade com **IC de Wilson** | Qual a % de graves em cada categoria e a diferença é confiável? |
# | 5.2 | **Risco Relativo (RR)** | Quantas vezes o risco é maior que no restante? |
# | 5.3 | **Cramér V** | Quais variáveis têm maior associação geral com a gravidade? |
# | 5.4 | **Regressão logística** (numpy) | Qual o efeito de cada fator **mantendo os demais constantes**? |
# | 5.5 | Interações | Combinações que potencializam o risco |
#
# ### 5.1 Funções reutilizáveis
# `taxa_por_fator` calcula, para qualquer coluna categórica, o tamanho do grupo, a taxa de graves, o **intervalo de confiança de Wilson** (mais confiável que o intervalo "normal" para proporções pequenas) e o **risco relativo** contra todos os demais grupos.

# %%
def wilson(casos, n, z=1.96):
    """Intervalo de confiança de Wilson para proporções (vetorizado)."""
    casos, n = np.asarray(casos, float), np.asarray(n, float)
    p = casos / n
    den = 1 + z**2 / n
    centro = (p + z**2 / (2 * n)) / den
    margem = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    return centro - margem, centro + margem

def taxa_por_fator(df, col, alvo="grave", min_n=MIN_N_FATOR):
    """Taxa do alvo por categoria de `col`, com IC 95% e risco relativo vs. restante."""
    g = df.groupby(col, observed=True)[alvo].agg(n="size", casos="sum").reset_index()
    N, K = g.n.sum(), g.casos.sum()
    g["taxa"] = g.casos / g.n
    g["ic_inf"], g["ic_sup"] = wilson(g.casos, g.n)
    resto_n, resto_k = N - g.n, K - g.casos
    with np.errstate(divide="ignore", invalid="ignore"):
        g["taxa_resto"] = resto_k / resto_n
        g["risco_relativo"] = g.taxa / g.taxa_resto
    return g[g.n >= min_n].sort_values("taxa", ascending=False).reset_index(drop=True)

def grafico_taxas(ax, res, col, base, top=12, titulo=None):
    r = res.head(top).iloc[::-1]
    err = [np.clip(r.taxa - r.ic_inf, 0, None) * 100, np.clip(r.ic_sup - r.taxa, 0, None) * 100]
    ax.barh(r[col].astype(str), r.taxa * 100, xerr=err, capsize=3, color=np.where(r.ic_inf > base, COR_ALERTA, COR_NEUTRA))
    ax.axvline(base * 100, color="k", ls="--", lw=1)
    for i, (t, n) in enumerate(zip(r.taxa, r.n)): ax.text(0.3, i, f"n={n:,}", va="center", fontsize=7, color="white")
    ax.set_xlabel("% de acidentes graves"); ax.set_title(titulo or col)

BASE = a.grave.mean()
print(f"Taxa geral de acidentes graves: {BASE:.1%}  (linha tracejada nos gráficos; barras vermelhas = IC inteiro acima da média)")

# %% [markdown]
# ### 5.2 Fatores do ambiente, da via e do tempo
# Seis painéis comparando a taxa de gravidade por categoria. **Barras vermelhas** indicam categorias cujo intervalo de confiança está **inteiramente acima** da média geral, ou seja, risco elevado com segurança estatística. As barras de erro mostram a incerteza: categorias pequenas têm barras longas.

# %%
fatores_ambiente = ["fase_dia", "condicao_metereologica", "tipo_pista", "tracado_via", "uso_solo", "faixa_horaria"]
tabelas_fatores = {}
fig, axes = plt.subplots(2, 3, figsize=(17, 9))
for ax, f in zip(axes.ravel(), fatores_ambiente):
    res = taxa_por_fator(a.dropna(subset=[f]), f)
    tabelas_fatores[f] = res
    grafico_taxas(ax, res, f, BASE, titulo=f.replace("_", " ").title())
plt.suptitle("Taxa de acidentes graves por fator ambiental (IC 95%)", fontweight="bold", y=1.0)
plt.tight_layout(); salvar(fig, "10_fatores_ambientais")
display(pd.concat({f: t.head(3)[[f, "n", "taxa", "risco_relativo"]].rename(columns={f: "categoria"}) for f, t in tabelas_fatores.items()}))

# %% [markdown]
# ### 5.3 Tipo de acidente e causa
# Estes são os dois fatores com maior poder de explicar a gravidade e os que mais orientam **o tipo de intervenção** (engenharia, fiscalização ou educação). Mostramos a taxa de gravidade e o **risco relativo**.

# %%
res_tipo  = taxa_por_fator(a.dropna(subset=["tipo_acidente"]), "tipo_acidente")
res_causa = taxa_por_fator(a.dropna(subset=["causa_acidente"]), "causa_acidente")
tabelas_fatores["tipo_acidente"], tabelas_fatores["causa_acidente"] = res_tipo, res_causa

fig, axes = plt.subplots(1, 2, figsize=(17, 7))
grafico_taxas(axes[0], res_tipo, "tipo_acidente", BASE, top=14, titulo="Tipo de acidente")
grafico_taxas(axes[1], res_causa, "causa_acidente", BASE, top=14, titulo="Causa do acidente")
plt.tight_layout(); salvar(fig, "11_tipo_e_causa")

display(res_tipo.head(8)[["tipo_acidente", "n", "casos", "taxa", "risco_relativo"]])
display(res_causa.head(8)[["causa_acidente", "n", "casos", "taxa", "risco_relativo"]])

# %% [markdown]
# ### 5.4 Composição do acidente (quem se envolve)
# Presença de motocicleta, caminhão, bicicleta ou pedestre, e número de veículos. Usuários vulneráveis (sem proteção de carroceria) tendem a ter risco muito maior, o que justifica ações dedicadas a eles.

# %%
linhas = []
for flag, rotulo in [("tem_moto", "Motocicleta"), ("tem_caminhao", "Caminhão"), ("tem_bicicleta", "Bicicleta"), ("tem_pedestre", "Pedestre")]:
    for val, txt in [(True, "com"), (False, "sem")]:
        sub = a[a[flag] == val]
        lo, hi = wilson(sub.grave.sum(), len(sub))
        linhas.append({"fator": rotulo, "grupo": txt, "n": len(sub), "taxa": sub.grave.mean(), "lo": float(lo), "hi": float(hi)})
comp = pd.DataFrame(linhas)

fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
ax = axes[0]
x = np.arange(4); w = .38
for k, (txt, cor) in enumerate([("sem", COR_NEUTRA), ("com", COR_ALERTA)]):
    s = comp[comp.grupo == txt]
    ax.bar(x + (k - .5) * w, s.taxa * 100, w, color=cor, label=f"{txt} o fator",
           yerr=[(s.taxa - s.lo) * 100, (s.hi - s.taxa) * 100], capsize=3)
ax.set_xticks(x); ax.set_xticklabels(comp.fator.unique()); ax.set_ylabel("% de acidentes graves"); ax.legend()
ax.axhline(BASE * 100, color="k", ls="--", lw=1); ax.set_title("Presença de usuário/veículo específico")

res_nv = taxa_por_fator(a, "n_veiculos_cat", min_n=10).sort_values("n_veiculos_cat")
grafico_taxas(axes[1], res_nv, "n_veiculos_cat", BASE, titulo="Número de veículos envolvidos")
plt.tight_layout(); salvar(fig, "12_composicao_acidente")
display(comp)

# %% [markdown]
# ### 5.5 Quais variáveis mais se associam à gravidade? (Cramér V)
# O **V de Cramér** mede a força da associação entre duas variáveis categóricas (0 = nenhuma, 1 = perfeita). Usamos a versão com **correção de viés** (Bergsma), importante quando há muitas categorias. É um filtro rápido para decidir onde investir atenção: as variáveis no topo merecem análise detalhada.

# %%
def cramers_v(x, y):
    """V de Cramér com correção de viés (Bergsma, 2013), apenas numpy/pandas."""
    ct = pd.crosstab(x, y).values.astype(float)
    n = ct.sum()
    esperado = ct.sum(1, keepdims=True) @ ct.sum(0, keepdims=True) / n
    chi2 = ((ct - esperado) ** 2 / esperado).sum()
    r, k = ct.shape
    phi2 = max(0, chi2 / n - (k - 1) * (r - 1) / (n - 1))
    rc, kc = r - (r - 1) ** 2 / (n - 1), k - (k - 1) ** 2 / (n - 1)
    return np.sqrt(phi2 / max(min(kc - 1, rc - 1), 1e-9))

candidatas = ["tipo_acidente", "causa_acidente", "fase_dia", "condicao_metereologica", "tipo_pista", "tracado_via", "uso_solo",
              "sentido_via", "dia_semana", "faixa_horaria", "n_veiculos_cat", "tem_moto", "tem_caminhao", "tem_bicicleta",
              "tem_pedestre", "fim_de_semana", "uf"]
assoc = []
for c in candidatas:
    d = a[[c, "grave"]].dropna()
    if d[c].nunique() > 1: assoc.append((c, cramers_v(d[c].astype(str), d.grave)))
assoc = pd.DataFrame(assoc, columns=["variavel", "cramers_v"]).sort_values("cramers_v")

fig, ax = plt.subplots(figsize=(8, 6.5))
ax.barh(assoc.variavel, assoc.cramers_v, color=np.where(assoc.cramers_v >= assoc.cramers_v.quantile(.75), COR_ALERTA, COR_NEUTRA))
for i, v in enumerate(assoc.cramers_v): ax.text(v, i, f" {v:.3f}", va="center", fontsize=8)
ax.set_xlabel("V de Cramér (associação com acidente grave)"); ax.set_title("Ranking de associação com a gravidade")
salvar(fig, "13_cramers_v")

# %% [markdown]
# ### 5.6 Regressão logística multivariada (efeito ajustado)
# As taxas anteriores olham **um fator por vez** e podem confundir. Exemplo: "chuva" pode parecer perigosa só porque coincide com rodovias de pista simples. A regressão logística estima o efeito de cada fator **mantendo os demais constantes**, produzindo **Razões de Chance (Odds Ratio, OR)**: OR = 1,5 significa 50% mais chance de acidente grave que a categoria de referência.
#
# Implementamos a regressão com **IRLS** (mínimos quadrados reponderados iterativamente) em numpy puro, com uma leve **regularização L2** que dá estabilidade numérica. Para uso em produção, `statsmodels.Logit` fornece os mesmos coeficientes com testes e diagnósticos completos (a estrutura da tabela é a mesma).
#
# **Cuidados de modelagem aplicados:**
# * categorias raras (< 1%) agrupadas em "Outros";
# * categoria de referência = a **mais frequente** de cada variável;
# * separação **treino (70%) / teste (30%)** e avaliação por **AUC** (capacidade de ordenar acidentes graves acima dos não graves).

# %%
def agrupar_raras(s, min_pct=0.01, top=None):
    s = s.astype("object").fillna("Não informado")
    freq = s.value_counts(normalize=True)
    manter = set(freq[freq >= min_pct].index) if top is None else set(freq.head(top).index)
    return s.where(s.isin(manter), "Outros")

def montar_X(df, variaveis_cat, variaveis_bin):
    partes, refs = [], {}
    for v in variaveis_cat:
        s = agrupar_raras(df[v], top=10 if v == "causa_acidente" else None)
        ref = s.value_counts().idxmax(); refs[v] = ref
        d = pd.get_dummies(s, prefix=v, prefix_sep="=", dtype=float).drop(columns=f"{v}={ref}")
        partes.append(d)
    for v in variaveis_bin:
        partes.append(df[v].astype(float).rename(v).to_frame())
    X = pd.concat(partes, axis=1)
    return X, refs

def logit_irls(X, y, l2=1e-2, max_iter=100, tol=1e-8):
    """Regressão logística por IRLS. X já deve conter a coluna de intercepto na posição 0."""
    n, p = X.shape
    beta = np.zeros(p)
    pen = l2 * np.eye(p); pen[0, 0] = 0                       # não penaliza o intercepto
    for _ in range(max_iter):
        eta = np.clip(X @ beta, -30, 30)
        mu = 1 / (1 + np.exp(-eta))
        W = np.clip(mu * (1 - mu), 1e-9, None)
        z = eta + (y - mu) / W
        XtW = X.T * W
        H = XtW @ X + pen
        novo = np.linalg.solve(H, XtW @ z)
        if np.max(np.abs(novo - beta)) < tol: beta = novo; break
        beta = novo
    cov = np.linalg.inv(H)
    return beta, np.sqrt(np.diag(cov))

def auc_rank(y, score):
    """AUC pela estatística de Mann-Whitney (via ranks)."""
    r = pd.Series(score).rank().values
    n1, n0 = y.sum(), len(y) - y.sum()
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

cat_vars = ["fase_dia", "condicao_metereologica", "tipo_pista", "tracado_via", "uso_solo", "tipo_acidente", "causa_acidente", "n_veiculos_cat"]
bin_vars = ["tem_moto", "tem_caminhao", "tem_bicicleta", "fim_de_semana"]

dados = a.dropna(subset=["fase_dia", "tipo_pista", "tipo_acidente"]).reset_index(drop=True)
X_df, refs = montar_X(dados, cat_vars, bin_vars)
y = dados.grave.astype(float).values

rng = np.random.default_rng(42)
idx = rng.permutation(len(dados)); corte = int(.7 * len(idx))
tr, te = idx[:corte], idx[corte:]

Xtr = np.column_stack([np.ones(len(tr)), X_df.values[tr]])
Xte = np.column_stack([np.ones(len(te)), X_df.values[te]])
beta, se = logit_irls(Xtr, y[tr])

auc_tr = auc_rank(y[tr], Xtr @ beta); auc_te = auc_rank(y[te], Xte @ beta)
print(f"AUC treino: {auc_tr:.3f} | AUC teste: {auc_te:.3f}  (0,5 = acaso; ≥0,7 = discriminação aceitável)")
print("Categorias de referência:", refs)

odds = pd.DataFrame({"variavel": ["intercepto"] + list(X_df.columns), "coef": beta, "OR": np.exp(beta),
                     "OR_inf": np.exp(beta - 1.96 * se), "OR_sup": np.exp(beta + 1.96 * se)})
odds["significativo"] = (odds.OR_inf > 1) | (odds.OR_sup < 1)
odds = odds[odds.variavel != "intercepto"].sort_values("OR", ascending=False).reset_index(drop=True)
display(odds.head(10))

# %% [markdown]
# **Gráfico de floresta (forest plot):** cada ponto é uma razão de chance ajustada com seu IC 95%. A linha vertical em **1** significa "sem efeito". Pontos à direita aumentam a chance de acidente grave; à esquerda, protegem. Em vermelho/azul, os efeitos estatisticamente significativos.

# %%
t = pd.concat([odds.head(12), odds.tail(8)]).drop_duplicates("variavel").sort_values("OR")
fig, ax = plt.subplots(figsize=(10, 9))
cores = np.where(~t.significativo, "#aab7b8", np.where(t.OR > 1, COR_ALERTA, COR_NEUTRA))
ax.errorbar(t.OR, range(len(t)), xerr=[t.OR - t.OR_inf, t.OR_sup - t.OR], fmt="none", ecolor="gray", capsize=3, zorder=1)
ax.scatter(t.OR, range(len(t)), c=cores, s=60, zorder=2)
ax.axvline(1, color="k", ls="--"); ax.set_xscale("log")
ax.set_yticks(range(len(t))); ax.set_yticklabels(t.variavel.str.replace("_", " "), fontsize=8)
ax.set_xlabel("Razão de chance ajustada (escala log) para acidente grave")
ax.set_title(f"Fatores de risco e proteção (AUC teste = {auc_te:.2f})\nvermelho = aumenta risco | azul = reduz | cinza = não significativo")
ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"{v:g}"))
salvar(fig, "14_forest_plot_odds")

# %% [markdown]
# ### 5.7 Interações: combinações que potencializam o risco
# Efeitos isolados não contam toda a história. Cruzamos **tipo de acidente × fase do dia** e **tipo de pista × condição meteorológica**. Células escuras indicam combinações especialmente perigosas, úteis para definir *quando e onde* cada ação é mais necessária. Células com menos de 15 acidentes ficam em branco para evitar conclusões frágeis.

# %%
def tabela_cruzada(df, linha, coluna, min_n=15, top_linhas=8):
    n = df.pivot_table(index=linha, columns=coluna, values="id", aggfunc="count")
    taxa = df.pivot_table(index=linha, columns=coluna, values="grave", aggfunc="mean") * 100
    taxa = taxa.where(n >= min_n)
    linhas_top = df[linha].value_counts().head(top_linhas).index
    return taxa.loc[taxa.index.isin(linhas_top)].dropna(how="all").dropna(axis=1, how="all")

t1 = tabela_cruzada(a, "tipo_acidente", "fase_dia")
t2 = tabela_cruzada(a, "tipo_pista", "condicao_metereologica", top_linhas=5)

fig, axes = plt.subplots(1, 2, figsize=(17, 6))
heatmap(axes[0], t1, "% graves: tipo de acidente × fase do dia", cmap="Reds", rotulo_cb="% graves")
heatmap(axes[1], t2, "% graves: tipo de pista × condição meteorológica", cmap="Reds", rotulo_cb="% graves")
plt.tight_layout(); salvar(fig, "15_interacoes")

# %% [markdown]
# ---
# # Parte 6 — Perfis de risco (para quem agir)
#
# Aqui a unidade de análise volta a ser a **pessoa** (uma linha por envolvido). O alvo é `pessoa_grave` (óbito ou lesão grave) e, em alguns gráficos, só o óbito. Os perfis respondem: *quem* sofre consequências mais graves, em *qual veículo* e *em que contexto*, para direcionar campanhas, fiscalização e proteção de usuários vulneráveis.
#
# > ⚠️ **Interpretação:** as taxas abaixo são **entre os envolvidos em acidentes registrados**, não taxas populacionais. Dizem "dado que se envolveu em acidente, qual a chance de resultado grave", e não "qual o risco de dirigir". Para risco por exposição, seria necessário volume de tráfego (VMD) ou frota/habitantes.
#
# ### 6.1 Preparação: base de pessoas válidas

# %%
pv = p[p.gravidade_pessoa != "Ignorado"].copy()
pv["sexo_ok"] = pv.sexo.where(pv.sexo.isin(["Masculino", "Feminino"]))      # descarta "Não informado"/inválido
BASE_P = pv.pessoa_grave.mean()
print(f"Pessoas analisadas: {len(pv):,} | % graves (óbito ou lesão grave): {BASE_P:.1%} | % óbitos: {pv.pessoa_fatal.mean():.2%}")

# %% [markdown]
# ### 6.2 Perfil demográfico: faixa etária × sexo
# Dois gráficos complementares: (a) a **taxa** de gravidade por faixa etária e sexo (quem tem pior desfecho) e (b) a **pirâmide** de vítimas graves/fatais em números absolutos (onde está o maior volume de vítimas). Taxa alta com volume baixo e volume alto com taxa moderada pedem ações diferentes.

# %%
d = pv.dropna(subset=["faixa_etaria", "sexo_ok"])
tx = d.pivot_table(index="faixa_etaria", columns="sexo_ok", values="pessoa_grave", aggfunc="mean", observed=True) * 100
vol_v = d[d.pessoa_grave].pivot_table(index="faixa_etaria", columns="sexo_ok", values="id", aggfunc="count", observed=True).fillna(0)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
ax = axes[0]
for sexo, cor, mk in [("Masculino", COR_NEUTRA, "o"), ("Feminino", COR_ALERTA, "s")]:
    ax.plot(tx.index.astype(str), tx[sexo], marker=mk, color=cor, lw=2, label=sexo)
ax.axhline(BASE_P * 100, color="k", ls="--", lw=1); ax.set_ylabel("% de óbito ou lesão grave"); ax.set_xlabel("Faixa etária")
ax.set_title("Gravidade por faixa etária e sexo (entre os envolvidos)"); ax.legend()

ax = axes[1]; y_pos = np.arange(len(vol_v))
ax.barh(y_pos, -vol_v["Masculino"], color=COR_NEUTRA, label="Masculino"); ax.barh(y_pos, vol_v["Feminino"], color=COR_ALERTA, label="Feminino")
ax.set_yticks(y_pos); ax.set_yticklabels(vol_v.index.astype(str)); ax.axvline(0, color="k", lw=.8)
ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"{abs(int(v)):,}")); ax.set_xlabel("Vítimas graves e fatais (nº de pessoas)")
ax.set_title("Pirâmide de vítimas graves/fatais"); ax.legend()
plt.tight_layout(); salvar(fig, "16_perfil_idade_sexo")

# %% [markdown]
# ### 6.3 Perfil do usuário e do veículo
# Tipo de envolvido (condutor, passageiro, pedestre), grupo de veículo e **idade do veículo**. Veículos mais antigos costumam ter menos itens de segurança (airbag, ABS, estrutura de absorção), e esse efeito aparece como taxa crescente com a idade da frota.

# %%
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for ax, (col, tit) in zip(axes, [("tipo_envolvido", "Tipo de envolvido"), ("grupo_veiculo", "Grupo de veículo"), ("faixa_idade_veiculo", "Idade do veículo (anos)")]):
    res = taxa_por_fator(pv.dropna(subset=[col]), col, alvo="pessoa_grave", min_n=30)
    if col == "faixa_idade_veiculo": res = res.sort_values(col)
    grafico_taxas(ax, res, col, BASE_P, titulo=tit)
plt.suptitle("Taxa de óbito ou lesão grave por perfil do envolvido", fontweight="bold", y=1.0)
plt.tight_layout(); salvar(fig, "17_perfil_usuario_veiculo")

# Letalidade (só óbitos) por grupo de veículo: tabela de apoio
let = pv.groupby("grupo_veiculo").agg(pessoas=("id", "size"), obitos=("pessoa_fatal", "sum"), graves=("pessoa_grave", "sum"))
let["letalidade_%"] = let.obitos / let.pessoas * 100
display(let.sort_values("letalidade_%", ascending=False))

# %% [markdown]
# ### 6.4 Perfis combinados (os "tipos de pessoa/situação" mais críticos)
# Combinamos **três variáveis por vez** (veículo, faixa etária, sexo, tipo de envolvido, fase do dia, tipo de pista) e procuramos grupos com taxa significativamente acima da média. Para priorizar, usamos:
#
# * **Lift** = taxa do perfil ÷ taxa geral (quantas vezes pior que a média);
# * **Casos excedentes** = graves observados − graves esperados pela média. Esta métrica equilibra **risco** e **volume**: um perfil com lift 5 mas 20 pessoas rende pouco; um perfil com lift 1,6 e 5.000 pessoas gera muitos casos evitáveis.
#
# Só entram perfis com **n ≥ 50** e **limite inferior do IC acima da média**.

# %%
def perfis_combinados(df, variaveis, tamanho=3, min_n=50, alvo="pessoa_grave"):
    base = df[alvo].mean()
    saidas = []
    for combo in itertools.combinations(variaveis, tamanho):
        dd = df.dropna(subset=list(combo))
        g = dd.groupby(list(combo), observed=True)[alvo].agg(n="size", casos="sum").reset_index()
        g = g[g.n >= min_n]
        if g.empty: continue
        g["perfil"] = g[list(combo)].astype(str).agg("  |  ".join, axis=1)
        saidas.append(g[["perfil", "n", "casos"]])
    r = pd.concat(saidas, ignore_index=True)
    r["taxa"] = r.casos / r.n
    r["lift"] = r.taxa / base
    r["ic_inf"], _ = wilson(r.casos, r.n)
    r["casos_excedentes"] = r.casos - r.n * base
    return r[r.ic_inf > base].sort_values("casos_excedentes", ascending=False)

vars_perfil = ["grupo_veiculo", "faixa_etaria", "sexo_ok", "tipo_envolvido", "fase_dia", "tipo_pista"]
perfis = perfis_combinados(pv, vars_perfil, min_n=max(50, int(len(pv) * 0.0005)))
print(f"Perfis significativamente acima da média: {len(perfis):,}")

t = perfis.head(TOP_N).iloc[::-1]
fig, ax = plt.subplots(figsize=(14, 7.5))
sc = ax.barh(t.perfil, t.casos_excedentes, color=plt.cm.YlOrRd((t.lift - 1) / max(t.lift.max() - 1, 1e-9) * .8 + .2))
for i, r in enumerate(t.itertuples()): ax.text(r.casos_excedentes, i, f"  lift {r.lift:.1f}× | n={r.n:,} | {r.taxa:.0%}", va="center", fontsize=8)
ax.set_xlim(0, t.casos_excedentes.max() * 1.35); ax.tick_params(axis="y", labelsize=8)
ax.set_xlabel("Casos graves excedentes (observado − esperado pela média)")
ax.set_title("Perfis de maior prioridade: risco × volume (cor = lift)")
salvar(fig, "18_perfis_combinados")
display(perfis.head(10)[["perfil", "n", "casos", "taxa", "lift", "casos_excedentes"]])

# %% [markdown]
# ### 6.5 Perfil de cada hotspot (a "impressão digital" do trecho)
# Dois hotspots com o mesmo UPS podem exigir intervenções opostas: um é um problema de **ultrapassagem em pista simples à noite**; outro, de **moto em travessia urbana**. Para cada trecho calculamos causa e tipo dominantes e a participação de noite, chuva, moto, caminhão, pedestre, pista simples e curva.
#
# A **prioridade** combina gravidade (UPS) e persistência: `prioridade = UPS × (0,5 + 0,5 × recorrência)`. Trechos crônicos sobem no ranking porque são mais previsíveis e respondem melhor a obras permanentes.

# %%
def perfil_trecho(r):
    d = acidentes_do_trecho(r)
    n = len(d)
    moda = lambda s: (s.value_counts().index[0], s.value_counts(normalize=True).iloc[0]) if s.notna().any() else ("-", 0)
    causa, pc = moda(d.causa_acidente); tipo, pt = moda(d.tipo_acidente)
    return pd.Series({
        "n": n, "causa_dominante": causa, "pct_causa": pc, "tipo_dominante": tipo, "pct_tipo": pt,
        "pct_noite":   d.fase_dia.str.contains("noite", case=False, na=False).mean(),
        "pct_chuva":   d.condicao_metereologica.str.contains("chuva|garoa|neblina|nevoeiro", case=False, na=False).mean(),
        "pct_moto":    d.tem_moto.mean(), "pct_caminhao": d.tem_caminhao.mean(), "pct_pedestre": d.tem_pedestre.mean(),
        "pct_pista_simples": d.tipo_pista.str.contains("simples", case=False, na=False).mean(),
        "pct_curva":   d.tracado_via.str.contains("curva", case=False, na=False).mean(),
        "taxa_grave":  d.grave.mean(),
    })

hot_top = hot.head(30).copy().reset_index(drop=True)
perf = hot_top.apply(perfil_trecho, axis=1)
hot_top = pd.concat([hot_top, perf.drop(columns="n")], axis=1)
hot_top["prioridade"] = hot_top.ups * (0.5 + 0.5 * hot_top.recorrencia_mensal.fillna(0))
hot_top = hot_top.sort_values("prioridade", ascending=False).reset_index(drop=True)

cols_ind = ["pct_noite", "pct_chuva", "pct_moto", "pct_caminhao", "pct_pedestre", "pct_pista_simples", "pct_curva", "taxa_grave"]
mapa = hot_top.head(12).set_index("trecho")[cols_ind] * 100
mapa.columns = ["Noite", "Chuva/neblina", "Moto", "Caminhão", "Pedestre", "Pista simples", "Curva", "% graves"]

fig, ax = plt.subplots(figsize=(13, 6.5))
heatmap(ax, mapa, "Perfil dos 12 hotspots de maior prioridade (% dos acidentes do trecho)", cmap="Blues", fmt="{:.0f}", rotulo_cb="%")
ax.set_xticklabels(mapa.columns, rotation=30, ha="right")
plt.tight_layout(); salvar(fig, "19_perfil_hotspots")
display(hot_top.head(10)[["trecho", "n_acid", "ups", "recorrencia_mensal", "prioridade", "causa_dominante", "tipo_dominante"]])

# %% [markdown]
# ---
# # Parte 7 — Matriz de intervenções (do diagnóstico à ação)
#
# Convertemos o perfil de cada hotspot em **hipóteses de intervenção** por regras simples e transparentes, organizadas em três eixos clássicos de segurança viária:
#
# * 🛠️ **Engenharia** (geometria, sinalização, defensas, iluminação, drenagem);
# * 👮 **Fiscalização/Operação** (radar, blitz, escalas);
# * 📣 **Educação/Comunicação** (campanhas direcionadas ao perfil dominante).
#
# > As regras são **ponto de partida**, não receita. Cada trecho deve passar por **inspeção de segurança viária em campo** antes de qualquer obra. A transparência das regras permite que especialistas as revisem e ajustem; é um bom padrão para sistemas de apoio à decisão.

# %%
# (expressão regular sobre causa/tipo em minúsculas, eixo, ação)
REGRAS_CAUSA = [
    ("velocidade",              "Fiscalização", "Fiscalização eletrônica de velocidade e revisão do limite; redutores em travessias"),
    ("álcool|alcool|embriaguez","Fiscalização", "Operações Lei Seca em horários noturnos e de fim de semana"),
    ("dormindo|sono|fadiga",    "Engenharia",   "Sonorizadores (rumble strips), áreas de descanso e paradas para motoristas profissionais"),
    ("animal",                  "Engenharia",   "Cercamento, passagens de fauna e sinalização específica de animais"),
    ("ultrapassagem",           "Engenharia",   "Terceira faixa/faixa de ultrapassagem, linha dupla contínua e separador físico"),
    ("distância|distancia",     "Educação",     "Campanha de distância segura e sinalização horizontal de espaçamento (chevrons)"),
    ("atenção|atencao",         "Educação",     "Campanha sobre uso de celular/distração e sonorizadores no acostamento"),
    ("desobediência|desobediencia|normas", "Fiscalização", "Fiscalização de infrações e revisão de semaforização/sinalização de conflito"),
    ("defeito|mecânico|mecanico","Fiscalização","Operações de inspeção de veículos (freios, pneus) e transporte de carga"),
]
REGRAS_TIPO = [
    ("frontal",                 "Engenharia",   "Defensa/barreira central e tratamento de trechos de ultrapassagem"),
    ("pedestre",                "Engenharia",   "Passarela/travessia segura, iluminação e acostamento para pedestres"),
    ("saída|saida",             "Engenharia",   "Defensas, delineadores e tratamento de curvas e drenagem"),
    ("capotamento|tombamento",  "Engenharia",   "Revisão de curva (raio/superelevação), sinalização de velocidade e atrito do pavimento"),
    ("transversal|lateral",     "Engenharia",   "Revisão de interseções: rotatória, semáforo, retorno e acessos"),
    ("traseira|engavet",        "Engenharia",   "Sinalização antecipada de lentidão/congestionamento (painéis) e fiscalização de distância"),
]
def regras_contexto(r):
    saidas = []
    if r.pct_noite >= 0.40:       saidas.append(("Engenharia",   "Iluminação e sinalização refletiva; fiscalização noturna"))
    if r.pct_chuva >= 0.25:       saidas.append(("Engenharia",   "Drenagem, pavimento antiderrapante e sinalização para chuva/neblina"))
    if r.pct_moto >= 0.25:        saidas.append(("Educação",     "Ações dirigidas a motociclistas e defensas protetivas para motociclistas"))
    if r.pct_pedestre >= 0.10:    saidas.append(("Engenharia",   "Travessias, calçadas e redução de velocidade em área urbana"))
    if r.pct_curva >= 0.40:       saidas.append(("Engenharia",   "Revisão geométrica da curva e sinalização vertical/horizontal"))
    if r.pct_pista_simples >= 0.60 and r.pct_caminhao >= 0.25: saidas.append(("Engenharia", "Faixa adicional para veículos lentos / duplicação"))
    return saidas

def recomendar(r):
    acoes = []
    for padrao, eixo, txt in REGRAS_CAUSA:
        if pd.Series([str(r.causa_dominante).lower()]).str.contains(padrao, regex=True).iloc[0]: acoes.append((eixo, txt)); break
    for padrao, eixo, txt in REGRAS_TIPO:
        if pd.Series([str(r.tipo_dominante).lower()]).str.contains(padrao, regex=True).iloc[0]: acoes.append((eixo, txt)); break
    acoes += regras_contexto(r)
    return acoes[:5]

hot_top["acoes"] = hot_top.apply(recomendar, axis=1)
hot_top["recomendacao"] = hot_top.acoes.apply(lambda l: " • ".join(f"[{e}] {t}" for e, t in l))

matriz = hot_top.head(15)[["trecho", "ups", "mortos", "recorrencia_mensal", "causa_dominante", "tipo_dominante", "recomendacao"]]
pd.set_option("display.max_colwidth", 220)
display(matriz)

# %% [markdown]
# **Gráfico-síntese:** quantas vezes cada ação foi sugerida entre os 15 hotspots prioritários, colorida por eixo. Ajuda a ver, por exemplo, se o portfólio de intervenções é predominantemente de engenharia (investimento) ou de fiscalização (custeio).

# %%
linhas = [(e, t) for l in hot_top.head(15).acoes for e, t in l]
cont = pd.DataFrame(linhas, columns=["eixo", "acao"]).value_counts().reset_index(name="trechos").sort_values("trechos")
cor_eixo = {"Engenharia": COR_NEUTRA, "Fiscalização": COR_ALERTA, "Educação": COR_DESTAQUE}

fig, ax = plt.subplots(figsize=(13, 7))
ax.barh(cont.acao.str.slice(0, 85), cont.trechos, color=cont.eixo.map(cor_eixo))
ax.set_xlabel("Nº de hotspots prioritários em que a ação é sugerida"); ax.tick_params(axis="y", labelsize=8)
ax.set_title("Portfólio de intervenções sugeridas para os 15 hotspots prioritários")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=c, label=e) for e, c in cor_eixo.items()], loc="lower right")
salvar(fig, "20_portfolio_intervencoes")

# %% [markdown]
# ---
# # Parte 8 — Exportação, resumo executivo e limitações
#
# ### 8.1 Exportar resultados
# * **CSV** com `utf-8-sig` (abre com acentos corretos no Excel) para as tabelas finais;
# * **Parquet** (via DuckDB, nativo e sem dependências extras) para as camadas *silver*, reaproveitáveis em painéis e em atualizações futuras.

# %%
hot_export = hot_top.drop(columns=["acoes"])
hot_export.to_csv(PASTA / "tabelas" / "hotspots_priorizados.csv", index=False, encoding="utf-8-sig", sep=";", decimal=",")
odds.to_csv(PASTA / "tabelas" / "odds_ratio_gravidade.csv", index=False, encoding="utf-8-sig", sep=";", decimal=",")
perfis.head(200).to_csv(PASTA / "tabelas" / "perfis_de_risco.csv", index=False, encoding="utf-8-sig", sep=";", decimal=",")
mun.sort_values("ups", ascending=False).to_csv(PASTA / "tabelas" / "municipios.csv", index=False, encoding="utf-8-sig", sep=";", decimal=",")
del_rank.to_csv(PASTA / "tabelas" / "delegacias.csv", index=False, encoding="utf-8-sig", sep=";", decimal=",")
uop_rank.to_csv(PASTA / "tabelas" / "uops.csv", index=False, encoding="utf-8-sig", sep=";", decimal=",")
pd.concat(tabelas_fatores, names=["fator"]).reset_index(level=0).to_csv(PASTA / "tabelas" / "taxas_por_fator.csv", index=False, encoding="utf-8-sig", sep=";", decimal=",")

# Parquet com DuckDB (camadas silver)
con.register("acidentes_final", a)
con.execute(f"COPY pessoas_sql TO '{(PASTA / 'pessoas_silver.parquet').as_posix()}' (FORMAT PARQUET)")
con.execute(f"COPY acidentes_final TO '{(PASTA / 'acidentes_silver.parquet').as_posix()}' (FORMAT PARQUET)")
print("Arquivos gerados:")
for f in sorted(PASTA.rglob("*")):
    if f.is_file() and f.suffix in (".csv", ".parquet", ".html", ".txt"): print(" -", f.relative_to(PASTA))

# %% [markdown]
# ### 8.2 Resumo executivo automático
# Textos-síntese gerados a partir dos resultados, para colar em relatório ou apresentação. Como são calculados no código, **atualizam sozinhos** quando você rodar o notebook com um novo arquivo.

# %%
top_h = hot_top.iloc[0]
fator_top = odds[odds.significativo].head(3).variavel.tolist()
protetor  = odds[odds.significativo].tail(2).variavel.tolist()
perfil_top = perfis.iloc[0] if len(perfis) else None
letal_veic = let.sort_values("letalidade_%", ascending=False).index[0]

resumo = f"""
RESUMO EXECUTIVO — ACIDENTES {ANO_REF}
{'=' * 60}
PANORAMA
 • {len(a):,} acidentes, {a.mortos.sum():,} mortos e {a.feridos_graves.sum():,} feridos graves.
 • {a.grave.mean():.1%} dos acidentes foram graves (óbito ou lesão grave); {a.fatal.mean():.1%} tiveram óbito.

HOTSPOTS
 • 50% dos acidentes se concentram em apenas {p50_acid:.1f}% dos quilômetros com ocorrência.
 • Trecho prioritário nº 1: {top_h.trecho} — {int(top_h.n_acid)} acidentes, {int(top_h.mortos)} mortos, UPS {int(top_h.ups)}, recorrência {top_h.recorrencia_mensal:.0%} dos meses.
 • {hot.significativo_bonferroni.sum()} dos {len(hot)} principais trechos são estatisticamente anômalos (Poisson + Bonferroni).

FATORES DE GRAVIDADE (efeito ajustado, regressão logística, AUC teste = {auc_te:.2f})
 • Maiores aumentos de risco: {'; '.join(fator_top)}.
 • Maiores efeitos protetores: {'; '.join(protetor)}.

PERFIS DE RISCO
 • Grupo com maior letalidade entre os envolvidos: {letal_veic} ({let.loc[letal_veic, 'letalidade_%']:.1f}% de óbitos).
 • Perfil de maior prioridade (risco × volume): {perfil_top.perfil if perfil_top is not None else 'n/d'}.
"""
print(resumo)
_ = (PASTA / "resumo_executivo.txt").write_text(resumo, encoding="utf-8")

# %% [markdown]
# ### 8.3 Limitações e boas práticas (leia antes de apresentar resultados)
#
# 1. **Sub-registro e viés de registro:** acidentes sem vítima são menos notificados; a taxa de gravidade pode ficar superestimada em locais onde só ocorrências sérias chegam à polícia.
# 2. **Falta de exposição:** sem volume de tráfego (VMD) ou frota, contamos acidentes, não **risco**. Rodovias movimentadas aparecem mais por terem mais veículos. O próximo passo é calcular acidentes por milhão de veículos-km.
# 3. **Associação não é causalidade:** os OR mostram o que *anda junto* com gravidade. Ex.: "noite" pode refletir álcool, cansaço e baixa visibilidade ao mesmo tempo.
# 4. **Regressão ao longo da média:** trechos escolhidos por terem tido muitos acidentes em um ano tendem a melhorar sozinhos no seguinte. Para **avaliar uma intervenção**, use método antes/depois com **Empirical Bayes** e grupo de comparação.
# 5. **Precisão do km e das coordenadas:** erros de 1 a 2 km no registro do local deslocam acidentes entre trechos. A janela de 5 km ameniza, mas confira os hotspots no mapa.
# 6. **Janela de um único ano:** use 3 a 5 anos para estabilizar o ranking de hotspots (a base de 2026 isolada é um bom início).
# 7. **Comparações múltiplas:** com muitos trechos e categorias, algumas diferenças "significativas" aparecem por acaso. Priorize resultados consistentes entre análises (hotspot + recorrência + OR ajustado).
# 8. **Privacidade (LGPD):** os dados são anonimizados, mas ao cruzar idade, sexo, local e horário pode haver reidentificação em locais pequenos. Divulgue agregados.
#
# ### 8.4 Próximos passos para evoluir o projeto
# | Passo | Como |
# |---|---|
# | Unir **mais anos** (2020–2026) | `read_csv('dados/*.csv')` no DuckDB (aceita curinga) e coluna `ano` |
# | Hotspots geoestatísticos | **KDE/Getis-Ord** com `geopandas` + `pysal`, com *snap* das coordenadas ao eixo da rodovia |
# | Modelo preditivo de risco | `scikit-learn`/`LightGBM` + **SHAP** para explicabilidade; validação temporal |
# | Previsão de volume | Séries temporais mensais por trecho (`statsmodels`, `Prophet`) |
# | Painel interativo | **Streamlit** ou **Power BI** lendo os Parquet de `saida/` |
# | Automatização | Script `.py` agendado (cron/Prefect) que gera os relatórios mensalmente |
# | Avaliação de impacto | Antes/depois com Empirical Bayes em trechos tratados × grupo de controle |

# %% [markdown]
# ### 8.5 Encerrar a conexão
# Fecha o DuckDB de forma limpa (libera memória e arquivos).

# %%
import gc
for _nome in ["pessoas_df", "acidentes_df", "acidentes_final"]:
    try: con.unregister(_nome)
    except Exception: pass
con.close()
del con; gc.collect()
print("Análise concluída. Resultados em:", PASTA)