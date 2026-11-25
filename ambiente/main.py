
#-Parte 1 — Configuração, carga e qualidade dos dados
#--1.1 Bibliotecas e parâmetros
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
ARQUIVO   = "ambiente/acidentes2026_causas_tipos.csv"   # caminho do seu arquivo (relativo à pasta do script ou absoluto)
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
#(PASTA / "tabelas").mkdir(parents=True, exist_ok=True)

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

#--1.2 Carga do arquivo com DuckDB
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

#--1.3 Diagnóstico de qualidade (valores ausentes)
# %%
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

#-Parte 2 — Limpeza e modelagem dos dados
#--2.1 Tabela `pessoas` (silver)
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

#--2.2 Variáveis derivadas por pessoa
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

#--2.3 Tabela `acidentes` (uma linha por ocorrência)
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

#-Parte 3 — Visão geral (EDA) Painel geral
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

#-Parte 4 — Hotspots (onde agir)
#--4.1 Segmentação por km e janela móvel (SQL + pandas)
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

#--4.2 Teste estatístico: o trecho é realmente anômalo?
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

#--4.3 Gráfico: ranking dos hotspots (UPS e composição de vítimas)
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

#-Parte 5 — Fatores associados à gravidade (o que agravar ou proteger)
#--5.1 Funções reutilizáveis
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

#--5.2 Fatores do ambiente, da via e do tempo
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

#--5.3 Tipo de acidente e causa
res_tipo  = taxa_por_fator(a.dropna(subset=["tipo_acidente"]), "tipo_acidente")
res_causa = taxa_por_fator(a.dropna(subset=["causa_acidente"]), "causa_acidente")
tabelas_fatores["tipo_acidente"], tabelas_fatores["causa_acidente"] = res_tipo, res_causa

fig, axes = plt.subplots(1, 2, figsize=(17, 7))
grafico_taxas(axes[0], res_tipo, "tipo_acidente", BASE, top=14, titulo="Tipo de acidente")
grafico_taxas(axes[1], res_causa, "causa_acidente", BASE, top=14, titulo="Causa do acidente")
plt.tight_layout(); salvar(fig, "11_tipo_e_causa")

display(res_tipo.head(8)[["tipo_acidente", "n", "casos", "taxa", "risco_relativo"]])
display(res_causa.head(8)[["causa_acidente", "n", "casos", "taxa", "risco_relativo"]])

#--5.3 Quais variáveis mais se associam à gravidade? (Cramér V)
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

#-Parte 6 — Perfis de risco (para quem agir)
#--6.1 Preparação: base de pessoas válidas
# %%
pv = p[p.gravidade_pessoa != "Ignorado"].copy()
pv["sexo_ok"] = pv.sexo.where(pv.sexo.isin(["Masculino", "Feminino"]))      # descarta "Não informado"/inválido
BASE_P = pv.pessoa_grave.mean()
print(f"Pessoas analisadas: {len(pv):,} | % graves (óbito ou lesão grave): {BASE_P:.1%} | % óbitos: {pv.pessoa_fatal.mean():.2%}")

#--6.2 Perfil demográfico: faixa etária × sexo
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

#--6.3 Perfil do usuário e do veículo
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

#-Parte 7 — Exportação, resumo executivo e limitações
#--7.2 Resumo executivo automático
# %%
def melhores_categorias(res, col, base, k=3):
    """Até k categorias com taxa significativamente acima da média, já formatadas em texto."""
    sig = res[res.ic_inf > base].head(k)
    return [f"{r[col]} ({r.taxa:.0%} graves; risco {r.risco_relativo:.1f}× o restante; n={int(r.n):,})"
            for _, r in sig.iterrows()]


def linhas_ou_aviso(lista):
    """Transforma uma lista de textos em linhas com marcador, ou avisa que nada foi significativo."""
    if not lista:
        return [" • nenhuma categoria claramente acima da média."]
    return [f" • {x}" for x in lista]


# --- panorama
n_acid, n_mortos, n_graves_vit = len(a), int(a.mortos.sum()), int(a.feridos_graves.sum())

# --- concentração: % dos km (com acidente) que reúne metade dos acidentes
acum = seg_km.n_acid.sort_values(ascending=False).cumsum() / seg_km.n_acid.sum()
pct_km_50 = (int((acum < 0.5).sum()) + 1) / len(seg_km) * 100

# --- hotspots
top_h = hot.sort_values("ups", ascending=False).iloc[0]
n_sig = int(hot.significativo_bonferroni.sum())

# --- fatores: ambiente/via/tempo (um por fator), tipo e causa
fatores_amb = [f for f in tabelas_fatores if f not in ("tipo_acidente", "causa_acidente")]
linhas_ambiente = []
for f in fatores_amb:
    melhor = melhores_categorias(tabelas_fatores[f], f, BASE, k=1)
    if melhor:
        linhas_ambiente.append(f"{f.replace('_', ' ')}: {melhor[0]}")

# --- associação geral (V de Cramér)
top_assoc = assoc.sort_values("cramers_v", ascending=False).head(3)
txt_assoc = "; ".join(f"{r.variavel} (V={r.cramers_v:.2f})" for r in top_assoc.itertuples())

# --- perfis de risco
let_ok = let[let.pessoas >= MIN_N_FATOR].sort_values("letalidade_%", ascending=False)
faixas = (pv.dropna(subset=["faixa_etaria"]).groupby("faixa_etaria", observed=True)
            .pessoa_grave.agg(n="size", taxa="mean"))
faixas = faixas[faixas.n >= MIN_N_FATOR]
sexos = pv.dropna(subset=["sexo_ok"]).groupby("sexo_ok").pessoa_grave.mean()

linhas_perfil = []
if len(let_ok):
    g = let_ok.index[0]
    linhas_perfil.append(f"Grupo de veículo mais letal entre os envolvidos: {g} ({let_ok.loc[g, 'letalidade_%']:.1f}% de óbitos).")
if len(faixas):
    f_top = faixas.taxa.idxmax()
    linhas_perfil.append(f"Faixa etária com maior taxa de óbito/lesão grave: {f_top} anos ({faixas.loc[f_top, 'taxa']:.1%}).")
if {"Masculino", "Feminino"} <= set(sexos.index):
    linhas_perfil.append(f"Taxa de óbito/lesão grave: homens {sexos['Masculino']:.1%} | mulheres {sexos['Feminino']:.1%}.")

linhas = [
    f"RESUMO EXECUTIVO — ACIDENTES {ANO_REF}",
    "=" * 62,
    "PANORAMA",
    f" • {n_acid:,} acidentes, {n_mortos:,} mortos e {n_graves_vit:,} feridos graves.",
    f" • {a.grave.mean():.1%} dos acidentes foram graves (óbito ou lesão grave); {a.fatal.mean():.1%} tiveram óbito.",
    "",
    "HOTSPOTS",
    f" • 50% dos acidentes estão em apenas {pct_km_50:.1f}% dos quilômetros com ocorrência.",
    f" • Trecho nº 1 em severidade: {top_h.trecho} — {int(top_h.n_acid)} acidentes, {int(top_h.mortos)} mortos, UPS {int(top_h.ups)}.",
    f" • {n_sig} dos {len(hot)} principais trechos são estatisticamente anômalos (Poisson + Bonferroni).",
    "",
    "FATORES ASSOCIADOS À GRAVIDADE (taxas com IC 95%)",
    " Tipo de acidente:",
    *linhas_ou_aviso(melhores_categorias(res_tipo, "tipo_acidente", BASE)),
    " Causa:",
    *linhas_ou_aviso(melhores_categorias(res_causa, "causa_acidente", BASE)),
    " Ambiente, via e horário:",
    *linhas_ou_aviso(linhas_ambiente),
    f" • Maior associação geral com a gravidade (V de Cramér): {txt_assoc}.",
    "",
    "PERFIS DE RISCO",
    *linhas_ou_aviso(linhas_perfil),
    "",
    "Observação: taxas calculadas entre os envolvidos em acidentes registrados (sem dados de exposição/tráfego).",
]
resumo = "\n".join(linhas)
print(resumo)
(PASTA / "resumo_executivo.txt").write_text(resumo, encoding="utf-8")
print("\nResumo salvo em:", PASTA / "resumo_executivo.txt")
