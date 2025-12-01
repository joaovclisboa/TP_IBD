"""Junta o CSV consolidado ao shapefile de municípios de Minas Gerais.

Tudo é estático: caminhos fixos definidos no topo do arquivo. O script:
1. Localiza o shapefile de MG; se não existir, tenta abrir o shapefile completo do Brasil e filtra SIGLA=="MG".
2. Lê o CSV `dados_consolidados_violencia_feminicidio.csv`.
3. Tenta casar os dados pelo código IBGE, ignorando o último dígito presente no shapefile (usa os 6 primeiros dígitos).
4. Cai para o casamento pelos nomes normalizados se não houver código.
5. Exporta um novo shapefile `municipios_MG_data.shp` com as colunas originais + atributos do CSV.
"""

from pathlib import Path
import sys
import unicodedata

import geopandas as gpd
import pandas as pd


# ---------------------------------------------------------------------------
# Configurações estáticas — edite os caminhos conforme necessário
ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "dados_consolidados_violencia_feminicidio.csv"
MG_SHP_PATH = ROOT / "TP_IBD" / "geoprocessamento" / "shapefile" / "municipios_MG.shp"
BR_SHP_PATH = ROOT / "TP_IBD" / "geoprocessamento" / "shapefile" / "municipios.shp"
OUT_PATH = Path(__file__).resolve().parent / "municipios_MG_data.shp"
ESTADO_SIGLA = "MG"
CSV_SEP = ";"
CSV_ENCODING = "utf-8"
YEAR_COLUMN = "ano"
METRIC_COLUMNS = ["total_violencia_domestica", "feminicidios", "taxa_feminicidio"]
METRIC_MEAN = {"taxa_feminicidio"}
EXPORT_BY_YEAR = True  # gera shapefiles separados por ano com sufixo _2022, _2023...
# ---------------------------------------------------------------------------


def normalizar_nome(valor: str) -> str:
    """Remove acentos e deixa Title Case para facilitar comparação por nome."""
    if pd.isna(valor):
        return ""
    valor = str(valor)
    valor = unicodedata.normalize("NFKD", valor).encode("ASCII", "ignore").decode("utf-8")
    return " ".join(p.title() for p in valor.split())


def detect_code_column(colunas):
    """Procura uma coluna que pareça conter código IBGE."""
    candidatos = [
        "municipio_cod",
        "codigo_ibge",
        "cod_ibge",
        "cod_mun",
        "cd_mun",
        "cd_municip",
        "cd_ibge",
        "codigo",
        "ibge",
        "cod",
    ]
    nomes = [c.lower() for c in colunas]
    for alvo in candidatos:
        for idx, nome in enumerate(nomes):
            if alvo in nome:
                return colunas[idx]
    return None


def detect_name_column(colunas):
    """Procura uma coluna de nome de município."""
    candidatos = [
        "municipio_nome",
        "nome",
        "municipio",
        "nm_mun",
        "nm_municip",
        "name",
    ]
    nomes = [c.lower() for c in colunas]
    for alvo in candidatos:
        for idx, nome in enumerate(nomes):
            if alvo in nome:
                return colunas[idx]
    return None


def carregar_shapefile() -> gpd.GeoDataFrame:
    """Lê o shapefile de MG (ou o shapefile completo filtrando SIGLA)."""
    if MG_SHP_PATH.exists():
        gdf = gpd.read_file(MG_SHP_PATH)
        print(f"Lido shapefile de MG: {len(gdf)} feições")
        return gdf

    if not BR_SHP_PATH.exists():
        print("Nenhum shapefile encontrado. Coloque `municipios_MG.shp` ou `municipios.shp` na pasta `geoprocessamento/shapefile`." )
        sys.exit(1)

    gdf_brasil = gpd.read_file(BR_SHP_PATH)
    if "SIGLA" not in gdf_brasil.columns:
        print("Shapefile completo encontrado, mas sem coluna 'SIGLA'. Informe um shapefile apenas de MG.")
        sys.exit(1)

    gdf_mg = gdf_brasil[gdf_brasil["SIGLA"].astype(str).str.upper() == ESTADO_SIGLA]
    print(
        f"Lido shapefile do Brasil: {len(gdf_brasil)} feições; após filtro {ESTADO_SIGLA}: {len(gdf_mg)}"
    )
    return gdf_mg


def pivotar_dados_por_ano(df: pd.DataFrame):
    """Gera uma tabela com uma linha por município e colunas por ano para evitar duplicação no shapefile."""

    if YEAR_COLUMN not in df.columns or "municipio_cod" not in df.columns:
        anos = sorted(df[YEAR_COLUMN].dropna().unique()) if YEAR_COLUMN in df.columns else None
        print("CSV sem coluna de ano ou código — mantendo formato original para o merge principal.")
        return df.copy(), anos

    metrics = [col for col in METRIC_COLUMNS if col in df.columns]
    if not metrics:
        anos = sorted(df[YEAR_COLUMN].dropna().unique())
        print("CSV não possui métricas esperadas para pivotar por ano; mantendo formato original.")
        return df.copy(), anos

    index_cols = ["municipio_cod"]
    if "municipio_nome" in df.columns:
        index_cols.append("municipio_nome")

    agg_dict = {metric: ("mean" if metric in METRIC_MEAN else "sum") for metric in metrics}
    agrupado = df.groupby(index_cols + [YEAR_COLUMN], as_index=False).agg(agg_dict)

    pivot = (
        agrupado.pivot_table(
            index=index_cols,
            columns=YEAR_COLUMN,
            values=metrics,
            aggfunc="first",
        )
        .sort_index(axis=1, level=1)
    )

    pivot.columns = [f"{metric}_{int(year)}" for metric, year in pivot.columns]
    pivot = pivot.reset_index()
    anos_disponiveis = sorted(df[YEAR_COLUMN].dropna().unique())
    print(
        f"Pivot criado com {len(anos_disponiveis)} ano(s). Colunas geradas: "
        f"{[col for col in pivot.columns if col not in index_cols]}"
    )
    return pivot, anos_disponiveis


def agrupar_por_municipio(df: pd.DataFrame, csv_code_col: str) -> pd.DataFrame:
    """Agrega valores por município para um único ano (usado nos shapefiles filtrados)."""

    if csv_code_col not in df.columns:
        return df.copy()

    agg_dict = {}
    for col in df.columns:
        if col in {csv_code_col, YEAR_COLUMN}:
            continue
        if col == "municipio_nome":
            agg_dict[col] = "first"
        elif pd.api.types.is_numeric_dtype(df[col]):
            agg_dict[col] = "mean" if col in METRIC_MEAN else "sum"
    if not agg_dict:
        return df.drop_duplicates(subset=[csv_code_col])

    return df.groupby(csv_code_col, as_index=False).agg(agg_dict)


def preparar_codigos(gdf: gpd.GeoDataFrame, df: pd.DataFrame, shp_code_col: str, csv_code_col: str):
    """Cria colunas auxiliares com apenas dígitos e os primeiros 6 dígitos (ignora último dígito do shapefile)."""
    gdf = gdf.copy()
    df = df.copy()

    gdf["__shp_code_digits"] = (
        gdf[shp_code_col]
        .astype(str)
        .str.replace(r"\D", "", regex=True)
        .fillna("")
        .str.zfill(7)
    )
    gdf["__shp_code6"] = gdf["__shp_code_digits"].str[:6]

    df["__csv_code_digits"] = (
        df[csv_code_col]
        .astype(str)
        .str.replace(r"\D", "", regex=True)
        .fillna("")
        .str.zfill(6)
    )
    df["__csv_code6"] = df["__csv_code_digits"].str[:6]

    uniq_shp = gdf["__shp_code6"].nunique()
    uniq_csv = df["__csv_code6"].nunique()
    print(f"Codigos únicos (shp6 / csv6): {uniq_shp} / {uniq_csv}")

    merged = gdf.merge(df, left_on="__shp_code6", right_on="__csv_code6", how="left")
    matches = merged["total_violencia_domestica"].notna().sum() if "total_violencia_domestica" in merged.columns else 0
    print(f"Casamentos por código (6 dígitos): {matches}")

    for col in ["__shp_code_digits", "__shp_code6", "__csv_code_digits", "__csv_code6"]:
        if col in merged.columns:
            merged.drop(columns=col, inplace=True)

    return merged


def merge_por_nome(gdf: gpd.GeoDataFrame, df: pd.DataFrame, shp_name_col: str):
    gdf_aux = gdf.copy()
    df_aux = df.copy()
    gdf_aux["__nome_norm"] = gdf_aux[shp_name_col].apply(normalizar_nome)
    df_aux["__nome_norm"] = df_aux["municipio_nome"].apply(normalizar_nome)
    merged = gdf_aux.merge(df_aux, on="__nome_norm", how="left")
    if "__nome_norm" in merged.columns:
        merged.drop(columns="__nome_norm", inplace=True)
    return merged


def main():
    if not CSV_PATH.exists():
        print(f"CSV consolidado não encontrado em {CSV_PATH}. Gere o arquivo antes de rodar.")
        sys.exit(1)

    gdf = carregar_shapefile()
    df_raw = pd.read_csv(CSV_PATH, sep=CSV_SEP, encoding=CSV_ENCODING)
    print(f"Lido CSV: {len(df_raw)} linhas")

    df, anos_disponiveis = pivotar_dados_por_ano(df_raw)

    shp_code_col = detect_code_column(gdf.columns)
    csv_code_col = "municipio_cod" if "municipio_cod" in df.columns else detect_code_column(df.columns)

    merged = None
    if shp_code_col and csv_code_col:
        print(
            f"Fazendo merge por código (primeiros 6 dígitos do shapefile): shapefile[{shp_code_col}] <-> csv[{csv_code_col}]"
        )
        merged = preparar_codigos(gdf, df, shp_code_col, csv_code_col)

    if merged is None:
        shp_name_col = detect_name_column(gdf.columns)
        if shp_name_col is None or "municipio_nome" not in df.columns:
            print("Não foi possível detectar coluna de código nem de nome para reunir os dados.")
            sys.exit(1)
        print(
            f"Colunas de código indisponíveis. Tentando merge por nome: shapefile[{shp_name_col}] <-> csv[municipio_nome]"
        )
        merged = merge_por_nome(gdf, df, shp_name_col)

    total = len(merged)
    with_data = (
        merged["total_violencia_domestica"].notna().sum()
        if "total_violencia_domestica" in merged.columns
        else 0
    )
    print(f"Feições no shapefile: {total} | com dados do CSV: {with_data}")

    merged.to_file(OUT_PATH)
    print(f"Shapefile de saída gravado em: {OUT_PATH}")
    print(
        "Aviso: o formato Shapefile limita os nomes dos campos a 10 caracteres. "
        "Considere exportar para GeoPackage (.gpkg) se precisar manter os nomes completos."
    )

    if (
        EXPORT_BY_YEAR
        and anos_disponiveis
        and shp_code_col
        and csv_code_col
        and YEAR_COLUMN in df_raw.columns
    ):
        print("Gerando shapefiles separados por ano...")
        for ano in anos_disponiveis:
            df_ano = df_raw[df_raw[YEAR_COLUMN] == ano]
            if df_ano.empty:
                continue
            df_ano_agg = agrupar_por_municipio(df_ano, csv_code_col)
            merged_year = preparar_codigos(gdf, df_ano_agg, shp_code_col, csv_code_col)
            out_year_path = OUT_PATH.with_name(f"{OUT_PATH.stem}_{int(ano)}{OUT_PATH.suffix}")
            merged_year.to_file(out_year_path)
            print(f"  - Shapefile {ano} gravado em: {out_year_path}")


if __name__ == "__main__":
    main()
