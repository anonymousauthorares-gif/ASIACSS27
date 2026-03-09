import os
import glob
import math
import numpy as np
import pandas as pd
import openpyxl
import matplotlib.pyplot as plt
import joblib 
from scipy.linalg import sqrtm 
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from sklearn.preprocessing import StandardScaler

# --- FUNÇÕES AUXILIARES ---
def is_cell_red(cell):
    try:
        if cell.fill and cell.fill.start_color:
            color_hex = str(cell.fill.start_color.rgb).upper()
            if color_hex in ['FFFF0000', 'RED']: return True
    except: pass
    return False

def clean_and_convert(value):
    if value is None: return None
    val_str = str(value).strip()
    if val_str == "" or val_str.startswith('#'): return None
    val_str = val_str.replace('\xa0', '').replace(',', '.')
    try: return float(val_str)
    except ValueError: return None

def extrair_dados_arquivo(caminho_arquivo, cols_indices):
    try:
        wb = openpyxl.load_workbook(caminho_arquivo, data_only=True)
        ws = wb.active 
        rows_data = []
        for row in ws.iter_rows(min_row=2, values_only=False): 
            row_vals = []
            discard_row = False
            if not row: continue
            for idx in cols_indices:
                if idx >= len(row): discard_row = True; break
                cell = row[idx] 
                if is_cell_red(cell): discard_row = True; break
                val = clean_and_convert(cell.value)
                if val is None: discard_row = True; break
                row_vals.append(val)
            if not discard_row and len(row_vals) == 5:
                rows_data.append(row_vals)
        return rows_data
    except Exception as e:
        print(f"Erro ao ler {os.path.basename(caminho_arquivo)}: {e}")
        return []

def ler_dados_pasta(caminho_pasta, label, indices_fixos):
    print(f"--- Lendo pasta: {label} ---")
    arquivos = glob.glob(os.path.join(caminho_pasta, "*.xlsx"))
    if not arquivos: return pd.DataFrame()
    dados_consolidados = []
    for arquivo in arquivos:
        dados = extrair_dados_arquivo(arquivo, indices_fixos)
        dados_consolidados.extend(dados)
    df = pd.DataFrame(dados_consolidados, columns=['Time', 'Speed', 'Acceleration', 'Heading', 'RateOfTurn'])
    print(f"-> {label}: {len(df)} linhas válidas recuperadas.")
    return df

def ler_arquivo_teste_autodetect(caminho_arquivo):
    dados = extrair_dados_arquivo(caminho_arquivo, [23, 24, 25, 26, 27])
    if not dados:
        dados = extrair_dados_arquivo(caminho_arquivo, [0, 1, 2, 3, 4])
    if not dados: return pd.DataFrame()
    return pd.DataFrame(dados, columns=['Time', 'Speed', 'Acceleration', 'Heading', 'RateOfTurn'])

# --- CÁLCULO DE MÉTRICAS ---

def calcular_score_benford(df):
    """Retorna a divergência da Lei de Benford (quanto menor, mais 'natural')."""
    if df.empty: return 999.0
    valores = df[['Speed', 'Acceleration', 'Heading', 'RateOfTurn']].values.flatten()
    primeiros_digitos = []
    for v in valores:
        if v == 0 or np.isnan(v): continue
        s = str(abs(v)).replace('.', '').replace(',', '').lstrip('0')
        if s: primeiros_digitos.append(int(s[0]))
    if not primeiros_digitos: return 999.0 
    total = len(primeiros_digitos)
    contagem = pd.Series(primeiros_digitos).value_counts().sort_index()
    freq_obs = np.array([contagem.get(d, 0)/total for d in range(1, 10)])
    freq_teorica = np.log10(1 + 1/np.arange(1, 10))
    return np.linalg.norm(freq_obs - freq_teorica)

def calcular_todas_metricas_distancia(mu1, cov1, mu2, cov2):
    """Calcula distâncias entre duas distribuições (1 e 2)."""
    res = {}
    diff = mu1 - mu2
    
    # 1. Euclidiana
    res['Euclidiana'] = np.linalg.norm(diff)
    
    # 2. Mahalanobis
    cov_mean = (cov1 + cov2) / 2
    try: inv_cov = np.linalg.inv(cov_mean)
    except: inv_cov = np.linalg.pinv(cov_mean)
    res['Mahalanobis'] = np.sqrt(diff.T @ inv_cov @ diff)
    
    # 3. Bhattacharyya
    term1 = (1/8) * diff.T @ inv_cov @ diff
    d1, d2, dm = np.linalg.det(cov1), np.linalg.det(cov2), np.linalg.det(cov_mean)
    if d1 <= 1e-12 or d2 <= 1e-12 or dm <= 1e-12: term2 = 0
    else: term2 = 0.5 * np.log(dm / np.sqrt(d1 * d2))
    res['Bhattacharyya'] = term1 + term2

    # 4. Wasserstein
    cov_sqrt_prod = sqrtm(cov1 @ cov2)
    if np.iscomplexobj(cov_sqrt_prod): cov_sqrt_prod = cov_sqrt_prod.real
    w_sq = res['Euclidiana']**2 + np.trace(cov1 + cov2 - 2*cov_sqrt_prod)
    res['Wasserstein'] = np.sqrt(max(0, w_sq))
    
    # 5. Frobenius (Norma da diferença das covariâncias)
    res['Frobenius'] = np.linalg.norm(cov1 - cov2, 'fro')
    
    return res

# --- TREINAMENTO GP (MANTIDO) ---

def treinar_gp(df, nome):
    if df is None or df.empty: return None
    X = df[['Time']].values
    y = df[['Speed', 'Acceleration', 'Heading', 'RateOfTurn']].values
    if len(X) > 1000:
        idx = np.random.choice(len(X), 1000, replace=False); idx.sort()
        X, y = X[idx], y[idx]
    sX = StandardScaler().fit(X)
    sY = StandardScaler().fit(y)
    Xs, Ys = sX.transform(X), sY.transform(y)
    kernel = C(1.0, (1e-10, 1e5)) * RBF(1.0, (1e-10, 1e5)) + WhiteKernel(0.1, (1e-10, 1e5))
    try:
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0)
        gp.fit(Xs, Ys)
        joblib.dump({'modelo': gp, 'sx': sX, 'sy': sY}, f"modelo_gp_{nome}.pkl")
        return gp
    except: return None

# --- MAIN ---

def main():
    print("\n=== SISTEMA DE CLASSIFICAÇÃO HIERÁRQUICO COMPLETO ===\n")
    
    # CAMINHOS (Substitua pelos seus caminhos reais)
    p_real = r"C:\Users\bm-lo\Desktop\GP_project\White"
    p_falso = r"C:\Users\bm-lo\Desktop\GP_project\Black"
    p_teste = r"C:\Users\bm-lo\Desktop\GP_project\Grey"

    # 1. CARREGAMENTO DAS BASES
    df_real = ler_dados_pasta(p_real, "BASE_REAL", [23, 24, 25, 26, 27])
    df_falso = ler_dados_pasta(p_falso, "BASE_FALSO", [23, 24, 25, 26, 27]) 

    if df_real.empty or df_falso.empty:
        print("ERRO: Bases de treino vazias."); return

    # Estatísticas Bases
    mu_real, cov_real = np.mean(df_real.values, axis=0), np.cov(df_real.values, rowvar=False)
    mu_falso, cov_falso = np.mean(df_falso.values, axis=0), np.cov(df_falso.values, rowvar=False)
    
    # Benford Bases (para referência)
    benford_base_real = calcular_score_benford(df_real)
    benford_base_falso = calcular_score_benford(df_falso)

    # 2. CSV DE TREINO (Diferença entre Black e White)
    # Calcula a distância entre a média/cov da base Real e da base Falsa
    metricas_treino = calcular_todas_metricas_distancia(mu_real, cov_real, mu_falso, cov_falso)
    
    csv_treino_data = {
        'Tipo': 'Diferenca_Base_Real_vs_Base_Falsa',
        'Benford_Score_Real': benford_base_real,
        'Benford_Score_Falso': benford_base_falso,
        'Dist_Bhattacharyya': metricas_treino['Bhattacharyya'],
        'Dist_Mahalanobis': metricas_treino['Mahalanobis'],
        'Dist_Frobenius': metricas_treino['Frobenius'],
        'Dist_Euclidiana': metricas_treino['Euclidiana'],
        'Dist_Wasserstein': metricas_treino['Wasserstein']
    }
    pd.DataFrame([csv_treino_data]).to_csv("metricas_bases_treino.csv", sep=';', decimal=',', index=False)
    print("\n[OK] Métricas de treino salvas em 'metricas_bases_treino.csv'")

    # 3. TREINAR GP (Opcional, mantido conforme original)
    print("\n--- Treinando Modelos GP ---")
    treinar_gp(df_real, "REAL")
    treinar_gp(df_falso, "FALSO")

    # 4. CLASSIFICAÇÃO DOS TESTES
    print(f"\n--- Classificando Testes em: {p_teste} ---")
    arquivos_teste = glob.glob(os.path.join(p_teste, "*.xlsx"))
    
    if not arquivos_teste: print("Nenhum arquivo de teste."); return

    resultados = []

    for arq in arquivos_teste:
        nome = os.path.basename(arq)
        df_t = ler_arquivo_teste_autodetect(arq)
        
        if df_t.empty:
            resultados.append({'Arquivo': nome, 'Classificacao': 'ERRO_LEITURA'}); continue
            
        mu_t, cov_t = np.mean(df_t.values, axis=0), np.cov(df_t.values, rowvar=False)
        
        # Calcula métricas contra REAL e contra FALSO
        mts_vs_real = calcular_todas_metricas_distancia(mu_t, cov_t, mu_real, cov_real)
        mts_vs_falso = calcular_todas_metricas_distancia(mu_t, cov_t, mu_falso, cov_falso)
        
        # Calcula Benford do Teste
        benford_teste = calcular_score_benford(df_t)

        # --- LÓGICA DE VOTAÇÃO (Quem acusa?) ---
        # Uma métrica acusa "FALSO" se a distância para a base Falsa for menor que para a base Real
        # Ou, no caso do Benford, se o score for mais próximo do score da base Falsa

        acusadores = []

        # 1. Bhattacharyya (O JUIZ)
        bhatt_voto = "REAL"
        if mts_vs_falso['Bhattacharyya'] < mts_vs_real['Bhattacharyya']:
            bhatt_voto = "FALSO"
            # O Bhattacharyya não entra na lista de "outros" acusadores, ele é tratado separadamente

        # 2. Benford
        dist_benford_real = abs(benford_teste - benford_base_real)
        dist_benford_falso = abs(benford_teste - benford_base_falso)
        benford_voto = "REAL"
        if dist_benford_falso < dist_benford_real:
            benford_voto = "FALSO"
            acusadores.append("Benford")

        # 3. Frobenius
        frob_voto = "REAL"
        if mts_vs_falso['Frobenius'] < mts_vs_real['Frobenius']:
            frob_voto = "FALSO"
            acusadores.append("Frobenius")

        # 4. Mahalanobis
        mahal_voto = "REAL"
        if mts_vs_falso['Mahalanobis'] < mts_vs_real['Mahalanobis']:
            mahal_voto = "FALSO"
            acusadores.append("Mahalanobis")

        # --- SENTENÇA FINAL ---
        classificacao = ""
        
        if bhatt_voto == "FALSO":
            # Bhattacharyya diz que é culpado
            if len(acusadores) > 0:
                classificacao = "TOTALMENTE CULPADO"
            else:
                classificacao = "CULPADO" # Apenas Bhattacharyya acusou
        else:
            # Bhattacharyya diz que é inocente (REAL)
            if len(acusadores) > 0:
                classificacao = "SUSPEITO" # Bhatt diz ok, mas outros apontaram erro
            else:
                classificacao = "VERDADEIRO" # Ninguém acusou

        # String formatada dos acusadores para o CSV
        str_acusadores = ", ".join(acusadores) if acusadores else "Nenhum"
        if bhatt_voto == "FALSO":
            str_acusadores = "Bhattacharyya" + ((" + " + str_acusadores) if str_acusadores != "Nenhum" else "")

        # Montagem da linha de dados completa
        linha = {
            'Arquivo': nome,
            'Classificacao_Final': classificacao,
            'Lista_Acusadores': str_acusadores,
            
            # BENFORD
            'Benford_Score_Teste': benford_teste,
            'Benford_Voto': benford_voto,
            
            # BHATTACHARYYA (Métrica Principal)
            'Bhatt_vs_Real': mts_vs_real['Bhattacharyya'],
            'Bhatt_vs_Falso': mts_vs_falso['Bhattacharyya'],
            'Bhatt_Voto': bhatt_voto,
            
            # MAHALANOBIS
            'Mahal_vs_Real': mts_vs_real['Mahalanobis'],
            'Mahal_vs_Falso': mts_vs_falso['Mahalanobis'],
            'Mahal_Voto': mahal_voto,
            
            # FROBENIUS
            'Frob_vs_Real': mts_vs_real['Frobenius'],
            'Frob_vs_Falso': mts_vs_falso['Frobenius'],
            'Frob_Voto': frob_voto,
            
            # OUTRAS MÉTRICAS (Requisitadas no CSV)
            'Wasserstein_vs_Real': mts_vs_real['Wasserstein'],
            'Wasserstein_vs_Falso': mts_vs_falso['Wasserstein'],
            
            'Euclidiana_vs_Real': mts_vs_real['Euclidiana'],
            'Euclidiana_vs_Falso': mts_vs_falso['Euclidiana']
        }
        resultados.append(linha)
        
        # Log no console
        tag = ""
        if classificacao == "TOTALMENTE CULPADO": tag = " [!!!]"
        elif classificacao == "CULPADO": tag = " [!]"
        elif classificacao == "SUSPEITO": tag = " [?]"
        
        print(f" -> {nome}: {classificacao}{tag} | Acusadores: {str_acusadores}")

    if resultados:
        df_res = pd.DataFrame(resultados)
        # Reordenando colunas para facilitar leitura
        cols = ['Arquivo', 'Classificacao_Final', 'Lista_Acusadores', 
                'Benford_Score_Teste', 'Benford_Voto',
                'Bhatt_vs_Real', 'Bhatt_vs_Falso', 'Bhatt_Voto',
                'Mahal_vs_Real', 'Mahal_vs_Falso', 'Mahal_Voto',
                'Frob_vs_Real', 'Frob_vs_Falso', 'Frob_Voto',
                'Wasserstein_vs_Real', 'Wasserstein_vs_Falso',
                'Euclidiana_vs_Real', 'Euclidiana_vs_Falso']
        
        df_res = df_res[cols]
        
        caminho_csv = os.path.join(os.path.dirname(p_teste), "relatorio_analise_metricas.csv")
        df_res.to_csv(caminho_csv, index=False, sep=';', decimal=',')
        print(f"\n[SUCESSO] Relatório completo salvo em: {caminho_csv}")

if __name__ == "__main__":
    main()


'''

Lei de Benford: Distribuição de probabilidade fenomenológica onde o primeiro dígito d de dados naturais ocorre com frequência log_10(1 + 1/d). Violações indicam dados artificiais ou manipulados.

Distância de Mahalanobis: Mede a distância entre um ponto e uma distribuição (ou entre duas médias), normalizando pela matriz de covariância inversa. É invariante à escala e considera a correlação entre as variáveis.

Distância de Bhattacharyya: Quantifica a separabilidade entre duas distribuições de probabilidade medindo a sobreposição (overlap) de suas densidades. Quanto maior o valor, menor a interseção entre as classes.

Distância de Wasserstein: Baseada em transporte ótimo, calcula o "custo mínimo" de trabalho para transformar uma distribuição na outra. É mais robusta que outras métricas quando as distribuições não se sobrepõem.

Distância Euclidiana: A métrica geométrica padrão (Norma L2). Calcula a distância linear entre dois vetores de média, assumindo que o espaço é isotrópico (ignora a dispersão e correlação dos dados).

Norma de Frobenius: É a "Distância Euclidiana para Matrizes". Calcula a raiz quadrada da soma dos quadrados das diferenças entre os elementos de duas matrizes (usada aqui para comparar a "forma" das covariâncias).

'''