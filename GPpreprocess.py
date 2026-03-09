import os
import glob
import math
import numpy as np
import pandas as pd
import openpyxl
import joblib 
from scipy.linalg import sqrtm 
from scipy.interpolate import PchipInterpolator
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from sklearn.preprocessing import StandardScaler

# =============================================================================
# 1. FUNÇÕES DE LEITURA E LIMPEZA
# =============================================================================

def clean_float(value):
    """Converte valores do Excel para float de forma robusta."""
    if value is None: return np.nan
    s = str(value).strip().replace('\xa0', '')
    if not s or s.lower() in ['null', '#', 'nan']: return np.nan
    s = s.replace(',', '.')
    try:
        return float(s)
    except:
        return np.nan

def extrair_dados_brutos(caminho_arquivo):
    """
    Lê o Excel buscando colunas de dados físicos.
    """
    indices_padrao = [23, 24, 25, 26, 27]
    indices_alternativos = [0, 1, 2, 3, 4]
    
    try:
        wb = openpyxl.load_workbook(caminho_arquivo, data_only=True)
        ws = wb.active 
        
        def ler_indices(inds):
            data = []
            for row in ws.iter_rows(min_row=2, values_only=True):
                if not row or len(row) <= inds[0]: continue
                vals = [clean_float(row[i]) if i < len(row) else np.nan for i in inds]
                if not np.all(np.isnan(vals)):
                    data.append(vals)
            return data

        raw_data = ler_indices(indices_padrao)
        if not raw_data:
            raw_data = ler_indices(indices_alternativos)
            
        return pd.DataFrame(raw_data, columns=['Time', 'Speed', 'Acceleration', 'Heading', 'RateOfTurn'])
    except Exception as e:
        print(f"Erro leitura {os.path.basename(caminho_arquivo)}: {e}")
        return pd.DataFrame()

# =============================================================================
# 2. PRÉ-PROCESSAMENTO: RECONSTRUÇÃO PCHIP
# =============================================================================

def reconstruir_sinal_pchip(df, n_points_out=None):
    if df.empty or len(df) < 5: return df
    
    # Filtro de Absurdos Físicos
    df.loc[(df['Speed'] < 0) | (df['Speed'] > 150), 'Speed'] = np.nan
    df.loc[np.abs(df['Heading']) > 360, 'Heading'] = np.nan
    
    if n_points_out is None:
        n_points_out = len(df)
    
    x_original = np.arange(len(df))
    x_new = np.linspace(0, len(df)-1, n_points_out)
    
    df_reconstruido = pd.DataFrame(index=range(n_points_out))
    cols = ['Speed', 'Acceleration', 'Heading', 'RateOfTurn']
    
    for col in cols:
        series = df[col].values
        mask = ~np.isnan(series)
        
        if mask.sum() > 4:
            try:
                interpolator = PchipInterpolator(x_original[mask], series[mask])
                y_smooth = interpolator(x_new)
                df_reconstruido[col] = y_smooth
            except:
                df_reconstruido[col] = pd.Series(series).interpolate(limit_direction='both').values
        else:
            df_reconstruido[col] = 0.0

    df_reconstruido['Time'] = x_new 
    return df_reconstruido

def carregar_pasta(caminho, label):
    print(f"--- Carregando: {label} ---")
    arquivos = glob.glob(os.path.join(caminho, "*.xlsx"))
    dfs = []
    
    for arq in arquivos:
        df_raw = extrair_dados_brutos(arq)
        if df_raw.empty: continue
        
        df_clean = reconstruir_sinal_pchip(df_raw)
        if not df_clean.empty:
            dfs.append(df_clean)
            
    if not dfs: return pd.DataFrame()
    final_df = pd.concat(dfs, ignore_index=True)
    print(f"-> {label}: {len(final_df)} linhas processadas.")
    return final_df

# =============================================================================
# 3. CÁLCULOS ESTATÍSTICOS E MÉTRICAS
# =============================================================================

def calcular_score_benford(df):
    if df.empty: return 999.0
    # Benford usa dados BRUTOS (não normalizados)
    valores = df.values.flatten()
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

def calcular_todas_metricas(mu1, cov1, mu2, cov2):
    res = {}
    diff = mu1 - mu2
    
    # Regularização
    cov_mean = (cov1 + cov2) / 2
    try: 
        inv_cov = np.linalg.inv(cov_mean)
    except: 
        inv_cov = np.linalg.pinv(cov_mean)
        
    # 1. Distância Euclidiana
    res['Euclidiana'] = np.linalg.norm(diff)
    
    # 2. Distância de Mahalanobis
    res['Mahalanobis'] = np.sqrt(max(0, diff.T @ inv_cov @ diff))
    
    # 3. Norma de Frobenius
    res['Frobenius'] = np.linalg.norm(cov1 - cov2, 'fro')

    # 4. Distância de Bhattacharyya
    term1 = 0.125 * diff.T @ inv_cov @ diff
    d1, d2, dm = np.linalg.det(cov1), np.linalg.det(cov2), np.linalg.det(cov_mean)
    if d1 <= 1e-20 or d2 <= 1e-20 or dm <= 1e-20: 
        term2 = 0
    else: 
        term2 = 0.5 * np.log(dm / np.sqrt(d1 * d2))
    res['Bhattacharyya'] = term1 + term2

    # 5. Distância de Wasserstein
    prod = cov1 @ cov2
    sqrt_prod = sqrtm(prod)
    if np.iscomplexobj(sqrt_prod): sqrt_prod = sqrt_prod.real
    w_sq = res['Euclidiana']**2 + np.trace(cov1 + cov2 - 2*sqrt_prod)
    res['Wasserstein'] = np.sqrt(max(0, w_sq))
    
    return res

# =============================================================================
# 4. TREINAMENTO GP
# =============================================================================

def treinar_gp(X_scaled, y_scaled, nome):
    if len(X_scaled) > 1000:
        idx = np.random.choice(len(X_scaled), 1000, replace=False); idx.sort()
        X_s, y_s = X_scaled[idx], y_scaled[idx]
    else:
        X_s, y_s = X_scaled, y_scaled

    kernel = C(1.0) * RBF(1.0) + WhiteKernel(0.1)
    try:
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0)
        gp.fit(X_s, y_s) 
        joblib.dump(gp, f"modelo_gp_{nome}.pkl")
        return gp
    except: return None

# =============================================================================
# 5. MAIN
# =============================================================================

def main():
    print("\n=== SISTEMA COMPLETO: TODAS AS MÉTRICAS + BASELINE ===\n")
    
    # --- CONFIGURAÇÃO DE CAMINHOS ---
    p_real = r"C:\Users\bm-lo\Desktop\GP_project\White"
    p_falso = r"C:\Users\bm-lo\Desktop\GP_project\Black"
    p_teste = r"C:\Users\bm-lo\Desktop\GP_project\Grey"

    # 1. CARREGAMENTO
    df_real = carregar_pasta(p_real, "BASE_REAL")
    df_falso = carregar_pasta(p_falso, "BASE_FALSO")

    if df_real.empty or df_falso.empty:
        print("ERRO: Bases de treino vazias."); return

    # 2. NORMALIZAÇÃO (Feature Scaling)
    cols_fisicas = ['Speed', 'Acceleration', 'Heading', 'RateOfTurn']
    print("\n--- Aplicando StandardScaler ---")
    scaler = StandardScaler()
    scaler.fit(df_real[cols_fisicas])
    
    X_real = scaler.transform(df_real[cols_fisicas])
    X_falso = scaler.transform(df_falso[cols_fisicas])
    
    # 3. ESTATÍSTICAS BASE (GAUSSIANAS)
    mu_real, cov_real = np.mean(X_real, axis=0), np.cov(X_real, rowvar=False)
    mu_falso, cov_falso = np.mean(X_falso, axis=0), np.cov(X_falso, rowvar=False)
    
    # =========================================================================
    # SALVAMENTO DAS MÉTRICAS DE TREINO (DIFERENCIAÇÃO ENTRE AS BASES)
    # =========================================================================
    print("\n--- Calculando Diferenciação entre Bases (Baseline) ---")
    
    # A. Distâncias Estatísticas (Real vs Falso)
    metrics_baseline = calcular_todas_metricas(mu_real, cov_real, mu_falso, cov_falso)
    
    # B. Benford Score das Bases (Calculado nos dados brutos!)
    benford_score_real = calcular_score_benford(df_real[cols_fisicas])
    benford_score_falso = calcular_score_benford(df_falso[cols_fisicas])
    
    # C. Monta o Dicionário de Treino
    treino_dict = {
        'Descricao': 'Diferenca_entre_White_e_Black',
        'Benford_Score_Base_Real': benford_score_real,
        'Benford_Score_Base_Falsa': benford_score_falso
    }
    # Adiciona as distâncias calculadas
    for k, v in metrics_baseline.items():
        treino_dict[f'Distancia_{k}_Real_vs_Falso'] = v

    # D. Salva CSV de Treino
    df_baseline = pd.DataFrame([treino_dict])
    path_baseline = os.path.join(os.path.dirname(p_teste), "metricas_distancia_entre_bases_treino.csv")
    df_baseline.to_csv(path_baseline, index=False, sep=';', decimal=',')
    print(f"[BASELINE] Métricas salvas em: {path_baseline}")
    
    # =========================================================================

    # 4. TREINAMENTO GP
    t_real = df_real[['Time']].values; t_falso = df_falso[['Time']].values
    scaler_t = StandardScaler().fit(t_real)
    print("\n--- Treinando GPs ---")
    treinar_gp(scaler_t.transform(t_real), X_real, "REAL")
    treinar_gp(scaler_t.transform(t_falso), X_falso, "FALSO")

    # 5. CLASSIFICAÇÃO DOS TESTES
    print(f"\n--- Classificando Arquivos em: {p_teste} ---")
    arquivos_teste = glob.glob(os.path.join(p_teste, "*.xlsx"))
    resultados = []

    for arq in arquivos_teste:
        nome = os.path.basename(arq)
        
        # A. Extração
        df_t = extrair_dados_brutos(arq)
        df_t = reconstruir_sinal_pchip(df_t)
        
        if df_t.empty or len(df_t) < 5:
            print(f" -> {nome}: IGNORADO (Dados insuficientes)"); continue
            
        # B. Normalização
        try:
            X_t = scaler.transform(df_t[cols_fisicas])
        except Exception as e:
            print(f" -> {nome}: Erro de escala ({e})"); continue

        # C. Estatísticas do Teste
        mu_t, cov_t = np.mean(X_t, axis=0), np.cov(X_t, rowvar=False)
        score_benford = calcular_score_benford(df_t[cols_fisicas]) 
        
        # D. Cálculo de TODAS as distâncias
        mts_real = calcular_todas_metricas(mu_t, cov_t, mu_real, cov_real)
        mts_falso = calcular_todas_metricas(mu_t, cov_t, mu_falso, cov_falso)
        
        # =====================================================================
        # LÓGICA DE DECISÃO HIERÁRQUICA
        # =====================================================================
        
        acusadores = []

        # 1. Bhattacharyya
        if mts_falso['Bhattacharyya'] < mts_real['Bhattacharyya']:
            acusadores.append("Bhattacharyya")

        # 2. Mahalanobis
        if mts_falso['Mahalanobis'] < mts_real['Mahalanobis']:
            acusadores.append("Mahalanobis")

        # 3. Frobenius
        if mts_falso['Frobenius'] < mts_real['Frobenius']:
            acusadores.append("Frobenius")

        # 4. Benford (> 0.12)
        if score_benford > 0.12:
            acusadores.append(f"Benford({score_benford:.2f})")

        # Sentença
        classificacao = ""
        bhatt_acusou = "Bhattacharyya" in acusadores
        outros_acusadores = [x for x in acusadores if x != "Bhattacharyya"]
        
        if bhatt_acusou:
            if len(outros_acusadores) > 0:
                classificacao = "TOTALMENTE CULPADO"
            else:
                classificacao = "CULPADO"
        else:
            if len(acusadores) > 0:
                classificacao = "SUSPEITO"
            else:
                classificacao = "VERDADEIRO"

        str_acusadores = " | ".join(acusadores) if acusadores else "Nenhum"

        # =====================================================================
        # MONTAGEM DO RESULTADO FINAL
        # =====================================================================
        
        res_dict = {
            'Arquivo': nome,
            'Classificacao': classificacao,
            'Quem_Acusou': str_acusadores,
            'Benford_Score': round(score_benford, 4)
        }
        
        # Salva as 5 métricas de distância para Real e Falso
        for metrica in ['Bhattacharyya', 'Mahalanobis', 'Frobenius', 'Wasserstein', 'Euclidiana']:
            res_dict[f'{metrica}_Dist_Real'] = round(mts_real[metrica], 4)
            res_dict[f'{metrica}_Dist_Falso'] = round(mts_falso[metrica], 4)
            
        resultados.append(res_dict)
        
        cor = ""
        if "TOTALMENTE" in classificacao: cor = " [!!!]"
        elif "CULPADO" in classificacao: cor = " [!]"
        elif "SUSPEITO" in classificacao: cor = " [?]"
        print(f" -> {nome}: {classificacao} ({str_acusadores}){cor}")

    if resultados:
        # Organização das Colunas
        cols_order = [
            'Arquivo', 'Classificacao', 'Quem_Acusou', 'Benford_Score',
            'Bhattacharyya_Dist_Real', 'Bhattacharyya_Dist_Falso',
            'Mahalanobis_Dist_Real', 'Mahalanobis_Dist_Falso',
            'Frobenius_Dist_Real', 'Frobenius_Dist_Falso',
            'Wasserstein_Dist_Real', 'Wasserstein_Dist_Falso',
            'Euclidiana_Dist_Real', 'Euclidiana_Dist_Falso'
        ]
        
        path_csv = os.path.join(os.path.dirname(p_teste), "relatorio_final_completo.csv")
        df_final = pd.DataFrame(resultados)
        df_final = df_final.reindex(columns=cols_order)
        
        df_final.to_csv(path_csv, index=False, sep=';', decimal=',')
        print(f"\n[SUCESSO] Relatório Final salvo em: {path_csv}")

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
