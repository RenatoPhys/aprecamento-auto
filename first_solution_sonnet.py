import pandas as pd
import numpy as np
import awswrangler as wr
import statsmodels.formula.api as smf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# FUNÇÃO PRINCIPAL - MODELO DE ELASTICIDADE
# ============================================================================

def criar_modelo_elasticidade(df, prod_var, taxa_var, cat_vars, 
                               faixas_taxa=5, exibir_graficos=True):
    """
    Cria modelo de elasticidade com variáveis categóricas personalizadas
    
    Parâmetros:
    -----------
    df : DataFrame
        Dataframe com dados de contratos
    prod_var : str
        Nome da variável de produção/volume
    taxa_var : str
        Nome da variável de taxa de juros
    cat_vars : list
        Lista de variáveis categóricas para segmentação
    faixas_taxa : int
        Número de faixas para discretizar a taxa (padrão: 5)
    exibir_graficos : bool
        Se True, exibe visualizações (padrão: True)
    
    Retorna:
    --------
    dict : Dicionário com modelo, dados agregados, elasticidades e cenários
    """
    
    print("\n" + "="*60)
    print("CONFIGURAÇÃO DO MODELO")
    print("="*60)
    print(f"Variável de Produção: {prod_var}")
    print(f"Variável de Taxa: {taxa_var}")
    print(f"Variáveis Categóricas: {cat_vars}")
    print(f"Faixas de Taxa: {faixas_taxa}")
    print("="*60)
    
    # Validar variáveis
    missing_vars = [var for var in cat_vars if var not in df.columns]
    if missing_vars:
        raise ValueError(f"Variáveis não encontradas: {missing_vars}")
    
    # Criar faixas de taxa
    df['range_taxa'] = pd.qcut(df[taxa_var], q=faixas_taxa, 
                               duplicates='drop')
    
    # Agregação
    aux = df[[taxa_var, prod_var] + cat_vars + ['range_taxa', 'anomes']].groupby(
        cat_vars + ['range_taxa', 'anomes'], 
        observed=True
    ).agg(
        qtd_obs=(prod_var, 'count'),
        taxa_media=(taxa_var, 'mean'),
        producao_media=(prod_var, 'mean'),
        producao_total=(prod_var, 'sum')
    ).reset_index()
    
    print(f"\nDataset agregado: {len(aux):,} linhas")
    print(f"Grupos únicos: {aux.groupby(cat_vars, observed=True).ngroups}")
    
    # Construir fórmula
    y = 'producao_total'
    string_model = f'{y} ~ {taxa_var}'
    
    for var in cat_vars:
        string_model = string_model + f' + {taxa_var}:C({var})'
    
    print(f"\nFórmula: {string_model}\n")
    
    # Ajustar tipos
    for variavel in cat_vars:
        aux[variavel] = aux[variavel].astype('category')
    
    # Fit do modelo
    model = smf.ols(formula=string_model, data=aux).fit()
    print(model.summary())
    
    # Performance
    aux['y_pred'] = model.predict(aux)
    aux['y'] = aux['producao_total']
    aux['erro'] = aux['y_pred'] - aux['y']
    aux['erro_relativo'] = aux['erro'] / aux['y']
    
    mae = mean_absolute_error(aux['y'], aux['y_pred'])
    rmse = np.sqrt(mean_squared_error(aux['y'], aux['y_pred']))
    r2 = r2_score(aux['y'], aux['y_pred'])
    mape = np.mean(np.abs(aux['erro_relativo'])) * 100
    
    print("\n" + "="*60)
    print("MÉTRICAS DE PERFORMANCE")
    print("="*60)
    print(f"R² Score: {r2:.4f}")
    print(f"MAE: R$ {mae:,.2f}")
    print(f"RMSE: R$ {rmse:,.2f}")
    print(f"MAPE: {mape:.2f}%")
    
    # Elasticidades
    df_elasticidades = calcular_elasticidade(model, aux, taxa_var, cat_vars)
    
    # Cenários
    variacoes = [-20, -15, -10, -5, 0, 5, 10, 15, 20]
    df_cenarios = simular_cenarios(aux, model, taxa_var, variacoes)
    
    # Oportunidades
    df_oportunidades = identificar_oportunidades(df_elasticidades)
    
    if exibir_graficos:
        plotar_resultados(aux, r2, df_cenarios)
    
    return {
        'model': model,
        'dados_agregados': aux,
        'elasticidades': df_elasticidades,
        'cenarios': df_cenarios,
        'oportunidades': df_oportunidades,
        'metricas': {'r2': r2, 'mae': mae, 'rmse': rmse, 'mape': mape}
    }

# ============================================================================
# 1. IMPORTAÇÃO E PREPARAÇÃO DOS DADOS
# ============================================================================

query = '''
        select *, cast(safra_ajustado as varchar) as anomes
        from
            tb_producao_veiculos_partic
        where
            cast(safra_ajustado as int) >= 202503 
            and cast(safra_ajustado as int) <= 202507
            and segmento_pf_pj = 'F'
            and flag2 = 'LEVES'
            and anomesdia = 20250110
        '''

df = wr.athena.read_sql(query)

# Ajustando tipos de dados
df['taxa_'] = df['taxa_'].astype(float)
df['valor_prod'] = df['valor_prod'].astype(float)

# Análise exploratória
print("="*60)
print("ANÁLISE EXPLORATÓRIA")
print("="*60)
print(f"\nQuantidade total de contratos: {len(df):,}")
print(f"\nDistribuição por safra:")
print(df['anomes'].value_counts().sort_index())
print(f"\nEstatísticas de Taxa:")
print(df['taxa_'].describe())
print(f"\nEstatísticas de Produção:")
print(df['valor_prod'].describe())

# ============================================================================
# 2. SEGMENTAÇÃO E AGREGAÇÃO
# ============================================================================

# ============================================================================
# CONFIGURAÇÃO DE PARÂMETROS - AJUSTE AQUI
# ============================================================================

# Definindo variáveis do modelo
prod_var = 'valor_prod'  # Variável de produção/volume
taxa_var = 'taxa_'       # Variável de taxa de juros

# ESCOLHA AS VARIÁVEIS CATEGÓRICAS PARA SEGMENTAÇÃO
# Opções disponíveis: 'uf', 'rating_price', 'seg_cliente', etc.
cat_vars = ['uf', 'rating_price', 'seg_cliente']

# Outras opções de segmentação (descomente para usar):
# cat_vars = ['uf', 'rating_price']  # Apenas UF e Rating
# cat_vars = ['seg_cliente', 'rating_price']  # Apenas Cliente e Rating
# cat_vars = ['uf']  # Apenas por UF

print("\n" + "="*60)
print("CONFIGURAÇÃO DO MODELO")
print("="*60)
print(f"Variável de Produção: {prod_var}")
print(f"Variável de Taxa: {taxa_var}")
print(f"Variáveis Categóricas: {cat_vars}")
print("="*60)

# Validar se as variáveis categóricas existem no dataframe
missing_vars = [var for var in cat_vars if var not in df.columns]
if missing_vars:
    raise ValueError(f"Variáveis não encontradas no dataframe: {missing_vars}")

# Criando faixas de taxa
df['range_taxa'] = pd.qcut(df[taxa_var], q=5, labels=['Muito Baixa', 'Baixa', 'Média', 'Alta', 'Muito Alta'])

# Agregando dados por segmentos
aux = df[[taxa_var, prod_var] + cat_vars + ['range_taxa', 'anomes']].groupby(
    cat_vars + ['range_taxa', 'anomes'], 
    observed=True
).agg(
    qtd_obs=(prod_var, 'count'),
    taxa_media=(taxa_var, 'mean'),
    producao_media=(prod_var, 'mean'),
    producao_total=(prod_var, 'sum')
).reset_index()

print(f"\nDataset agregado: {len(aux):,} linhas")
print(f"Observações mínimas por grupo: {aux['qtd_obs'].min()}")

# ============================================================================
# 3. MODELAGEM - ELASTICIDADE PREÇO
# ============================================================================

# Construindo fórmula do modelo com interações
y = 'producao_total'
string_model = f'{y} ~ {taxa_var}'

for var in cat_vars:
    string_model = string_model + f' + {taxa_var}:C({var})'

print("\n" + "="*60)
print("MODELO DE ELASTICIDADE")
print("="*60)
print(f"\nFórmula: {string_model}\n")

# Ajustando variáveis categóricas
for variavel in cat_vars:
    aux[variavel] = aux[variavel].astype('category')

# Fit do modelo OLS
model = smf.ols(formula=string_model, data=aux).fit()
print(model.summary())

# ============================================================================
# 4. ANÁLISE DE PERFORMANCE DO MODELO
# ============================================================================

aux['y_pred'] = model.predict(aux)
aux['y'] = aux['producao_total']
aux['erro'] = aux['y_pred'] - aux['y']
aux['erro_relativo'] = aux['erro'] / aux['y']

# Métricas
mae = mean_absolute_error(aux['y'], aux['y_pred'])
rmse = np.sqrt(mean_squared_error(aux['y'], aux['y_pred']))
r2 = r2_score(aux['y'], aux['y_pred'])
mape = np.mean(np.abs(aux['erro_relativo'])) * 100

print("\n" + "="*60)
print("MÉTRICAS DE PERFORMANCE")
print("="*60)
print(f"R² Score: {r2:.4f}")
print(f"MAE: R$ {mae:,.2f}")
print(f"RMSE: R$ {rmse:,.2f}")
print(f"MAPE: {mape:.2f}%")

# Visualizações
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# 1. Real vs Previsto
axes[0, 0].scatter(aux['y'], aux['y_pred'], alpha=0.5)
axes[0, 0].plot([aux['y'].min(), aux['y'].max()], 
                [aux['y'].min(), aux['y'].max()], 'r--', lw=2)
axes[0, 0].set_xlabel('Produção Real')
axes[0, 0].set_ylabel('Produção Prevista')
axes[0, 0].set_title(f'Real vs Previsto (R² = {r2:.4f})')

# 2. Distribuição dos erros
axes[0, 1].hist(aux['erro_relativo'], bins=50, edgecolor='black')
axes[0, 1].axvline(0, color='red', linestyle='--', linewidth=2)
axes[0, 1].set_xlabel('Erro Relativo')
axes[0, 1].set_ylabel('Frequência')
axes[0, 1].set_title('Distribuição dos Erros Relativos')

# 3. Boxplot erro relativo
axes[1, 0].boxplot(aux['erro_relativo'])
axes[1, 0].set_ylabel('Erro Relativo')
axes[1, 0].set_title('Boxplot - Erro Relativo')

# 4. Resíduos vs fitted
axes[1, 1].scatter(aux['y_pred'], aux['erro'], alpha=0.5)
axes[1, 1].axhline(0, color='red', linestyle='--', linewidth=2)
axes[1, 1].set_xlabel('Valores Ajustados')
axes[1, 1].set_ylabel('Resíduos')
axes[1, 1].set_title('Resíduos vs Valores Ajustados')

plt.tight_layout()
plt.show()

# ============================================================================
# 5. CÁLCULO DE ELASTICIDADES POR SEGMENTO
# ============================================================================

def calcular_elasticidade(modelo, dados, taxa_var, cat_vars):
    """
    Calcula elasticidade preço-demanda por segmento
    Elasticidade = (dQ/dP) * (P/Q)
    """
    elasticidades = []
    
    for idx, row in dados.iterrows():
        # Coeficiente base
        coef = modelo.params[taxa_var]
        
        # Adicionar efeitos das interações
        for var in cat_vars:
            var_value = row[var]
            param_name = f'{taxa_var}:C({var})[T.{var_value}]'
            if param_name in modelo.params.index:
                coef += modelo.params[param_name]
        
        # Elasticidade = coef * (taxa / producao)
        elasticidade = coef * (row['taxa_media'] / row['producao_total'])
        
        elasticidades.append({
            'uf': row['uf'],
            'rating_price': row['rating_price'],
            'seg_cliente': row['seg_cliente'],
            'range_taxa': row['range_taxa'],
            'taxa_media': row['taxa_media'],
            'producao_total': row['producao_total'],
            'elasticidade': elasticidade,
            'sensibilidade': 'Alta' if abs(elasticidade) > 1 else 'Baixa'
        })
    
    return pd.DataFrame(elasticidades)

df_elasticidades = calcular_elasticidade(model, aux, taxa_var, cat_vars)

print("\n" + "="*60)
print("ELASTICIDADES POR SEGMENTO")
print("="*60)
print("\nTop 10 segmentos mais elásticos (em valor absoluto):")
print(df_elasticidades.nlargest(10, 'elasticidade')[
    ['uf', 'rating_price', 'seg_cliente', 'taxa_media', 'elasticidade']
])

# ============================================================================
# 6. SIMULAÇÃO DE CENÁRIOS DE PRECIFICAÇÃO
# ============================================================================

def simular_cenarios(dados_base, modelo, taxa_var, variacao_taxa_range):
    """
    Simula diferentes cenários de precificação
    variacao_taxa_range: lista de variações percentuais (ex: [-10, -5, 0, 5, 10])
    """
    resultados = []
    
    for variacao_pct in variacao_taxa_range:
        df_cenario = dados_base.copy()
        
        # Aplicar variação na taxa
        df_cenario[taxa_var] = df_cenario['taxa_media'] * (1 + variacao_pct/100)
        
        # Prever nova produção
        df_cenario['producao_prevista'] = modelo.predict(df_cenario)
        
        # Garantir que produção não seja negativa
        df_cenario['producao_prevista'] = df_cenario['producao_prevista'].clip(lower=0)
        
        # Calcular impactos
        producao_total_atual = df_cenario['producao_total'].sum()
        producao_total_nova = df_cenario['producao_prevista'].sum()
        variacao_producao = ((producao_total_nova / producao_total_atual) - 1) * 100
        
        resultados.append({
            'variacao_taxa_pct': variacao_pct,
            'producao_atual': producao_total_atual,
            'producao_nova': producao_total_nova,
            'variacao_producao_pct': variacao_producao,
            'variacao_producao_valor': producao_total_nova - producao_total_atual
        })
    
    return pd.DataFrame(resultados)

# Simulando cenários
variacoes = [-20, -15, -10, -5, 0, 5, 10, 15, 20]
df_cenarios = simular_cenarios(aux, model, variacoes)

print("\n" + "="*60)
print("SIMULAÇÃO DE CENÁRIOS")
print("="*60)
print("\nImpacto de variações na taxa sobre a produção total:\n")
print(df_cenarios.to_string(index=False))

# Visualização dos cenários
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Curva de demanda
axes[0].plot(df_cenarios['variacao_taxa_pct'], 
             df_cenarios['variacao_producao_pct'], 
             marker='o', linewidth=2, markersize=8)
axes[0].axhline(0, color='red', linestyle='--', alpha=0.5)
axes[0].axvline(0, color='red', linestyle='--', alpha=0.5)
axes[0].set_xlabel('Variação na Taxa (%)')
axes[0].set_ylabel('Variação na Produção (%)')
axes[0].set_title('Curva de Elasticidade - Efeito da Taxa na Produção')
axes[0].grid(True, alpha=0.3)

# Produção absoluta
axes[1].bar(df_cenarios['variacao_taxa_pct'], 
            df_cenarios['producao_nova']/1e6,
            color=['red' if x < 0 else 'green' for x in df_cenarios['variacao_producao_pct']])
axes[1].set_xlabel('Variação na Taxa (%)')
axes[1].set_ylabel('Produção Total (R$ Milhões)')
axes[1].set_title('Produção Total por Cenário')
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# ============================================================================
# 7. ANÁLISE DE OPORTUNIDADES POR SEGMENTO
# ============================================================================

def identificar_oportunidades(dados_elasticidades, limiar_elasticidade=-0.5):
    """
    Identifica segmentos com oportunidades de ganho
    Segmentos elásticos (|elasticidade| > limiar) podem se beneficiar de redução de taxa
    """
    oportunidades = dados_elasticidades[
        dados_elasticidades['elasticidade'] < limiar_elasticidade
    ].copy()
    
    # Simular redução de 5% na taxa
    oportunidades['taxa_nova'] = oportunidades['taxa_media'] * 0.95
    oportunidades['producao_incremental_estimada'] = (
        oportunidades['producao_total'] * 
        abs(oportunidades['elasticidade']) * 0.05
    )
    
    oportunidades = oportunidades.sort_values(
        'producao_incremental_estimada', ascending=False
    )
    
    return oportunidades

def plotar_resultados(aux, r2, df_cenarios):
    """Gera visualizações do modelo"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Real vs Previsto
    axes[0, 0].scatter(aux['y'], aux['y_pred'], alpha=0.5)
    axes[0, 0].plot([aux['y'].min(), aux['y'].max()], 
                    [aux['y'].min(), aux['y'].max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('Produção Real')
    axes[0, 0].set_ylabel('Produção Prevista')
    axes[0, 0].set_title(f'Real vs Previsto (R² = {r2:.4f})')
    
    # 2. Distribuição dos erros
    axes[0, 1].hist(aux['erro_relativo'], bins=50, edgecolor='black')
    axes[0, 1].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[0, 1].set_xlabel('Erro Relativo')
    axes[0, 1].set_ylabel('Frequência')
    axes[0, 1].set_title('Distribuição dos Erros Relativos')
    
    # 3. Boxplot erro relativo
    axes[1, 0].boxplot(aux['erro_relativo'])
    axes[1, 0].set_ylabel('Erro Relativo')
    axes[1, 0].set_title('Boxplot - Erro Relativo')
    
    # 4. Resíduos vs fitted
    axes[1, 1].scatter(aux['y_pred'], aux['erro'], alpha=0.5)
    axes[1, 1].axhline(0, color='red', linestyle='--', linewidth=2)
    axes[1, 1].set_xlabel('Valores Ajustados')
    axes[1, 1].set_ylabel('Resíduos')
    axes[1, 1].set_title('Resíduos vs Valores Ajustados')
    
    plt.tight_layout()
    plt.show()
    
    # Visualização dos cenários
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Curva de demanda
    axes[0].plot(df_cenarios['variacao_taxa_pct'], 
                 df_cenarios['variacao_producao_pct'], 
                 marker='o', linewidth=2, markersize=8)
    axes[0].axhline(0, color='red', linestyle='--', alpha=0.5)
    axes[0].axvline(0, color='red', linestyle='--', alpha=0.5)
    axes[0].set_xlabel('Variação na Taxa (%)')
    axes[0].set_ylabel('Variação na Produção (%)')
    axes[0].set_title('Curva de Elasticidade - Efeito da Taxa na Produção')
    axes[0].grid(True, alpha=0.3)
    
    # Produção absoluta
    axes[1].bar(df_cenarios['variacao_taxa_pct'], 
                df_cenarios['producao_nova']/1e6,
                color=['red' if x < 0 else 'green' for x in df_cenarios['variacao_producao_pct']])
    axes[1].set_xlabel('Variação na Taxa (%)')
    axes[1].set_ylabel('Produção Total (R$ Milhões)')
    axes[1].set_title('Produção Total por Cenário')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()

# ============================================================================
# EXECUÇÃO PRINCIPAL
# ============================================================================

print("\n" + "="*60)
print("OPORTUNIDADES DE CRESCIMENTO")
print("="*60)
print("\nTop 10 segmentos com maior potencial de crescimento")
print("(com redução de 5% na taxa):\n")
print(df_oportunidades.head(10)[
    ['uf', 'rating_price', 'seg_cliente', 'taxa_media', 
     'elasticidade', 'producao_incremental_estimada']
].to_string(index=False))

print("\n" + "="*60)
print("RESUMO EXECUTIVO")
print("="*60)
print(f"\nProdução Total Atual: R$ {aux['producao_total'].sum():,.2f}")
print(f"Elasticidade Média: {df_elasticidades['elasticidade'].mean():.4f}")
print(f"Segmentos Altamente Elásticos: {(abs(df_elasticidades['elasticidade']) > 1).sum()}")
print(f"Potencial de Crescimento (redução 5% taxa): R$ {df_oportunidades['producao_incremental_estimada'].sum():,.2f}")

# ============================================================================
# 8. PREPARAÇÃO PARA INTEGRAÇÃO COM MFL (Margem Financeira Líquida)
# ============================================================================

print("\n" + "="*60)
print("PREPARAÇÃO PARA CÁLCULO DE MFL")
print("="*60)
print("""
Para integração futura com calculadora de MFL, você precisará:

1. Dados necessários:
   - Taxa de juros (já temos)
   - Produção total por segmento (já temos)
   - Taxa de inadimplência esperada por segmento
   - Custo de captação
   - Despesas operacionais
   - Prazo médio das operações
   
2. Fórmula básica da MFL:
   MFL = Receita Financeira - Custo de Captação - Perda Esperada - Despesas Operacionais
   
3. Estrutura sugerida:
   def calcular_mfl(producao, taxa, inadimplencia, custo_captacao, desp_op):
       receita = producao * taxa
       custo = producao * custo_captacao
       perda = producao * inadimplencia
       mfl = receita - custo - perda - desp_op
       return mfl

4. Integração com cenários:
   - Para cada cenário de taxa, calcular a MFL esperada
   - Otimizar: max(MFL) considerando elasticidade
   - Trade-off: taxa vs volume vs margem
""")

# Exportar resultados para uso futuro
aux.to_csv('dados_modelo_elasticidade.csv', index=False)
df_elasticidades.to_csv('elasticidades_por_segmento.csv', index=False)
df_cenarios.to_csv('simulacao_cenarios.csv', index=False)
df_oportunidades.to_csv('oportunidades_crescimento.csv', index=False)

print("\n✓ Arquivos exportados com sucesso!")
print("  - dados_modelo_elasticidade.csv")
print("  - elasticidades_por_segmento.csv")
print("  - simulacao_cenarios.csv")
print("  - oportunidades_crescimento.csv")