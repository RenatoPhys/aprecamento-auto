"""
Modelo de Elasticidade de Demanda - Financiamento de Veículos
Análise completa de elasticidade-preço e simulação de cenários
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import datetime

# Bibliotecas para modelagem
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from scipy import stats

warnings.filterwarnings('ignore')

# Configuração de visualização
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ==========================================
# 1. CARREGAMENTO E PREPARAÇÃO DOS DADOS
# ==========================================

def load_and_prepare_data(anomes_list):
    """
    Carrega e prepara os dados para modelagem
    
    Parameters:
    anomes_list: lista com os períodos a serem analisados
    """
    
    # Query parametrizada
    anomes_str = ','.join([str(x) for x in anomes_list])
    query = f'''
        SELECT *
        FROM tb_funil_veiculos
        WHERE anomes IN ({anomes_str})
    '''
    
    # Carregamento dos dados
    print(f"📊 Carregando dados dos períodos: {anomes_str}")
    df = wr.athena.read_sql(query)
    
    # Conversão de tipos
    df['pct_txa_ofrt_simu_pmro_vers'] = df['pct_txa_ofrt_simu_pmro_vers'].astype(float)
    df['valor_prod'] = df['valor_prod'].astype(float)
    
    # Criação de features adicionais
    df['log_valor_prod'] = np.log1p(df['valor_prod'])  # Log para normalização
    df['taxa_squared'] = df['pct_txa_ofrt_simu_pmro_vers'] ** 2  # Termo quadrático
    
    print(f"✅ Dados carregados: {df.shape[0]:,} registros")
    print(f"📅 Períodos disponíveis: {df['anomes'].unique()}")
    
    return df

# ==========================================
# 2. ANÁLISE EXPLORATÓRIA (EDA)
# ==========================================

def perform_eda(df, prod_var, taxa_var, cat_vars):
    """
    Realiza análise exploratória dos dados
    """
    print("\n" + "="*60)
    print("📈 ANÁLISE EXPLORATÓRIA DOS DADOS")
    print("="*60)
    
    # Estatísticas descritivas
    print("\n📊 Estatísticas da Taxa de Juros:")
    print(df[taxa_var].describe())
    
    print("\n📊 Estatísticas da Produção:")
    print(df[prod_var].describe())
    
    # Criar figura com subplots
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Distribuição da Taxa
    ax1 = plt.subplot(2, 3, 1)
    df[taxa_var].hist(bins=50, edgecolor='black', alpha=0.7)
    plt.axvline(df[taxa_var].mean(), color='red', linestyle='--', label=f'Média: {df[taxa_var].mean():.2f}%')
    plt.xlabel('Taxa (%)')
    plt.ylabel('Frequência')
    plt.title('Distribuição da Taxa de Juros')
    plt.legend()
    
    # 2. Distribuição da Produção
    ax2 = plt.subplot(2, 3, 2)
    df[prod_var].hist(bins=50, edgecolor='black', alpha=0.7)
    plt.axvline(df[prod_var].mean(), color='red', linestyle='--', label=f'Média: R$ {df[prod_var].mean():,.0f}')
    plt.xlabel('Valor Produzido (R$)')
    plt.ylabel('Frequência')
    plt.title('Distribuição da Produção')
    plt.legend()
    
    # 3. Scatter plot Taxa vs Produção
    ax3 = plt.subplot(2, 3, 3)
    plt.scatter(df[taxa_var], df[prod_var], alpha=0.5, s=10)
    z = np.polyfit(df[taxa_var], df[prod_var], 1)
    p = np.poly1d(z)
    plt.plot(df[taxa_var].sort_values(), p(df[taxa_var].sort_values()), 
             "r--", alpha=0.8, label='Tendência Linear')
    plt.xlabel('Taxa (%)')
    plt.ylabel('Valor Produzido (R$)')
    plt.title('Relação Taxa vs Produção')
    plt.legend()
    
    # 4. Boxplot da Taxa por UF
    ax4 = plt.subplot(2, 3, 4)
    if 'uf' in cat_vars:
        df.boxplot(column=taxa_var, by='uf', ax=ax4, rot=90)
        plt.suptitle('')
        plt.title('Distribuição da Taxa por UF')
        plt.xlabel('UF')
        plt.ylabel('Taxa (%)')
    
    # 5. Produção média por segmento
    ax5 = plt.subplot(2, 3, 5)
    if 'seg_cliente' in cat_vars:
        prod_by_seg = df.groupby('seg_cliente')[prod_var].mean().sort_values(ascending=False)
        prod_by_seg.plot(kind='bar', ax=ax5)
        plt.title('Produção Média por Segmento de Cliente')
        plt.xlabel('Segmento')
        plt.ylabel('Produção Média (R$)')
        plt.xticks(rotation=45)
    
    # 6. Heatmap de correlação
    ax6 = plt.subplot(2, 3, 6)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlation_matrix = df[numeric_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, ax=ax6, cbar_kws={'label': 'Correlação'})
    plt.title('Matriz de Correlação')
    
    plt.tight_layout()
    plt.show()
    
    # Análise por período
    print("\n📅 Análise por Período (anomes):")
    period_analysis = df.groupby('anomes').agg({
        prod_var: ['count', 'mean', 'sum'],
        taxa_var: ['mean', 'std']
    }).round(2)
    print(period_analysis)

# ==========================================
# 3. ANÁLISE DE FEATURES E IMPORTÂNCIA
# ==========================================

def analyze_feature_importance(df, prod_var, taxa_var, cat_vars):
    """
    Analisa a importância das features para o modelo
    """
    print("\n" + "="*60)
    print("🔍 ANÁLISE DE IMPORTÂNCIA DAS FEATURES")
    print("="*60)
    
    # Criar dataframe agregado
    df['range_taxa'] = pd.qcut(df[taxa_var], q=5, labels=['Muito Baixa', 'Baixa', 'Média', 'Alta', 'Muito Alta'])
    
    results = []
    
    # Análise univariada para cada variável categórica
    for var in cat_vars:
        # Calcular produção média por categoria
        prod_by_cat = df.groupby(var)[prod_var].agg(['mean', 'std', 'count'])
        
        # ANOVA para testar significância
        groups = [group[prod_var].values for name, group in df.groupby(var)]
        f_stat, p_value = stats.f_oneway(*groups)
        
        # Coeficiente de variação
        cv = prod_by_cat['std'].mean() / prod_by_cat['mean'].mean()
        
        results.append({
            'Variável': var,
            'Categorias': df[var].nunique(),
            'F-statistic': f_stat,
            'P-value': p_value,
            'Coef. Variação': cv,
            'Significante': 'Sim' if p_value < 0.05 else 'Não'
        })
    
    # DataFrame com resultados
    importance_df = pd.DataFrame(results)
    importance_df = importance_df.sort_values('F-statistic', ascending=False)
    
    print("\n📊 Importância das Variáveis Categóricas (ANOVA):")
    print(importance_df.to_string(index=False))
    
    # Visualização
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # F-statistics
    axes[0].barh(importance_df['Variável'], importance_df['F-statistic'])
    axes[0].set_xlabel('F-statistic')
    axes[0].set_title('Importância das Features (F-statistic)')
    axes[0].invert_yaxis()
    
    # P-values
    axes[1].barh(importance_df['Variável'], -np.log10(importance_df['P-value']))
    axes[1].axvline(x=-np.log10(0.05), color='r', linestyle='--', label='p=0.05')
    axes[1].set_xlabel('-log10(P-value)')
    axes[1].set_title('Significância Estatística das Features')
    axes[1].legend()
    axes[1].invert_yaxis()
    
    plt.tight_layout()
    plt.show()
    
    return importance_df

# ==========================================
# 4. PREPARAÇÃO PARA MODELAGEM
# ==========================================

def prepare_modeling_data(df, prod_var, taxa_var, cat_vars, train_periods, test_periods):
    """
    Prepara os dados para treino e teste
    """
    print("\n" + "="*60)
    print("🔧 PREPARAÇÃO DOS DADOS PARA MODELAGEM")
    print("="*60)
    
    # Criar range de taxa
    df['range_taxa'] = pd.qcut(df[taxa_var], q=5)
    
    # Agregação dos dados
    agg_dict = {
        'qtd_obs': (prod_var, 'count'),
        'taxa_media': (taxa_var, 'mean'),
        'producao_simulacao': (prod_var, 'mean'),
        'producao_total': (prod_var, 'sum')
    }
    
    aux = df.groupby(cat_vars + ['range_taxa', 'anomes'], observed=True).agg(**agg_dict).reset_index()
    
    # Renomear coluna de taxa
    aux.rename(columns={'taxa_media': taxa_var}, inplace=True)
    
    # Separar treino e teste por período
    train_data = aux[aux['anomes'].isin(train_periods)].copy()
    test_data = aux[aux['anomes'].isin(test_periods)].copy()
    
    print(f"📊 Dados de Treino: {len(train_data)} registros (períodos: {train_periods})")
    print(f"📊 Dados de Teste: {len(test_data)} registros (períodos: {test_periods})")
    
    # Converter variáveis categóricas
    for var in cat_vars:
        train_data[var] = train_data[var].astype('category')
        test_data[var] = test_data[var].astype('category')
    
    return train_data, test_data

# ==========================================
# 5. MODELAGEM E ELASTICIDADES
# ==========================================

def build_elasticity_model(train_data, test_data, prod_var, taxa_var, cat_vars):
    """
    Constrói modelo de elasticidade com interações
    """
    print("\n" + "="*60)
    print("📊 CONSTRUÇÃO DO MODELO DE ELASTICIDADE")
    print("="*60)
    
    # Variável dependente
    y = 'producao_simulacao'
    
    # Construir fórmula do modelo com interações
    formula = f'{y} ~ {taxa_var}'
    
    # Adicionar interações para capturar elasticidades por segmento
    for var in cat_vars:
        formula += f' + C({var}) + {taxa_var}:C({var})'
    
    # Adicionar termo quadrático para capturar não-linearidade
    formula += f' + I({taxa_var}**2)'
    
    print(f"\n📝 Fórmula do modelo: {formula}")
    
    # Ajustar modelo
    model = smf.ols(formula=formula, data=train_data).fit()
    
    print("\n📈 SUMÁRIO DO MODELO:")
    print(model.summary())
    
    # Fazer previsões
    train_data['y_pred'] = model.predict(train_data)
    test_data['y_pred'] = model.predict(test_data)
    
    return model, train_data, test_data

# ==========================================
# 6. CÁLCULO DE ELASTICIDADES POR SEGMENTO
# ==========================================

def calculate_segment_elasticities(model, data, taxa_var, cat_vars):
    """
    Calcula elasticidades por segmento
    """
    print("\n" + "="*60)
    print("💹 ELASTICIDADES POR SEGMENTO")
    print("="*60)
    
    elasticities = {}
    
    # Taxa média para cálculo
    taxa_media = data[taxa_var].mean()
    
    # Elasticidade base (sem segmentação)
    base_elasticity = model.params[taxa_var]
    
    print(f"\n📊 Elasticidade Base: {base_elasticity:.4f}")
    print(f"   Interpretação: Aumento de 1% na taxa → {base_elasticity:.2f}% na produção")
    
    # Calcular elasticidade para cada combinação de segmentos
    for var in cat_vars:
        elasticities[var] = {}
        categories = data[var].unique()
        
        print(f"\n📈 Elasticidades por {var}:")
        
        for cat in categories:
            # Buscar coeficiente de interação
            interaction_term = f'{taxa_var}:C({var})[T.{cat}]'
            
            if interaction_term in model.params:
                segment_elasticity = base_elasticity + model.params[interaction_term]
            else:
                segment_elasticity = base_elasticity
            
            elasticities[var][cat] = segment_elasticity
            
            # Calcular impacto percentual
            impact = (segment_elasticity * taxa_media) / 100
            
            print(f"   {cat}: {segment_elasticity:.4f} (Impacto de 1pp na taxa: {impact:.2%})")
    
    # Criar DataFrame com elasticidades
    elasticity_df = pd.DataFrame()
    for var in cat_vars:
        temp_df = pd.DataFrame(list(elasticities[var].items()), 
                               columns=['Segmento', 'Elasticidade'])
        temp_df['Variável'] = var
        elasticity_df = pd.concat([elasticity_df, temp_df], ignore_index=True)
    
    # Visualização das elasticidades
    fig, axes = plt.subplots(1, len(cat_vars), figsize=(6*len(cat_vars), 5))
    if len(cat_vars) == 1:
        axes = [axes]
    
    for idx, var in enumerate(cat_vars):
        data_plot = elasticity_df[elasticity_df['Variável'] == var]
        axes[idx].barh(data_plot['Segmento'], data_plot['Elasticidade'])
        axes[idx].set_xlabel('Elasticidade')
        axes[idx].set_title(f'Elasticidade por {var}')
        axes[idx].axvline(x=0, color='r', linestyle='--', alpha=0.5)
        
        # Adicionar valores nas barras
        for i, (seg, elast) in enumerate(zip(data_plot['Segmento'], data_plot['Elasticidade'])):
            axes[idx].text(elast, i, f'{elast:.3f}', va='center')
    
    plt.tight_layout()
    plt.show()
    
    return elasticities

# ==========================================
# 7. DIAGNÓSTICO DO MODELO
# ==========================================

def model_diagnostics(model, train_data, test_data):
    """
    Realiza diagnósticos completos do modelo
    """
    print("\n" + "="*60)
    print("🔍 DIAGNÓSTICO DO MODELO")
    print("="*60)
    
    # Métricas para treino e teste
    datasets = {'Treino': train_data, 'Teste': test_data}
    metrics = {}
    
    for name, data in datasets.items():
        y_true = data['producao_simulacao']
        y_pred = data['y_pred']
        
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100
        
        metrics[name] = {
            'MAE': mae,
            'RMSE': rmse,
            'R²': r2,
            'MAPE': mape
        }
        
        print(f"\n📊 Métricas - {name}:")
        print(f"   MAE:  R$ {mae:,.2f}")
        print(f"   RMSE: R$ {rmse:,.2f}")
        print(f"   R²:   {r2:.4f}")
        print(f"   MAPE: {mape:.2f}%")
    
    # Plots de diagnóstico
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 1. Valores preditos vs reais (Treino)
    axes[0, 0].scatter(train_data['producao_simulacao'], train_data['y_pred'], alpha=0.5)
    axes[0, 0].plot([train_data['producao_simulacao'].min(), train_data['producao_simulacao'].max()],
                    [train_data['producao_simulacao'].min(), train_data['producao_simulacao'].max()],
                    'r--', lw=2)
    axes[0, 0].set_xlabel('Valor Real')
    axes[0, 0].set_ylabel('Valor Predito')
    axes[0, 0].set_title(f'Predito vs Real - Treino (R²={metrics["Treino"]["R²"]:.3f})')
    
    # 2. Valores preditos vs reais (Teste)
    axes[0, 1].scatter(test_data['producao_simulacao'], test_data['y_pred'], alpha=0.5, color='orange')
    axes[0, 1].plot([test_data['producao_simulacao'].min(), test_data['producao_simulacao'].max()],
                    [test_data['producao_simulacao'].min(), test_data['producao_simulacao'].max()],
                    'r--', lw=2)
    axes[0, 1].set_xlabel('Valor Real')
    axes[0, 1].set_ylabel('Valor Predito')
    axes[0, 1].set_title(f'Predito vs Real - Teste (R²={metrics["Teste"]["R²"]:.3f})')
    
    # 3. Distribuição dos resíduos
    train_residuals = train_data['producao_simulacao'] - train_data['y_pred']
    test_residuals = test_data['producao_simulacao'] - test_data['y_pred']
    
    axes[0, 2].hist(train_residuals, bins=30, alpha=0.7, label='Treino', edgecolor='black')
    axes[0, 2].hist(test_residuals, bins=30, alpha=0.7, label='Teste', edgecolor='black')
    axes[0, 2].axvline(x=0, color='r', linestyle='--')
    axes[0, 2].set_xlabel('Resíduos')
    axes[0, 2].set_ylabel('Frequência')
    axes[0, 2].set_title('Distribuição dos Resíduos')
    axes[0, 2].legend()
    
    # 4. Q-Q Plot dos resíduos
    stats.probplot(train_residuals, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot - Resíduos do Treino')
    
    # 5. Resíduos vs Valores Ajustados
    axes[1, 1].scatter(train_data['y_pred'], train_residuals, alpha=0.5)
    axes[1, 1].axhline(y=0, color='r', linestyle='--')
    axes[1, 1].set_xlabel('Valores Ajustados')
    axes[1, 1].set_ylabel('Resíduos')
    axes[1, 1].set_title('Resíduos vs Valores Ajustados')
    
    # 6. Boxplot dos erros relativos
    train_data['erro_relativo'] = (train_data['y_pred'] - train_data['producao_simulacao']) / train_data['producao_simulacao']
    test_data['erro_relativo'] = (test_data['y_pred'] - test_data['producao_simulacao']) / test_data['producao_simulacao']
    
    box_data = [train_data['erro_relativo'].dropna(), test_data['erro_relativo'].dropna()]
    axes[1, 2].boxplot(box_data, labels=['Treino', 'Teste'])
    axes[1, 2].set_ylabel('Erro Relativo')
    axes[1, 2].set_title('Distribuição do Erro Relativo')
    axes[1, 2].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()
    
    # Teste de normalidade dos resíduos
    _, p_value = stats.shapiro(train_residuals[:5000] if len(train_residuals) > 5000 else train_residuals)
    print(f"\n📊 Teste de Normalidade dos Resíduos (Shapiro-Wilk):")
    print(f"   P-valor: {p_value:.4f}")
    print(f"   Conclusão: {'Resíduos seguem distribuição normal' if p_value > 0.05 else 'Resíduos NÃO seguem distribuição normal'}")
    
    return metrics

# ==========================================
# 8. SIMULAÇÃO DE CENÁRIOS
# ==========================================

def simulate_pricing_scenarios(model, base_data, taxa_var, cat_vars):
    """
    Simula diferentes cenários de precificação
    """
    print("\n" + "="*60)
    print("🎯 SIMULAÇÃO DE CENÁRIOS DE PRECIFICAÇÃO")
    print("="*60)
    
    # Taxa base (média atual)
    taxa_base = base_data[taxa_var].mean()
    
    # Definir cenários de mudança na taxa
    scenarios = {
        'Redução Agressiva': -2.0,  # -2 pontos percentuais
        'Redução Moderada': -1.0,   # -1 ponto percentual
        'Redução Leve': -0.5,       # -0.5 ponto percentual
        'Baseline': 0.0,            # Sem mudança
        'Aumento Leve': 0.5,        # +0.5 ponto percentual
        'Aumento Moderado': 1.0,    # +1 ponto percentual
        'Aumento Agressivo': 2.0    # +2 pontos percentuais
    }
    
    results = []
    
    # Para cada cenário
    for scenario_name, taxa_change in scenarios.items():
        # Criar cópia dos dados
        scenario_data = base_data.copy()
        
        # Aplicar mudança na taxa
        scenario_data[taxa_var] = scenario_data[taxa_var] + taxa_change
        
        # Prever produção
        scenario_data['y_pred_scenario'] = model.predict(scenario_data)
        
        # Calcular produção total
        prod_total_base = base_data['producao_simulacao'].sum()
        prod_total_scenario = scenario_data['y_pred_scenario'].sum()
        
        # Variação percentual
        var_pct = ((prod_total_scenario - prod_total_base) / prod_total_base) * 100
        
        results.append({
            'Cenário': scenario_name,
            'Δ Taxa (pp)': taxa_change,
            'Taxa Média': taxa_base + taxa_change,
            'Produção Base': prod_total_base,
            'Produção Simulada': prod_total_scenario,
            'Δ Produção': prod_total_scenario - prod_total_base,
            'Δ Produção (%)': var_pct
        })
    
    # DataFrame com resultados
    scenarios_df = pd.DataFrame(results)
    
    print("\n📊 Resultados das Simulações:")
    print(scenarios_df.to_string(index=False))
    
    # Visualização
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. Impacto na produção total
    colors = ['red' if x < 0 else 'green' if x > 0 else 'gray' for x in scenarios_df['Δ Produção (%)']]
    axes[0].barh(scenarios_df['Cenário'], scenarios_df['Δ Produção (%)'], color=colors)
    axes[0].axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    axes[0].set_xlabel('Variação na Produção (%)')
    axes[0].set_title('Impacto dos Cenários na Produção Total')
    
    # Adicionar valores nas barras
    for i, (cenario, valor) in enumerate(zip(scenarios_df['Cenário'], scenarios_df['Δ Produção (%)'])):
        axes[0].text(valor, i, f'{valor:.1f}%', va='center', 
                    ha='left' if valor >= 0 else 'right')
    
    # 2. Curva de elasticidade
    taxa_range = np.linspace(taxa_base - 3, taxa_base + 3, 100)
    base_data_sim = base_data.copy()
    producao_sim = []
    
    for t in taxa_range:
        base_data_sim[taxa_var] = t
        pred = model.predict(base_data_sim).sum()
        producao_sim.append(pred)
    
    axes[1].plot(taxa_range, producao_sim, 'b-', linewidth=2)
    axes[1].scatter(scenarios_df['Taxa Média'], 
                   [scenarios_df[scenarios_df['Cenário']==s]['Produção Simulada'].values[0] 
                    for s in scenarios_df['Cenário']], 
                   color='red', s=100, zorder=5)
    axes[1].axvline(x=taxa_base, color='gray', linestyle='--', alpha=0.5, label='Taxa Atual')
    axes[1].set_xlabel('Taxa (%)')
    axes[1].set_ylabel('Produção Total Estimada')
    axes[1].set_title('Curva de Resposta à Taxa')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Análise por segmento (exemplo com UF)
    if 'uf' in cat_vars:
        print("\n📊 Impacto por UF (Cenário: Redução Moderada -1pp):")
        scenario_data = base_data.copy()
        scenario_data[taxa_var] = scenario_data[taxa_var] - 1.0
        scenario_data['y_pred_scenario'] = model.predict(scenario_data)
        
        impact_by_uf = scenario_data.groupby('uf').agg({
            'producao_simulacao': 'sum',
            'y_pred_scenario': 'sum'
        })
        impact_by_uf['Δ (%)'] = ((impact_by_uf['y_pred_scenario'] - impact_by_uf['producao_simulacao']) 
                                 / impact_by_uf['producao_simulacao'] * 100)
        impact_by_uf = impact_by_uf.sort_values('Δ (%)', ascending=False)
        
        print(impact_by_uf[['Δ (%)']].head(10))
    
    return scenarios_df

# ==========================================
# 9. FUNÇÃO PRINCIPAL DE EXECUÇÃO
# ==========================================

def run_elasticity_analysis(anomes_treino, anomes_teste):
    """
    Executa análise completa de elasticidade
    
    Parameters:
    anomes_treino: lista de períodos para treino (ex: [202501, 202502, 202503])
    anomes_teste: lista de períodos para teste (ex: [202504])
    """
    
    print("="*60)
    print("🚀 MODELO DE ELASTICIDADE - FINANCIAMENTO DE VEÍCULOS")
    print("="*60)
    print(f"📅 Períodos de Treino: {anomes_treino}")
    print(f"📅 Períodos de Teste: {anomes_teste}")
    
    # Definir variáveis
    prod_var = 'valor_prod'
    taxa_var = 'pct_txa_ofrt_simu_pmro_vers'
    cat_vars = ['uf', 'rating_price', 'seg_cliente']
    
    # 1. Carregar dados
    all_periods = anomes_treino + anomes_teste
    df = load_and_prepare_data(all_periods)
    
    # 2. Análise exploratória
    perform_eda(df, prod_var, taxa_var, cat_vars)
    
    # 3. Análise de importância das features
    importance_df = analyze_feature_importance(df, prod_var, taxa_var, cat_vars)
    
    # 4. Preparar dados para modelagem
    train_data, test_data = prepare_modeling_data(
        df, prod_var, taxa_var, cat_vars, anomes_treino, anomes_teste
    )
    
    # 5. Construir modelo
    model, train_data, test_data = build_elasticity_model(
        train_data, test_data, prod_var, taxa_var, cat_vars
    )
    
    # 6. Calcular elasticidades por segmento
    elasticities = calculate_segment_elasticities(
        model, train_data, taxa_var, cat_vars
    )
    
    # 7. Diagnóstico do modelo
    metrics = model_diagnostics(model, train_data, test_data)
    
    # 8. Simular cenários
    scenarios_df = simulate_pricing_scenarios(
        model, train_data, taxa_var, cat_vars
    )
    
    print("\n" + "="*60)
    print("✅ ANÁLISE CONCLUÍDA COM SUCESSO!")
    print("="*60)
    
    return {
        'model': model,
        'train_data': train_data,
        'test_data': test_data,
        'elasticities': elasticities,
        'metrics': metrics,
        'scenarios': scenarios_df
    }

# ==========================================
# EXECUÇÃO
# ==========================================

if __name__ == "__main__":
    # Definir períodos de análise
    ANOMES_TREINO = [202501, 202502, 202503]  # Janeiro a Março 2025
    ANOMES_TESTE = [202504]                   # Abril 2025
    
    # Executar análise completa
    results = run_elasticity_analysis(ANOMES_TREINO, ANOMES_TESTE)
    
    # Os resultados podem ser acessados através do dicionário 'results'
    # results['model'] - modelo treinado
    # results['elasticities'] - elasticidades por segmento
    # results['metrics'] - métricas de performance
    # results['scenarios'] - cenários simulados