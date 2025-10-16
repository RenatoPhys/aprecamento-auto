"""
Modelo de Elasticidade de Demanda - Financiamento de Veículos
Análise completa de elasticidade-preço e simulação de cenários
Com variáveis categóricas parametrizáveis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import datetime
from typing import List, Dict, Tuple, Optional

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
# CONFIGURAÇÃO DAS VARIÁVEIS DISPONÍVEIS
# ==========================================

# Lista de todas as variáveis categóricas disponíveis no dataset
AVAILABLE_CATEGORICAL_VARS = [
    'uf',                    # Unidade Federativa
    'rating_price',          # Rating de preço
    'seg_cliente',           # Segmento de cliente
    'tipo_veiculo',          # Tipo de veículo (se disponível)
    'canal_venda',           # Canal de venda (se disponível)
    'regiao',               # Região geográfica (se disponível)
    'faixa_financiamento',   # Faixa de valor financiado (se disponível)
]

# Configurações padrão
DEFAULT_CAT_VARS = ['uf', 'rating_price', 'seg_cliente']
DEFAULT_PROD_VAR = 'valor_prod'
DEFAULT_TAXA_VAR = 'pct_txa_ofrt_simu_pmro_vers'

# ==========================================
# 1. VALIDAÇÃO E PREPARAÇÃO DAS VARIÁVEIS
# ==========================================

def validate_categorical_vars(df: pd.DataFrame, 
                            requested_vars: List[str], 
                            min_categories: int = 2,
                            max_categories: int = 50) -> List[str]:
    """
    Valida e filtra as variáveis categóricas solicitadas
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame com os dados
    requested_vars : List[str]
        Lista de variáveis categóricas desejadas
    min_categories : int
        Número mínimo de categorias únicas para considerar a variável
    max_categories : int
        Número máximo de categorias únicas para considerar a variável
    
    Returns:
    --------
    List[str]: Lista de variáveis categóricas validadas
    """
    print("\n" + "="*60)
    print("🔍 VALIDAÇÃO DAS VARIÁVEIS CATEGÓRICAS")
    print("="*60)
    
    validated_vars = []
    
    for var in requested_vars:
        # Verificar se a variável existe no dataset
        if var not in df.columns:
            print(f"⚠️  Variável '{var}' não encontrada no dataset")
            continue
        
        # Verificar número de categorias únicas
        n_unique = df[var].nunique()
        
        if n_unique < min_categories:
            print(f"⚠️  Variável '{var}' tem apenas {n_unique} categoria(s) - ignorada")
            continue
        
        if n_unique > max_categories:
            print(f"⚠️  Variável '{var}' tem {n_unique} categorias - muito alta cardinalidade")
            # Perguntar se deseja continuar ou criar faixas
            print(f"    Considerando apenas as top {max_categories} categorias")
            # Você pode implementar lógica para agrupar categorias menos frequentes
        
        # Verificar valores nulos
        null_pct = df[var].isnull().sum() / len(df) * 100
        if null_pct > 50:
            print(f"⚠️  Variável '{var}' tem {null_pct:.1f}% de valores nulos - ignorada")
            continue
        
        validated_vars.append(var)
        print(f"✅ Variável '{var}' validada: {n_unique} categorias, {null_pct:.1f}% nulos")
    
    if not validated_vars:
        print("⚠️  Nenhuma variável categórica válida encontrada!")
        print("    Usando variáveis padrão disponíveis...")
        # Tentar usar variáveis padrão
        for var in DEFAULT_CAT_VARS:
            if var in df.columns and df[var].nunique() >= min_categories:
                validated_vars.append(var)
                print(f"✅ Variável padrão '{var}' adicionada")
    
    print(f"\n📊 Variáveis categóricas finais: {validated_vars}")
    return validated_vars

# ==========================================
# 2. CARREGAMENTO E PREPARAÇÃO DOS DADOS
# ==========================================

def load_and_prepare_data(anomes_list: List[int]) -> pd.DataFrame:
    """
    Carrega e prepara os dados para modelagem
    
    Parameters:
    -----------
    anomes_list : List[int]
        Lista com os períodos a serem analisados
    
    Returns:
    --------
    pd.DataFrame: DataFrame preparado
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
    df['log_valor_prod'] = np.log1p(df['valor_prod'])
    df['taxa_squared'] = df['pct_txa_ofrt_simu_pmro_vers'] ** 2
    
    print(f"✅ Dados carregados: {df.shape[0]:,} registros")
    print(f"📅 Períodos disponíveis: {df['anomes'].unique()}")
    
    return df

# ==========================================
# 3. ANÁLISE EXPLORATÓRIA ADAPTATIVA
# ==========================================

def perform_eda(df: pd.DataFrame, 
                prod_var: str, 
                taxa_var: str, 
                cat_vars: List[str]) -> None:
    """
    Realiza análise exploratória dos dados adaptada às variáveis categóricas
    """
    print("\n" + "="*60)
    print("📈 ANÁLISE EXPLORATÓRIA DOS DADOS")
    print("="*60)
    
    # Estatísticas descritivas
    print("\n📊 Estatísticas da Taxa de Juros:")
    print(df[taxa_var].describe())
    
    print("\n📊 Estatísticas da Produção:")
    print(df[prod_var].describe())
    
    # Ajustar número de subplots baseado nas variáveis disponíveis
    n_cat_vars = len(cat_vars)
    n_plots = 3 + min(n_cat_vars, 3)  # Plots básicos + plots por categoria (máx 3)
    
    # Calcular layout da figura
    n_cols = 3
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig = plt.figure(figsize=(20, 6 * n_rows))
    
    plot_idx = 1
    
    # 1. Distribuição da Taxa
    ax = plt.subplot(n_rows, n_cols, plot_idx)
    df[taxa_var].hist(bins=50, edgecolor='black', alpha=0.7, ax=ax)
    ax.axvline(df[taxa_var].mean(), color='red', linestyle='--', 
               label=f'Média: {df[taxa_var].mean():.2f}%')
    ax.set_xlabel('Taxa (%)')
    ax.set_ylabel('Frequência')
    ax.set_title('Distribuição da Taxa de Juros')
    ax.legend()
    plot_idx += 1
    
    # 2. Distribuição da Produção
    ax = plt.subplot(n_rows, n_cols, plot_idx)
    df[prod_var].hist(bins=50, edgecolor='black', alpha=0.7, ax=ax)
    ax.axvline(df[prod_var].mean(), color='red', linestyle='--', 
               label=f'Média: R$ {df[prod_var].mean():,.0f}')
    ax.set_xlabel('Valor Produzido (R$)')
    ax.set_ylabel('Frequência')
    ax.set_title('Distribuição da Produção')
    ax.legend()
    plot_idx += 1
    
    # 3. Scatter plot Taxa vs Produção
    ax = plt.subplot(n_rows, n_cols, plot_idx)
    ax.scatter(df[taxa_var], df[prod_var], alpha=0.5, s=10)
    z = np.polyfit(df[taxa_var], df[prod_var], 1)
    p = np.poly1d(z)
    ax.plot(df[taxa_var].sort_values(), p(df[taxa_var].sort_values()), 
            "r--", alpha=0.8, label='Tendência Linear')
    ax.set_xlabel('Taxa (%)')
    ax.set_ylabel('Valor Produzido (R$)')
    ax.set_title('Relação Taxa vs Produção')
    ax.legend()
    plot_idx += 1
    
    # Plots específicos para cada variável categórica (até 3)
    for i, var in enumerate(cat_vars[:3]):
        if plot_idx > n_rows * n_cols:
            break
            
        ax = plt.subplot(n_rows, n_cols, plot_idx)
        
        # Decidir tipo de plot baseado no número de categorias
        n_categories = df[var].nunique()
        
        if n_categories <= 10:
            # Boxplot para poucas categorias
            df.boxplot(column=taxa_var, by=var, ax=ax, rot=45)
            plt.suptitle('')
            ax.set_title(f'Taxa por {var}')
            ax.set_xlabel(var)
            ax.set_ylabel('Taxa (%)')
        else:
            # Barplot com top 10 categorias para muitas categorias
            top_cats = df.groupby(var)[prod_var].mean().nlargest(10)
            top_cats.plot(kind='bar', ax=ax)
            ax.set_title(f'Top 10 {var} por Produção Média')
            ax.set_xlabel(var)
            ax.set_ylabel('Produção Média (R$)')
            plt.xticks(rotation=45, ha='right')
        
        plot_idx += 1
    
    plt.tight_layout()
    plt.show()
    
    # Análise por período
    print("\n📅 Análise por Período (anomes):")
    period_analysis = df.groupby('anomes').agg({
        prod_var: ['count', 'mean', 'sum'],
        taxa_var: ['mean', 'std']
    }).round(2)
    print(period_analysis)
    
    # Análise por variável categórica
    for var in cat_vars:
        print(f"\n📊 Análise por {var}:")
        var_analysis = df.groupby(var).agg({
            prod_var: ['count', 'mean', 'sum'],
            taxa_var: ['mean', 'std']
        }).round(2)
        print(var_analysis.head(10))

# ==========================================
# 4. PREPARAÇÃO PARA MODELAGEM FLEXÍVEL
# ==========================================

def prepare_modeling_data(df: pd.DataFrame, 
                         prod_var: str, 
                         taxa_var: str, 
                         cat_vars: List[str], 
                         train_periods: List[int], 
                         test_periods: List[int]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepara os dados para treino e teste com variáveis categóricas flexíveis
    """
    print("\n" + "="*60)
    print("🔧 PREPARAÇÃO DOS DADOS PARA MODELAGEM")
    print("="*60)
    
    # Criar range de taxa
    df['range_taxa'] = pd.qcut(df[taxa_var], q=5)
    
    # Definir colunas de agrupamento dinamicamente
    groupby_cols = cat_vars + ['range_taxa', 'anomes']
    
    # Agregação dos dados
    agg_dict = {
        'qtd_obs': (prod_var, 'count'),
        'taxa_media': (taxa_var, 'mean'),
        'producao_simulacao': (prod_var, 'mean'),
        'producao_total': (prod_var, 'sum')
    }
    
    aux = df.groupby(groupby_cols, observed=True).agg(**agg_dict).reset_index()
    
    # Renomear coluna de taxa
    aux.rename(columns={'taxa_media': taxa_var}, inplace=True)
    
    # Separar treino e teste por período
    train_data = aux[aux['anomes'].isin(train_periods)].copy()
    test_data = aux[aux['anomes'].isin(test_periods)].copy()
    
    print(f"📊 Dados de Treino: {len(train_data)} registros (períodos: {train_periods})")
    print(f"📊 Dados de Teste: {len(test_data)} registros (períodos: {test_periods})")
    print(f"📊 Variáveis categóricas utilizadas: {cat_vars}")
    
    # Converter variáveis categóricas
    for var in cat_vars:
        if var in train_data.columns:
            train_data[var] = train_data[var].astype('category')
        if var in test_data.columns:
            test_data[var] = test_data[var].astype('category')
    
    return train_data, test_data

# ==========================================
# 5. MODELAGEM COM FÓRMULA DINÂMICA
# ==========================================

def build_elasticity_model(train_data: pd.DataFrame, 
                          test_data: pd.DataFrame, 
                          prod_var: str, 
                          taxa_var: str, 
                          cat_vars: List[str],
                          include_interactions: bool = True,
                          include_quadratic: bool = True) -> Tuple:
    """
    Constrói modelo de elasticidade com fórmula dinâmica
    
    Parameters:
    -----------
    include_interactions : bool
        Se True, inclui interações entre taxa e variáveis categóricas
    include_quadratic : bool
        Se True, inclui termo quadrático da taxa
    """
    print("\n" + "="*60)
    print("📊 CONSTRUÇÃO DO MODELO DE ELASTICIDADE")
    print("="*60)
    
    # Variável dependente
    y = 'producao_simulacao'
    
    # Construir fórmula do modelo dinamicamente
    formula_parts = [y, '~', taxa_var]
    
    # Adicionar variáveis categóricas e interações se solicitado
    if cat_vars:
        for var in cat_vars:
            formula_parts.append(f' + C({var})')
            if include_interactions:
                formula_parts.append(f' + {taxa_var}:C({var})')
    
    # Adicionar termo quadrático se solicitado
    if include_quadratic:
        formula_parts.append(f' + I({taxa_var}**2)')
    
    formula = ''.join(formula_parts)
    
    print(f"\n📝 Fórmula do modelo: {formula}")
    print(f"   - Variáveis categóricas: {cat_vars if cat_vars else 'Nenhuma'}")
    print(f"   - Interações: {'Sim' if include_interactions and cat_vars else 'Não'}")
    print(f"   - Termo quadrático: {'Sim' if include_quadratic else 'Não'}")
    
    # Ajustar modelo
    try:
        model = smf.ols(formula=formula, data=train_data).fit()
        
        print("\n📈 SUMÁRIO DO MODELO:")
        print(model.summary())
        
        # Fazer previsões
        train_data['y_pred'] = model.predict(train_data)
        test_data['y_pred'] = model.predict(test_data)
        
        return model, train_data, test_data
        
    except Exception as e:
        print(f"❌ Erro ao ajustar modelo: {e}")
        print("    Tentando modelo simplificado...")
        
        # Modelo simplificado sem interações
        simple_formula = f'{y} ~ {taxa_var}'
        if cat_vars:
            for var in cat_vars:
                simple_formula += f' + C({var})'
        
        model = smf.ols(formula=simple_formula, data=train_data).fit()
        
        print("\n📈 SUMÁRIO DO MODELO SIMPLIFICADO:")
        print(model.summary())
        
        train_data['y_pred'] = model.predict(train_data)
        test_data['y_pred'] = model.predict(test_data)
        
        return model, train_data, test_data

# ==========================================
# 6. CÁLCULO DE ELASTICIDADES DINÂMICO
# ==========================================

def calculate_segment_elasticities(model, 
                                  data: pd.DataFrame, 
                                  taxa_var: str, 
                                  cat_vars: List[str]) -> Dict:
    """
    Calcula elasticidades por segmento de forma dinâmica
    """
    print("\n" + "="*60)
    print("💹 ELASTICIDADES POR SEGMENTO")
    print("="*60)
    
    elasticities = {}
    
    # Taxa média para cálculo
    taxa_media = data[taxa_var].mean()
    
    # Elasticidade base (sem segmentação)
    base_elasticity = model.params.get(taxa_var, 0)
    
    print(f"\n📊 Elasticidade Base: {base_elasticity:.4f}")
    print(f"   Interpretação: Aumento de 1% na taxa → {base_elasticity:.2f}% na produção")
    
    if not cat_vars:
        print("\n⚠️  Sem variáveis categóricas para segmentação")
        return {'base': base_elasticity}
    
    # Calcular elasticidade para cada variável categórica
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
    
    # Visualização das elasticidades (adaptativa ao número de variáveis)
    n_vars = len(cat_vars)
    if n_vars > 0:
        fig, axes = plt.subplots(1, min(n_vars, 3), figsize=(6*min(n_vars, 3), 5))
        if n_vars == 1:
            axes = [axes]
        
        for idx, var in enumerate(cat_vars[:3]):  # Mostrar no máximo 3
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
# 7. ANÁLISE DE IMPORTÂNCIA DAS FEATURES
# ==========================================

def analyze_feature_importance(df: pd.DataFrame, 
                              prod_var: str, 
                              taxa_var: str, 
                              cat_vars: List[str]) -> pd.DataFrame:
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
        try:
            # Calcular produção média por categoria
            prod_by_cat = df.groupby(var)[prod_var].agg(['mean', 'std', 'count'])
            
            # ANOVA para testar significância
            groups = [group[prod_var].values for name, group in df.groupby(var)]
            if len(groups) > 1:  # Precisa de pelo menos 2 grupos
                f_stat, p_value = stats.f_oneway(*groups)
            else:
                f_stat, p_value = 0, 1
            
            # Coeficiente de variação
            cv = prod_by_cat['std'].mean() / prod_by_cat['mean'].mean() if prod_by_cat['mean'].mean() != 0 else 0
            
            results.append({
                'Variável': var,
                'Categorias': df[var].nunique(),
                'F-statistic': f_stat,
                'P-value': p_value,
                'Coef. Variação': cv,
                'Significante': 'Sim' if p_value < 0.05 else 'Não'
            })
        except Exception as e:
            print(f"⚠️  Erro ao analisar {var}: {e}")
            continue
    
    if not results:
        print("⚠️  Nenhuma análise de importância disponível")
        return pd.DataFrame()
    
    # DataFrame com resultados
    importance_df = pd.DataFrame(results)
    importance_df = importance_df.sort_values('F-statistic', ascending=False)
    
    print("\n📊 Importância das Variáveis Categóricas (ANOVA):")
    print(importance_df.to_string(index=False))
    
    # Visualização
    if len(importance_df) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # F-statistics
        axes[0].barh(importance_df['Variável'], importance_df['F-statistic'])
        axes[0].set_xlabel('F-statistic')
        axes[0].set_title('Importância das Features (F-statistic)')
        axes[0].invert_yaxis()
        
        # P-values
        axes[1].barh(importance_df['Variável'], -np.log10(importance_df['P-value'] + 1e-10))
        axes[1].axvline(x=-np.log10(0.05), color='r', linestyle='--', label='p=0.05')
        axes[1].set_xlabel('-log10(P-value)')
        axes[1].set_title('Significância Estatística das Features')
        axes[1].legend()
        axes[1].invert_yaxis()
        
        plt.tight_layout()
        plt.show()
    
    return importance_df

# ==========================================
# 8. DIAGNÓSTICO DO MODELO
# ==========================================

def model_diagnostics(model, 
                     train_data: pd.DataFrame, 
                     test_data: pd.DataFrame) -> Dict:
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
        
        # MAPE com tratamento de zeros
        mask = y_true != 0
        if mask.sum() > 0:
            mape = mean_absolute_percentage_error(y_true[mask], y_pred[mask]) * 100
        else:
            mape = np.nan
        
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
        if not np.isnan(mape):
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
    train_data['erro_relativo'] = np.where(
        train_data['producao_simulacao'] != 0,
        (train_data['y_pred'] - train_data['producao_simulacao']) / train_data['producao_simulacao'],
        np.nan
    )
    test_data['erro_relativo'] = np.where(
        test_data['producao_simulacao'] != 0,
        (test_data['y_pred'] - test_data['producao_simulacao']) / test_data['producao_simulacao'],
        np.nan
    )
    
    box_data = [train_data['erro_relativo'].dropna(), test_data['erro_relativo'].dropna()]
    if all(len(d) > 0 for d in box_data):
        axes[1, 2].boxplot(box_data, labels=['Treino', 'Teste'])
        axes[1, 2].set_ylabel('Erro Relativo')
        axes[1, 2].set_title('Distribuição do Erro Relativo')
        axes[1, 2].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()
    
    # Teste de normalidade dos resíduos
    if len(train_residuals) > 0:
        sample_size = min(5000, len(train_residuals))
        _, p_value = stats.shapiro(train_residuals[:sample_size])
        print(f"\n📊 Teste de Normalidade dos Resíduos (Shapiro-Wilk):")
        print(f"   P-valor: {p_value:.4f}")
        print(f"   Conclusão: {'Resíduos seguem distribuição normal' if p_value > 0.05 else 'Resíduos NÃO seguem distribuição normal'}")
    
    return metrics

# ==========================================
# 9. SIMULAÇÃO DE CENÁRIOS
# ==========================================

def simulate_pricing_scenarios(model, 
                              base_data: pd.DataFrame, 
                              taxa_var: str, 
                              cat_vars: List[str]) -> pd.DataFrame:
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
        var_pct = ((prod_total_scenario - prod_total_base) / prod_total_base) * 100 if prod_total_base != 0 else 0
        
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
    taxa_range = np.linspace(max(0, taxa_base - 3), taxa_base + 3, 100)
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
    
    # Análise por segmento (se houver variáveis categóricas)
    if cat_vars and 'uf' in cat_vars:
        print("\n📊 Impacto por UF (Cenário: Redução Moderada -1pp):")
        scenario_data = base_data.copy()
        scenario_data[taxa_var] = scenario_data[taxa_var] - 1.0
        scenario_data['y_pred_scenario'] = model.predict(scenario_data)
        
        impact_by_uf = scenario_data.groupby('uf').agg({
            'producao_simulacao': 'sum',
            'y_pred_scenario': 'sum'
        })
        impact_by_uf['Δ (%)'] = np.where(
            impact_by_uf['producao_simulacao'] != 0,
            ((impact_by_uf['y_pred_scenario'] - impact_by_uf['producao_simulacao']) 
             / impact_by_uf['producao_simulacao'] * 100),
            0
        )
        impact_by_uf = impact_by_uf.sort_values('Δ (%)', ascending=False)
        
        print(impact_by_uf[['Δ (%)']].head(10))
    
    return scenarios_df

# ==========================================
# 10. FUNÇÃO PRINCIPAL DE EXECUÇÃO
# ==========================================

def run_elasticity_analysis(anomes_treino: List[int], 
                          anomes_teste: List[int],
                          categorical_vars: Optional[List[str]] = None,
                          prod_var: Optional[str] = None,
                          taxa_var: Optional[str] = None,
                          include_interactions: bool = True,
                          include_quadratic: bool = True,
                          validate_vars: bool = True) -> Dict:
    """
    Executa análise completa de elasticidade com variáveis parametrizáveis
    
    Parameters:
    -----------
    anomes_treino : List[int]
        Lista de períodos para treino (ex: [202501, 202502, 202503])
    anomes_teste : List[int]
        Lista de períodos para teste (ex: [202504])
    categorical_vars : List[str], optional
        Lista de variáveis categóricas a usar. Se None, usa DEFAULT_CAT_VARS
    prod_var : str, optional
        Variável de produção. Se None, usa DEFAULT_PROD_VAR
    taxa_var : str, optional
        Variável de taxa. Se None, usa DEFAULT_TAXA_VAR
    include_interactions : bool
        Se True, inclui interações no modelo
    include_quadratic : bool
        Se True, inclui termo quadrático
    validate_vars : bool
        Se True, valida as variáveis categóricas
    
    Returns:
    --------
    Dict: Dicionário com todos os resultados da análise
    """
    
    print("="*60)
    print("🚀 MODELO DE ELASTICIDADE - FINANCIAMENTO DE VEÍCULOS")
    print("="*60)
    print(f"📅 Períodos de Treino: {anomes_treino}")
    print(f"📅 Períodos de Teste: {anomes_teste}")
    
    # Usar variáveis padrão se não especificadas
    if prod_var is None:
        prod_var = DEFAULT_PROD_VAR
    if taxa_var is None:
        taxa_var = DEFAULT_TAXA_VAR
    if categorical_vars is None:
        categorical_vars = DEFAULT_CAT_VARS
    
    print(f"\n📊 Configuração:")
    print(f"   - Variável de produção: {prod_var}")
    print(f"   - Variável de taxa: {taxa_var}")
    print(f"   - Variáveis categóricas solicitadas: {categorical_vars}")
    
    # 1. Carregar dados
    all_periods = anomes_treino + anomes_teste
    df = load_and_prepare_data(all_periods)
    
    # 2. Validar variáveis categóricas
    if validate_vars:
        cat_vars = validate_categorical_vars(df, categorical_vars)
    else:
        cat_vars = categorical_vars
        print(f"\n⚠️  Validação de variáveis desabilitada")
        print(f"    Usando: {cat_vars}")
    
    # 3. Análise exploratória
    perform_eda(df, prod_var, taxa_var, cat_vars)
    
    # 4. Análise de importância das features
    if cat_vars:
        importance_df = analyze_feature_importance(df, prod_var, taxa_var, cat_vars)
    else:
        print("\n⚠️  Análise de importância ignorada (sem variáveis categóricas)")
        importance_df = None
    
    # 5. Preparar dados para modelagem
    train_data, test_data = prepare_modeling_data(
        df, prod_var, taxa_var, cat_vars, anomes_treino, anomes_teste
    )
    
    # 6. Construir modelo
    model, train_data, test_data = build_elasticity_model(
        train_data, test_data, prod_var, taxa_var, cat_vars,
        include_interactions=include_interactions,
        include_quadratic=include_quadratic
    )
    
    # 7. Calcular elasticidades por segmento
    elasticities = calculate_segment_elasticities(
        model, train_data, taxa_var, cat_vars
    )
    
    # 8. Diagnóstico do modelo
    metrics = model_diagnostics(model, train_data, test_data)
    
    # 9. Simular cenários
    scenarios_df = simulate_pricing_scenarios(
        model, train_data, taxa_var, cat_vars
    )
    
    print("\n" + "="*60)
    print("✅ ANÁLISE CONCLUÍDA COM SUCESSO!")
    print("="*60)
    print(f"   Variáveis categóricas utilizadas: {cat_vars}")
    print("="*60)
    
    return {
        'model': model,
        'train_data': train_data,
        'test_data': test_data,
        'elasticities': elasticities,
        'metrics': metrics,
        'scenarios': scenarios_df,
        'categorical_vars_used': cat_vars,
        'importance_df': importance_df
    }

# ==========================================
# EXEMPLOS DE USO
# ==========================================

if __name__ == "__main__":
    # Definir períodos de análise
    ANOMES_TREINO = [202501, 202502, 202503]  # Janeiro a Março 2025
    ANOMES_TESTE = [202504]                   # Abril 2025
    
    # ====== EXEMPLO 1: Usar configuração padrão ======
    print("\n🔹 EXEMPLO 1: Configuração Padrão")
    results1 = run_elasticity_analysis(
        anomes_treino=ANOMES_TREINO,
        anomes_teste=ANOMES_TESTE
    )
    
    # ====== EXEMPLO 2: Especificar variáveis categóricas ======
    print("\n🔹 EXEMPLO 2: Variáveis Categóricas Customizadas")
    results2 = run_elasticity_analysis(
        anomes_treino=ANOMES_TREINO,
        anomes_teste=ANOMES_TESTE,
        categorical_vars=['uf', 'seg_cliente'],  # Apenas UF e segmento
        include_interactions=True,
        include_quadratic=True
    )
    
    # ====== EXEMPLO 3: Modelo sem interações ======
    print("\n🔹 EXEMPLO 3: Modelo Sem Interações")
    results3 = run_elasticity_analysis(
        anomes_treino=ANOMES_TREINO,
        anomes_teste=ANOMES_TESTE,
        categorical_vars=['rating_price'],  # Apenas rating
        include_interactions=False,  # Sem interações
        include_quadratic=False      # Sem termo quadrático
    )
    
    # ====== EXEMPLO 4: Adicionar novas variáveis categóricas ======
    print("\n🔹 EXEMPLO 4: Testando Novas Variáveis")
    results4 = run_elasticity_analysis(
        anomes_treino=ANOMES_TREINO,
        anomes_teste=ANOMES_TESTE,
        categorical_vars=['uf', 'rating_price', 'seg_cliente', 'tipo_veiculo', 'canal_venda'],
        validate_vars=True  # Validar se existem no dataset
    )
    
    # ====== EXEMPLO 5: Análise mínima (sem segmentação) ======
    print("\n🔹 EXEMPLO 5: Modelo Sem Segmentação")
    results5 = run_elasticity_analysis(
        anomes_treino=ANOMES_TREINO,
        anomes_teste=ANOMES_TESTE,
        categorical_vars=[],  # Sem variáveis categóricas
        include_interactions=False,
        include_quadratic=True
    )
    
    # Os resultados podem ser acessados através dos dicionários
    # results['model'] - modelo treinado
    # results['elasticities'] - elasticidades por segmento
    # results['metrics'] - métricas de performance
    # results['scenarios'] - cenários simulados
    # results['categorical_vars_used'] - variáveis efetivamente utilizadas
    # results['importance_df'] - importância das variáveis