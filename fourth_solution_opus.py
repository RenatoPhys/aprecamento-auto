"""
Modelo Hurdle de Elasticidade - Financiamento de Veículos
Modelo de dois estágios: Probabilidade de Conversão + Valor da Produção
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
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score, 
    mean_absolute_percentage_error, roc_auc_score, 
    classification_report, confusion_matrix
)
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.special import expit  # função logística

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
DEFAULT_CONVERSION_VAR = 'flag_conversao'  # Variável binária de conversão

# ==========================================
# 1. VALIDAÇÃO E PREPARAÇÃO DAS VARIÁVEIS
# ==========================================

def validate_categorical_vars(df: pd.DataFrame, 
                            requested_vars: List[str], 
                            min_categories: int = 2,
                            max_categories: int = 50) -> List[str]:
    """
    Valida e filtra as variáveis categóricas solicitadas
    """
    print("\n" + "="*60)
    print("🔍 VALIDAÇÃO DAS VARIÁVEIS CATEGÓRICAS")
    print("="*60)
    
    validated_vars = []
    
    for var in requested_vars:
        if var not in df.columns:
            print(f"⚠️  Variável '{var}' não encontrada no dataset")
            continue
        
        n_unique = df[var].nunique()
        
        if n_unique < min_categories:
            print(f"⚠️  Variável '{var}' tem apenas {n_unique} categoria(s) - ignorada")
            continue
        
        if n_unique > max_categories:
            print(f"⚠️  Variável '{var}' tem {n_unique} categorias - muito alta cardinalidade")
            print(f"    Considerando apenas as top {max_categories} categorias")
        
        null_pct = df[var].isnull().sum() / len(df) * 100
        if null_pct > 50:
            print(f"⚠️  Variável '{var}' tem {null_pct:.1f}% de valores nulos - ignorada")
            continue
        
        validated_vars.append(var)
        print(f"✅ Variável '{var}' validada: {n_unique} categorias, {null_pct:.1f}% nulos")
    
    if not validated_vars:
        print("⚠️  Nenhuma variável categórica válida encontrada!")
        print("    Usando variáveis padrão disponíveis...")
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
    Carrega e prepara os dados para modelagem Hurdle
    """
    anomes_str = ','.join([str(x) for x in anomes_list])
    query = f'''
        SELECT *
        FROM tb_funil_veiculos
        WHERE anomes IN ({anomes_str})
    '''
    
    print(f"📊 Carregando dados dos períodos: {anomes_str}")
    df = wr.athena.read_sql(query)
    
    # Conversão de tipos
    df['pct_txa_ofrt_simu_pmro_vers'] = df['pct_txa_ofrt_simu_pmro_vers'].astype(float)
    df['valor_prod'] = df['valor_prod'].astype(float)
    
    # CRIAR VARIÁVEL DE CONVERSÃO (binária)
    # Assumindo que valor_prod > 0 indica conversão
    df['flag_conversao'] = (df['valor_prod'] > 0).astype(int)
    
    # Criar features adicionais
    df['log_valor_prod'] = np.log1p(df['valor_prod'])
    df['taxa_squared'] = df['pct_txa_ofrt_simu_pmro_vers'] ** 2
    
    print(f"✅ Dados carregados: {df.shape[0]:,} registros")
    print(f"📅 Períodos disponíveis: {df['anomes'].unique()}")
    print(f"📊 Taxa de conversão geral: {df['flag_conversao'].mean():.2%}")
    print(f"💰 Ticket médio (conversões): R$ {df[df['flag_conversao']==1]['valor_prod'].mean():,.2f}")
    
    return df

# ==========================================
# 3. ANÁLISE EXPLORATÓRIA - MODELO HURDLE
# ==========================================

def perform_hurdle_eda(df: pd.DataFrame, 
                       prod_var: str, 
                       taxa_var: str, 
                       conversion_var: str,
                       cat_vars: List[str]) -> None:
    """
    Análise exploratória adaptada para modelo Hurdle
    """
    print("\n" + "="*60)
    print("📈 ANÁLISE EXPLORATÓRIA - MODELO HURDLE")
    print("="*60)
    
    # Estatísticas gerais
    print("\n📊 ESTATÍSTICAS GERAIS:")
    print(f"   Total de observações: {len(df):,}")
    print(f"   Taxa de conversão: {df[conversion_var].mean():.2%}")
    print(f"   Número de conversões: {df[conversion_var].sum():,}")
    
    # Estatísticas por conversão
    converters = df[df[conversion_var] == 1]
    non_converters = df[df[conversion_var] == 0]
    
    print("\n📊 COMPARAÇÃO: Conversores vs Não-Conversores")
    print(f"   Taxa média - Conversores: {converters[taxa_var].mean():.2f}%")
    print(f"   Taxa média - Não-Conversores: {non_converters[taxa_var].mean():.2f}%")
    print(f"   Diferença: {non_converters[taxa_var].mean() - converters[taxa_var].mean():.2f}pp")
    
    # Visualizações
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Taxa de conversão por faixa de taxa
    ax1 = plt.subplot(3, 3, 1)
    df['taxa_bins'] = pd.qcut(df[taxa_var], q=10, duplicates='drop')
    conv_by_taxa = df.groupby('taxa_bins')[conversion_var].mean()
    conv_by_taxa.plot(kind='bar', ax=ax1, color='steelblue')
    ax1.set_xlabel('Faixa de Taxa')
    ax1.set_ylabel('Taxa de Conversão')
    ax1.set_title('Taxa de Conversão por Faixa de Taxa de Juros')
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. Distribuição da taxa por status de conversão
    ax2 = plt.subplot(3, 3, 2)
    converters[taxa_var].hist(bins=30, alpha=0.6, label='Conversores', ax=ax2, color='green')
    non_converters[taxa_var].hist(bins=30, alpha=0.6, label='Não-Conversores', ax=ax2, color='red')
    ax2.set_xlabel('Taxa (%)')
    ax2.set_ylabel('Frequência')
    ax2.set_title('Distribuição da Taxa por Status de Conversão')
    ax2.legend()
    
    # 3. Produção dos conversores por taxa
    ax3 = plt.subplot(3, 3, 3)
    if len(converters) > 0:
        ax3.scatter(converters[taxa_var], converters[prod_var], alpha=0.5, s=10)
        z = np.polyfit(converters[taxa_var], converters[prod_var], 1)
        p = np.poly1d(z)
        ax3.plot(converters[taxa_var].sort_values(), 
                p(converters[taxa_var].sort_values()), 
                "r--", alpha=0.8, label='Tendência')
        ax3.set_xlabel('Taxa (%)')
        ax3.set_ylabel('Valor Produzido (R$)')
        ax3.set_title('Produção vs Taxa (Apenas Conversores)')
        ax3.legend()
    
    # 4. Taxa de conversão por variável categórica (primeira)
    if cat_vars:
        ax4 = plt.subplot(3, 3, 4)
        var = cat_vars[0]
        conv_by_cat = df.groupby(var)[conversion_var].mean().sort_values(ascending=False).head(10)
        conv_by_cat.plot(kind='bar', ax=ax4, color='teal')
        ax4.set_xlabel(var)
        ax4.set_ylabel('Taxa de Conversão')
        ax4.set_title(f'Taxa de Conversão por {var} (Top 10)')
        ax4.tick_params(axis='x', rotation=45)
    
    # 5. Ticket médio por faixa de taxa (conversores)
    ax5 = plt.subplot(3, 3, 5)
    if len(converters) > 0:
        converters['taxa_bins'] = pd.qcut(converters[taxa_var], q=10, duplicates='drop')
        ticket_by_taxa = converters.groupby('taxa_bins')[prod_var].mean()
        ticket_by_taxa.plot(kind='bar', ax=ax5, color='orange')
        ax5.set_xlabel('Faixa de Taxa')
        ax5.set_ylabel('Ticket Médio (R$)')
        ax5.set_title('Ticket Médio por Faixa de Taxa (Conversores)')
        ax5.tick_params(axis='x', rotation=45)
    
    # 6. Curva de conversão suavizada
    ax6 = plt.subplot(3, 3, 6)
    taxa_sorted = df[taxa_var].sort_values()
    window_size = max(100, len(df) // 50)
    
    conv_rates = []
    taxa_points = []
    
    for i in range(0, len(taxa_sorted) - window_size, window_size // 2):
        window_data = df[(df[taxa_var] >= taxa_sorted.iloc[i]) & 
                        (df[taxa_var] <= taxa_sorted.iloc[i + window_size])]
        if len(window_data) > 0:
            conv_rates.append(window_data[conversion_var].mean())
            taxa_points.append(window_data[taxa_var].mean())
    
    if taxa_points:
        ax6.plot(taxa_points, conv_rates, 'b-', linewidth=2)
        ax6.set_xlabel('Taxa (%)')
        ax6.set_ylabel('Taxa de Conversão')
        ax6.set_title('Curva de Conversão (Suavizada)')
        ax6.grid(True, alpha=0.3)
    
    # 7. Análise bivariada - conversão e produção
    if len(cat_vars) > 1:
        ax7 = plt.subplot(3, 3, 7)
        var = cat_vars[1]
        analysis_df = df.groupby(var).agg({
            conversion_var: 'mean',
            prod_var: lambda x: x[x > 0].mean() if (x > 0).any() else 0
        }).reset_index()
        
        ax7_twin = ax7.twinx()
        ax7.bar(range(len(analysis_df)), analysis_df[conversion_var], 
                alpha=0.6, label='Taxa Conversão', color='blue')
        ax7_twin.bar(range(len(analysis_df)), analysis_df[prod_var], 
                    alpha=0.6, label='Ticket Médio', color='orange')
        
        ax7.set_xlabel(var)
        ax7.set_ylabel('Taxa de Conversão', color='blue')
        ax7_twin.set_ylabel('Ticket Médio (R$)', color='orange')
        ax7.set_title(f'Conversão e Ticket por {var}')
        ax7.set_xticks(range(len(analysis_df)))
        ax7.set_xticklabels(analysis_df[var], rotation=45)
    
    # 8. Matriz de correlação
    ax8 = plt.subplot(3, 3, 8)
    numeric_cols = [taxa_var, conversion_var, prod_var, 'taxa_squared']
    correlation_matrix = df[numeric_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, fmt='.3f', 
                cmap='coolwarm', center=0, ax=ax8)
    ax8.set_title('Correlações - Modelo Hurdle')
    
    # 9. Análise de elasticidade empírica
    ax9 = plt.subplot(3, 3, 9)
    elasticity_df = pd.DataFrame()
    
    percentiles = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    for p in percentiles:
        taxa_threshold = df[taxa_var].quantile(p/100)
        subset = df[df[taxa_var] <= taxa_threshold]
        if len(subset) > 100:
            conv_rate = subset[conversion_var].mean()
            avg_prod = subset[subset[conversion_var] == 1][prod_var].mean() if (subset[conversion_var] == 1).any() else 0
            elasticity_df = pd.concat([elasticity_df, pd.DataFrame({
                'percentil': [p],
                'taxa_limite': [taxa_threshold],
                'conv_rate': [conv_rate],
                'ticket_medio': [avg_prod],
                'receita_esperada': [conv_rate * avg_prod]
            })], ignore_index=True)
    
    if not elasticity_df.empty:
        ax9_twin = ax9.twinx()
        ax9.plot(elasticity_df['taxa_limite'], elasticity_df['conv_rate'], 
                'b-', label='Taxa Conversão', marker='o')
        ax9_twin.plot(elasticity_df['taxa_limite'], elasticity_df['receita_esperada'], 
                     'g-', label='Receita Esperada', marker='s')
        ax9.set_xlabel('Taxa Limite (%)')
        ax9.set_ylabel('Taxa de Conversão', color='blue')
        ax9_twin.set_ylabel('Receita Esperada (R$)', color='green')
        ax9.set_title('Análise de Elasticidade Empírica')
        ax9.legend(loc='upper left')
        ax9_twin.legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()

# ==========================================
# 4. PREPARAÇÃO DE DADOS PARA MODELO HURDLE
# ==========================================

def prepare_hurdle_data(df: pd.DataFrame,
                        prod_var: str,
                        taxa_var: str,
                        conversion_var: str,
                        cat_vars: List[str],
                        train_periods: List[int],
                        test_periods: List[int]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepara dados para modelo Hurdle de dois estágios
    """
    print("\n" + "="*60)
    print("🔧 PREPARAÇÃO DOS DADOS - MODELO HURDLE")
    print("="*60)
    
    # Separar treino e teste
    train_data = df[df['anomes'].isin(train_periods)].copy()
    test_data = df[df['anomes'].isin(test_periods)].copy()
    
    print(f"\n📊 DADOS DE TREINO:")
    print(f"   Total: {len(train_data):,} registros")
    print(f"   Conversões: {train_data[conversion_var].sum():,} ({train_data[conversion_var].mean():.2%})")
    print(f"   Ticket médio: R$ {train_data[train_data[conversion_var]==1][prod_var].mean():,.2f}")
    
    print(f"\n📊 DADOS DE TESTE:")
    print(f"   Total: {len(test_data):,} registros")
    print(f"   Conversões: {test_data[conversion_var].sum():,} ({test_data[conversion_var].mean():.2%})")
    print(f"   Ticket médio: R$ {test_data[test_data[conversion_var]==1][prod_var].mean():,.2f}")
    
    # Converter variáveis categóricas
    for var in cat_vars:
        if var in train_data.columns:
            train_data[var] = train_data[var].astype('category')
        if var in test_data.columns:
            test_data[var] = test_data[var].astype('category')
    
    return train_data, test_data

# ==========================================
# 5. MODELO HURDLE - ESTÁGIO 1: CONVERSÃO
# ==========================================

def build_conversion_model(train_data: pd.DataFrame,
                         test_data: pd.DataFrame,
                         conversion_var: str,
                         taxa_var: str,
                         cat_vars: List[str],
                         include_interactions: bool = True) -> Tuple:
    """
    Estágio 1: Modelo Logístico para Probabilidade de Conversão
    """
    print("\n" + "="*60)
    print("🎯 ESTÁGIO 1: MODELO DE CONVERSÃO (LOGÍSTICO)")
    print("="*60)
    
    # Construir fórmula
    formula_parts = [conversion_var, '~', taxa_var, '+ I(', taxa_var, '**2)']
    
    if cat_vars:
        for var in cat_vars:
            formula_parts.append(f' + C({var})')
            if include_interactions:
                formula_parts.append(f' + {taxa_var}:C({var})')
    
    formula = ''.join(formula_parts)
    
    print(f"\n📝 Fórmula do modelo de conversão: {formula}")
    
    try:
        # Ajustar modelo logístico
        conversion_model = smf.logit(formula=formula, data=train_data).fit(disp=0)
        
        print("\n📈 SUMÁRIO DO MODELO DE CONVERSÃO:")
        print(conversion_model.summary())
        
        # Previsões
        train_data['prob_conversao'] = conversion_model.predict(train_data)
        test_data['prob_conversao'] = conversion_model.predict(test_data)
        
        # Previsões binárias (threshold = 0.5)
        train_data['pred_conversao'] = (train_data['prob_conversao'] > 0.5).astype(int)
        test_data['pred_conversao'] = (test_data['prob_conversao'] > 0.5).astype(int)
        
        # Métricas
        print("\n📊 MÉTRICAS DO MODELO DE CONVERSÃO:")
        
        for name, data in [('Treino', train_data), ('Teste', test_data)]:
            auc = roc_auc_score(data[conversion_var], data['prob_conversao'])
            accuracy = (data[conversion_var] == data['pred_conversao']).mean()
            
            print(f"\n   {name}:")
            print(f"      AUC-ROC: {auc:.4f}")
            print(f"      Acurácia: {accuracy:.4f}")
            
            # Matriz de confusão
            cm = confusion_matrix(data[conversion_var], data['pred_conversao'])
            precision = cm[1,1] / (cm[0,1] + cm[1,1]) if (cm[0,1] + cm[1,1]) > 0 else 0
            recall = cm[1,1] / (cm[1,0] + cm[1,1]) if (cm[1,0] + cm[1,1]) > 0 else 0
            
            print(f"      Precisão: {precision:.4f}")
            print(f"      Recall: {recall:.4f}")
        
        return conversion_model, train_data, test_data
        
    except Exception as e:
        print(f"❌ Erro no modelo de conversão: {e}")
        print("   Tentando modelo simplificado...")
        
        simple_formula = f'{conversion_var} ~ {taxa_var} + I({taxa_var}**2)'
        conversion_model = smf.logit(formula=simple_formula, data=train_data).fit(disp=0)
        
        train_data['prob_conversao'] = conversion_model.predict(train_data)
        test_data['prob_conversao'] = conversion_model.predict(test_data)
        
        return conversion_model, train_data, test_data

# ==========================================
# 6. MODELO HURDLE - ESTÁGIO 2: PRODUÇÃO
# ==========================================

def build_production_model(train_data: pd.DataFrame,
                         test_data: pd.DataFrame,
                         prod_var: str,
                         taxa_var: str,
                         cat_vars: List[str],
                         conversion_var: str,
                         include_interactions: bool = True) -> Tuple:
    """
    Estágio 2: Modelo Linear para Valor da Produção (dado que converteu)
    """
    print("\n" + "="*60)
    print("💰 ESTÁGIO 2: MODELO DE PRODUÇÃO (LINEAR)")
    print("="*60)
    
    # Filtrar apenas conversores
    train_converters = train_data[train_data[conversion_var] == 1].copy()
    test_converters = test_data[test_data[conversion_var] == 1].copy()
    
    print(f"\n📊 Dados para modelo de produção:")
    print(f"   Treino: {len(train_converters):,} conversores")
    print(f"   Teste: {len(test_converters):,} conversores")
    
    if len(train_converters) < 30:
        print("⚠️  Poucos conversores para treinar modelo robusto!")
    
    # Usar log da produção para melhor ajuste
    train_converters['log_prod'] = np.log1p(train_converters[prod_var])
    test_converters['log_prod'] = np.log1p(test_converters[prod_var])
    
    # Construir fórmula
    formula_parts = ['log_prod', '~', taxa_var, '+ I(', taxa_var, '**2)']
    
    if cat_vars:
        for var in cat_vars:
            formula_parts.append(f' + C({var})')
            if include_interactions:
                formula_parts.append(f' + {taxa_var}:C({var})')
    
    formula = ''.join(formula_parts)
    
    print(f"\n📝 Fórmula do modelo de produção: {formula}")
    
    try:
        # Ajustar modelo linear
        production_model = smf.ols(formula=formula, data=train_converters).fit()
        
        print("\n📈 SUMÁRIO DO MODELO DE PRODUÇÃO:")
        print(production_model.summary())
        
        # Previsões em log
        train_converters['pred_log_prod'] = production_model.predict(train_converters)
        test_converters['pred_log_prod'] = production_model.predict(test_converters)
        
        # Converter de volta para escala original
        train_converters['pred_prod'] = np.expm1(train_converters['pred_log_prod'])
        test_converters['pred_prod'] = np.expm1(test_converters['pred_log_prod'])
        
        # Aplicar previsões de volta ao dataset completo
        train_data['pred_prod_if_convert'] = 0.0
        test_data['pred_prod_if_convert'] = 0.0
        
        train_data.loc[train_data[conversion_var] == 1, 'pred_prod_if_convert'] = train_converters['pred_prod'].values
        test_data.loc[test_data[conversion_var] == 1, 'pred_prod_if_convert'] = test_converters['pred_prod'].values
        
        # Para não-conversores, usar previsão do modelo
        non_conv_train = train_data[train_data[conversion_var] == 0].copy()
        non_conv_test = test_data[test_data[conversion_var] == 0].copy()
        
        if len(non_conv_train) > 0:
            non_conv_train['log_prod'] = production_model.predict(non_conv_train)
            train_data.loc[train_data[conversion_var] == 0, 'pred_prod_if_convert'] = np.expm1(non_conv_train['log_prod'].values)
        
        if len(non_conv_test) > 0:
            non_conv_test['log_prod'] = production_model.predict(non_conv_test)
            test_data.loc[test_data[conversion_var] == 0, 'pred_prod_if_convert'] = np.expm1(non_conv_test['log_prod'].values)
        
        # Métricas
        print("\n📊 MÉTRICAS DO MODELO DE PRODUÇÃO (apenas conversores):")
        
        for name, data in [('Treino', train_converters), ('Teste', test_converters)]:
            if len(data) > 0:
                mae = mean_absolute_error(data[prod_var], data['pred_prod'])
                rmse = np.sqrt(mean_squared_error(data[prod_var], data['pred_prod']))
                r2 = r2_score(data[prod_var], data['pred_prod'])
                
                print(f"\n   {name}:")
                print(f"      MAE:  R$ {mae:,.2f}")
                print(f"      RMSE: R$ {rmse:,.2f}")
                print(f"      R²:   {r2:.4f}")
        
        return production_model, train_data, test_data
        
    except Exception as e:
        print(f"❌ Erro no modelo de produção: {e}")
        return None, train_data, test_data

# ==========================================
# 7. CÁLCULO DE ELASTICIDADES - MODELO HURDLE
# ==========================================

def calculate_hurdle_elasticities(conversion_model,
                                 production_model,
                                 data: pd.DataFrame,
                                 taxa_var: str,
                                 cat_vars: List[str]) -> Dict:
    """
    Calcula elasticidades para o modelo Hurdle completo
    """
    print("\n" + "="*60)
    print("💹 ELASTICIDADES - MODELO HURDLE")
    print("="*60)
    
    elasticities = {
        'conversao': {},
        'producao': {},
        'total': {}
    }
    
    taxa_media = data[taxa_var].mean()
    
    # ===== ELASTICIDADE DE CONVERSÃO =====
    print("\n📊 ELASTICIDADE DE CONVERSÃO:")
    
    # Coeficiente base da taxa no modelo de conversão
    beta_taxa_conv = conversion_model.params.get(taxa_var, 0)
    beta_taxa2_conv = conversion_model.params.get(f'I({taxa_var} ** 2)', 0)
    
    # Elasticidade na média
    prob_media = expit(conversion_model.predict(data).mean())
    elasticity_conv_base = beta_taxa_conv * (1 - prob_media)
    
    print(f"   Base: {elasticity_conv_base:.4f}")
    print(f"   Interpretação: +1pp na taxa → {elasticity_conv_base*100:.2f}pp na prob. conversão")
    
    # ===== ELASTICIDADE DE PRODUÇÃO =====
    if production_model:
        print("\n💰 ELASTICIDADE DE PRODUÇÃO (conversores):")
        
        # Como usamos log, o coeficiente já é semi-elasticidade
        beta_taxa_prod = production_model.params.get(taxa_var, 0)
        
        print(f"   Base: {beta_taxa_prod:.4f}")
        print(f"   Interpretação: +1pp na taxa → {beta_taxa_prod*100:.2f}% no valor financiado")
    
    # ===== ELASTICIDADE TOTAL =====
    print("\n🎯 ELASTICIDADE TOTAL (Conversão × Produção):")
    
    # Elasticidade total = elasticidade conversão + elasticidade produção
    if production_model:
        elasticity_total = elasticity_conv_base + beta_taxa_prod
        print(f"   Total: {elasticity_total:.4f}")
        print(f"   Interpretação: +1pp na taxa → {elasticity_total*100:.2f}% na receita esperada")
    
    # Calcular por segmento
    if cat_vars:
        for var in cat_vars:
            elasticities['conversao'][var] = {}
            elasticities['producao'][var] = {}
            elasticities['total'][var] = {}
            
            categories = data[var].unique()
            
            print(f"\n📈 Elasticidades por {var}:")
            
            for cat in categories:
                # Elasticidade de conversão por segmento
                interaction_conv = f'{taxa_var}:C({var})[T.{cat}]'
                if interaction_conv in conversion_model.params:
                    elast_conv_seg = elasticity_conv_base + conversion_model.params[interaction_conv] * (1 - prob_media)
                else:
                    elast_conv_seg = elasticity_conv_base
                
                elasticities['conversao'][var][cat] = elast_conv_seg
                
                # Elasticidade de produção por segmento
                if production_model:
                    interaction_prod = f'{taxa_var}:C({var})[T.{cat}]'
                    if interaction_prod in production_model.params:
                        elast_prod_seg = beta_taxa_prod + production_model.params[interaction_prod]
                    else:
                        elast_prod_seg = beta_taxa_prod
                    
                    elasticities['producao'][var][cat] = elast_prod_seg
                    elasticities['total'][var][cat] = elast_conv_seg + elast_prod_seg
                    
                    print(f"   {cat}:")
                    print(f"      Conversão: {elast_conv_seg:.4f}")
                    print(f"      Produção:  {elast_prod_seg:.4f}")
                    print(f"      Total:     {elasticities['total'][var][cat]:.4f}")
    
    return elasticities

# ==========================================
# 8. MODELO HURDLE COMPLETO
# ==========================================

def build_hurdle_model(train_data: pd.DataFrame,
                      test_data: pd.DataFrame,
                      prod_var: str,
                      taxa_var: str,
                      conversion_var: str,
                      cat_vars: List[str],
                      include_interactions: bool = True) -> Dict:
    """
    Constrói o modelo Hurdle completo (dois estágios)
    """
    print("\n" + "="*60)
    print("🚀 CONSTRUINDO MODELO HURDLE COMPLETO")
    print("="*60)
    
    # ESTÁGIO 1: Modelo de Conversão
    conversion_model, train_data, test_data = build_conversion_model(
        train_data, test_data, conversion_var, taxa_var, 
        cat_vars, include_interactions
    )
    
    # ESTÁGIO 2: Modelo de Produção
    production_model, train_data, test_data = build_production_model(
        train_data, test_data, prod_var, taxa_var, 
        cat_vars, conversion_var, include_interactions
    )
    
    # PREVISÃO FINAL: Probabilidade × Valor Esperado
    train_data['pred_receita_esperada'] = train_data['prob_conversao'] * train_data['pred_prod_if_convert']
    test_data['pred_receita_esperada'] = test_data['prob_conversao'] * test_data['pred_prod_if_convert']
    
    # Receita real para comparação
    train_data['receita_real'] = train_data[prod_var]
    test_data['receita_real'] = test_data[prod_var]
    
    print("\n" + "="*60)
    print("📊 MÉTRICAS DO MODELO HURDLE COMPLETO")
    print("="*60)
    
    for name, data in [('Treino', train_data), ('Teste', test_data)]:
        # Métricas de receita esperada
        mae = mean_absolute_error(data['receita_real'], data['pred_receita_esperada'])
        rmse = np.sqrt(mean_squared_error(data['receita_real'], data['pred_receita_esperada']))
        
        # R² apenas para valores positivos
        mask = data['receita_real'] > 0
        if mask.sum() > 0:
            r2 = r2_score(data[mask]['receita_real'], data[mask]['pred_receita_esperada'])
        else:
            r2 = np.nan
        
        print(f"\n   {name} - Receita Esperada:")
        print(f"      MAE:  R$ {mae:,.2f}")
        print(f"      RMSE: R$ {rmse:,.2f}")
        if not np.isnan(r2):
            print(f"      R² (conversores): {r2:.4f}")
        
        # Comparação de totais
        total_real = data['receita_real'].sum()
        total_pred = data['pred_receita_esperada'].sum()
        erro_total = (total_pred - total_real) / total_real * 100
        
        print(f"\n      Total Real: R$ {total_real:,.2f}")
        print(f"      Total Predito: R$ {total_pred:,.2f}")
        print(f"      Erro Total: {erro_total:.2f}%")
    
    return {
        'conversion_model': conversion_model,
        'production_model': production_model,
        'train_data': train_data,
        'test_data': test_data
    }

# ==========================================
# 9. DIAGNÓSTICOS DO MODELO HURDLE
# ==========================================

def hurdle_model_diagnostics(hurdle_results: Dict) -> None:
    """
    Diagnósticos visuais para o modelo Hurdle
    """
    print("\n" + "="*60)
    print("🔍 DIAGNÓSTICOS VISUAIS - MODELO HURDLE")
    print("="*60)
    
    train_data = hurdle_results['train_data']
    test_data = hurdle_results['test_data']
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    
    # 1. ROC Curve - Conversão
    from sklearn.metrics import roc_curve
    
    ax = axes[0, 0]
    for name, data, color in [('Treino', train_data, 'blue'), ('Teste', test_data, 'red')]:
        fpr, tpr, _ = roc_curve(data['flag_conversao'], data['prob_conversao'])
        auc = roc_auc_score(data['flag_conversao'], data['prob_conversao'])
        ax.plot(fpr, tpr, color=color, label=f'{name} (AUC={auc:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax.set_xlabel('Taxa de Falso Positivo')
    ax.set_ylabel('Taxa de Verdadeiro Positivo')
    ax.set_title('Curva ROC - Modelo de Conversão')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Calibração - Probabilidade de Conversão
    ax = axes[0, 1]
    n_bins = 10
    for name, data, color in [('Treino', train_data, 'blue'), ('Teste', test_data, 'red')]:
        prob_bins = pd.qcut(data['prob_conversao'], q=n_bins, duplicates='drop')
        calibration = data.groupby(prob_bins).agg({
            'prob_conversao': 'mean',
            'flag_conversao': 'mean'
        })
        ax.scatter(calibration['prob_conversao'], calibration['flag_conversao'], 
                  color=color, label=name, s=50)
    
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax.set_xlabel('Probabilidade Média Predita')
    ax.set_ylabel('Proporção Real de Conversões')
    ax.set_title('Calibração do Modelo de Conversão')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Predito vs Real - Produção (conversores)
    ax = axes[0, 2]
    converters_train = train_data[train_data['flag_conversao'] == 1]
    converters_test = test_data[test_data['flag_conversao'] == 1]
    
    if len(converters_train) > 0:
        ax.scatter(converters_train['valor_prod'], converters_train['pred_prod_if_convert'], 
                  alpha=0.5, s=10, label='Treino')
    if len(converters_test) > 0:
        ax.scatter(converters_test['valor_prod'], converters_test['pred_prod_if_convert'], 
                  alpha=0.5, s=10, color='orange', label='Teste')
    
    if len(converters_train) > 0 or len(converters_test) > 0:
        all_vals = pd.concat([
            converters_train[['valor_prod', 'pred_prod_if_convert']] if len(converters_train) > 0 else pd.DataFrame(),
            converters_test[['valor_prod', 'pred_prod_if_convert']] if len(converters_test) > 0 else pd.DataFrame()
        ])
        if len(all_vals) > 0:
            max_val = max(all_vals.max().max(), 1)
            ax.plot([0, max_val], [0, max_val], 'r--', alpha=0.5)
    
    ax.set_xlabel('Produção Real (R$)')
    ax.set_ylabel('Produção Predita (R$)')
    ax.set_title('Modelo de Produção - Conversores')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Distribuição de Probabilidades
    ax = axes[1, 0]
    ax.hist(train_data['prob_conversao'], bins=30, alpha=0.6, label='Treino', density=True)
    ax.hist(test_data['prob_conversao'], bins=30, alpha=0.6, label='Teste', density=True)
    ax.set_xlabel('Probabilidade de Conversão')
    ax.set_ylabel('Densidade')
    ax.set_title('Distribuição das Probabilidades Preditas')
    ax.legend()
    
    # 5. Resíduos - Modelo de Produção
    ax = axes[1, 1]
    if len(converters_train) > 0:
        residuals_train = converters_train['valor_prod'] - converters_train['pred_prod_if_convert']
        ax.scatter(converters_train['pred_prod_if_convert'], residuals_train, 
                  alpha=0.5, s=10, label='Treino')
    if len(converters_test) > 0:
        residuals_test = converters_test['valor_prod'] - converters_test['pred_prod_if_convert']
        ax.scatter(converters_test['pred_prod_if_convert'], residuals_test, 
                  alpha=0.5, s=10, color='orange', label='Teste')
    
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax.set_xlabel('Produção Predita (R$)')
    ax.set_ylabel('Resíduos (R$)')
    ax.set_title('Resíduos - Modelo de Produção')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. Receita Esperada vs Real
    ax = axes[1, 2]
    ax.scatter(train_data['receita_real'], train_data['pred_receita_esperada'], 
              alpha=0.3, s=5, label='Treino')
    ax.scatter(test_data['receita_real'], test_data['pred_receita_esperada'], 
              alpha=0.3, s=5, color='orange', label='Teste')
    
    max_val = max(
        train_data[['receita_real', 'pred_receita_esperada']].max().max(),
        test_data[['receita_real', 'pred_receita_esperada']].max().max()
    )
    ax.plot([0, max_val], [0, max_val], 'r--', alpha=0.5)
    ax.set_xlabel('Receita Real (R$)')
    ax.set_ylabel('Receita Esperada Predita (R$)')
    ax.set_title('Modelo Hurdle Completo')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 7. Análise por Taxa - Conversão
    ax = axes[2, 0]
    taxa_bins = pd.qcut(train_data['pct_txa_ofrt_simu_pmro_vers'], q=10, duplicates='drop')
    conv_analysis = train_data.groupby(taxa_bins).agg({
        'flag_conversao': 'mean',
        'prob_conversao': 'mean'
    })
    
    x_pos = range(len(conv_analysis))
    width = 0.35
    ax.bar([p - width/2 for p in x_pos], conv_analysis['flag_conversao'], 
           width, label='Real', alpha=0.7)
    ax.bar([p + width/2 for p in x_pos], conv_analysis['prob_conversao'], 
           width, label='Predito', alpha=0.7)
    ax.set_xlabel('Faixa de Taxa')
    ax.set_ylabel('Taxa de Conversão')
    ax.set_title('Conversão por Faixa de Taxa')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'{x.left:.1f}-{x.right:.1f}' for x in conv_analysis.index], 
                       rotation=45, ha='right')
    ax.legend()
    
    # 8. Análise por Taxa - Produção
    ax = axes[2, 1]
    prod_analysis = converters_train.groupby(
        pd.qcut(converters_train['pct_txa_ofrt_simu_pmro_vers'], q=5, duplicates='drop')
    ).agg({
        'valor_prod': 'mean',
        'pred_prod_if_convert': 'mean'
    })
    
    if len(prod_analysis) > 0:
        x_pos = range(len(prod_analysis))
        ax.bar([p - width/2 for p in x_pos], prod_analysis['valor_prod'], 
               width, label='Real', alpha=0.7)
        ax.bar([p + width/2 for p in x_pos], prod_analysis['pred_prod_if_convert'], 
               width, label='Predito', alpha=0.7)
        ax.set_xlabel('Faixa de Taxa')
        ax.set_ylabel('Ticket Médio (R$)')
        ax.set_title('Ticket Médio por Faixa de Taxa')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f'{x.left:.1f}-{x.right:.1f}' for x in prod_analysis.index], 
                           rotation=45, ha='right')
        ax.legend()
    
    # 9. Feature Importance (simplificado)
    ax = axes[2, 2]
    if hurdle_results['conversion_model']:
        importance_conv = pd.Series(
            np.abs(hurdle_results['conversion_model'].params),
            index=hurdle_results['conversion_model'].params.index
        ).sort_values(ascending=False).head(10)
        
        ax.barh(range(len(importance_conv)), importance_conv.values)
        ax.set_yticks(range(len(importance_conv)))
        ax.set_yticklabels(importance_conv.index)
        ax.set_xlabel('|Coeficiente|')
        ax.set_title('Top 10 Features - Modelo Conversão')
        ax.invert_yaxis()
    
    plt.tight_layout()
    plt.show()

# ==========================================
# 10. SIMULAÇÃO DE CENÁRIOS - MODELO HURDLE
# ==========================================

def simulate_hurdle_scenarios(hurdle_results: Dict,
                             taxa_var: str,
                             cat_vars: List[str]) -> pd.DataFrame:
    """
    Simula cenários de precificação com modelo Hurdle
    """
    print("\n" + "="*60)
    print("🎯 SIMULAÇÃO DE CENÁRIOS - MODELO HURDLE")
    print("="*60)
    
    base_data = hurdle_results['train_data'].copy()
    conversion_model = hurdle_results['conversion_model']
    production_model = hurdle_results['production_model']
    
    # Taxa base
    taxa_base = base_data[taxa_var].mean()
    
    # Cenários
    scenarios = {
        'Redução Agressiva': -2.0,
        'Redução Moderada': -1.0,
        'Redução Leve': -0.5,
        'Baseline': 0.0,
        'Aumento Leve': 0.5,
        'Aumento Moderado': 1.0,
        'Aumento Agressivo': 2.0
    }
    
    results = []
    
    for scenario_name, taxa_change in scenarios.items():
        # Aplicar mudança na taxa
        scenario_data = base_data.copy()
        scenario_data[taxa_var] = scenario_data[taxa_var] + taxa_change
        
        # Prever conversão
        scenario_data['prob_conversao_scenario'] = conversion_model.predict(scenario_data)
        
        # Prever produção
        if production_model:
            scenario_data['log_prod'] = production_model.predict(scenario_data)
            scenario_data['pred_prod_scenario'] = np.expm1(scenario_data['log_prod'])
        else:
            scenario_data['pred_prod_scenario'] = scenario_data['pred_prod_if_convert']
        
        # Receita esperada
        scenario_data['receita_esperada_scenario'] = (
            scenario_data['prob_conversao_scenario'] * 
            scenario_data['pred_prod_scenario']
        )
        
        # Métricas agregadas
        conversao_base = base_data['prob_conversao'].mean()
        conversao_scenario = scenario_data['prob_conversao_scenario'].mean()
        
        ticket_base = base_data['pred_prod_if_convert'].mean()
        ticket_scenario = scenario_data['pred_prod_scenario'].mean()
        
        receita_base = base_data['pred_receita_esperada'].sum()
        receita_scenario = scenario_data['receita_esperada_scenario'].sum()
        
        results.append({
            'Cenário': scenario_name,
            'Δ Taxa (pp)': taxa_change,
            'Taxa Média': taxa_base + taxa_change,
            'Conversão Base (%)': conversao_base * 100,
            'Conversão Cenário (%)': conversao_scenario * 100,
            'Δ Conversão (pp)': (conversao_scenario - conversao_base) * 100,
            'Ticket Base (R$)': ticket_base,
            'Ticket Cenário (R$)': ticket_scenario,
            'Δ Ticket (%)': (ticket_scenario - ticket_base) / ticket_base * 100,
            'Receita Base (R$)': receita_base,
            'Receita Cenário (R$)': receita_scenario,
            'Δ Receita (%)': (receita_scenario - receita_base) / receita_base * 100
        })
    
    scenarios_df = pd.DataFrame(results)
    
    print("\n📊 Resultados das Simulações:")
    print(scenarios_df[['Cenário', 'Δ Taxa (pp)', 'Δ Conversão (pp)', 
                        'Δ Ticket (%)', 'Δ Receita (%)']].to_string(index=False))
    
    # Visualização
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Impacto na Conversão
    ax = axes[0, 0]
    colors = ['red' if x < 0 else 'green' if x > 0 else 'gray' 
              for x in scenarios_df['Δ Conversão (pp)']]
    ax.barh(scenarios_df['Cenário'], scenarios_df['Δ Conversão (pp)'], color=colors)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Variação na Taxa de Conversão (pp)')
    ax.set_title('Impacto na Conversão')
    
    # 2. Impacto no Ticket Médio
    ax = axes[0, 1]
    colors = ['red' if x < 0 else 'green' if x > 0 else 'gray' 
              for x in scenarios_df['Δ Ticket (%)']]
    ax.barh(scenarios_df['Cenário'], scenarios_df['Δ Ticket (%)'], color=colors)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Variação no Ticket Médio (%)')
    ax.set_title('Impacto no Ticket')
    
    # 3. Impacto na Receita Total
    ax = axes[1, 0]
    colors = ['red' if x < 0 else 'green' if x > 0 else 'gray' 
              for x in scenarios_df['Δ Receita (%)']]
    ax.barh(scenarios_df['Cenário'], scenarios_df['Δ Receita (%)'], color=colors)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Variação na Receita Total (%)')
    ax.set_title('Impacto na Receita')
    
    for i, valor in enumerate(scenarios_df['Δ Receita (%)']):
        ax.text(valor, i, f'{valor:.1f}%', va='center', 
               ha='left' if valor >= 0 else 'right')
    
    # 4. Decomposição do Impacto
    ax = axes[1, 1]
    x = range(len(scenarios_df))
    width = 0.35
    
    ax.bar([i - width/2 for i in x], scenarios_df['Δ Conversão (pp)'], 
           width, label='Δ Conversão', alpha=0.7)
    ax.bar([i + width/2 for i in x], scenarios_df['Δ Ticket (%)'], 
           width, label='Δ Ticket', alpha=0.7)
    
    ax.set_xlabel('Cenário')
    ax.set_ylabel('Variação (%/pp)')
    ax.set_title('Decomposição: Conversão vs Ticket')
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios_df['Cenário'], rotation=45, ha='right')
    ax.legend()
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return scenarios_df

# ==========================================
# 11. FUNÇÃO PRINCIPAL - MODELO HURDLE
# ==========================================

def run_hurdle_elasticity_analysis(anomes_treino: List[int],
                                  anomes_teste: List[int],
                                  categorical_vars: Optional[List[str]] = None,
                                  prod_var: Optional[str] = None,
                                  taxa_var: Optional[str] = None,
                                  conversion_var: Optional[str] = None,
                                  include_interactions: bool = True,
                                  validate_vars: bool = True) -> Dict:
    """
    Executa análise completa com Modelo Hurdle
    
    Parameters:
    -----------
    anomes_treino : List[int]
        Períodos de treino
    anomes_teste : List[int]
        Períodos de teste
    categorical_vars : List[str], optional
        Variáveis categóricas
    prod_var : str, optional
        Variável de produção
    taxa_var : str, optional
        Variável de taxa
    conversion_var : str, optional
        Variável de conversão (será criada se não existir)
    include_interactions : bool
        Se inclui interações nos modelos
    validate_vars : bool
        Se valida variáveis categóricas
    
    Returns:
    --------
    Dict: Resultados completos da análise
    """
    
    print("="*60)
    print("🚀 MODELO HURDLE - FINANCIAMENTO DE VEÍCULOS")
    print("="*60)
    print(f"📅 Períodos de Treino: {anomes_treino}")
    print(f"📅 Períodos de Teste: {anomes_teste}")
    
    # Usar variáveis padrão se não especificadas
    if prod_var is None:
        prod_var = DEFAULT_PROD_VAR
    if taxa_var is None:
        taxa_var = DEFAULT_TAXA_VAR
    if conversion_var is None:
        conversion_var = DEFAULT_CONVERSION_VAR
    if categorical_vars is None:
        categorical_vars = DEFAULT_CAT_VARS
    
    print(f"\n📊 Configuração:")
    print(f"   - Variável de produção: {prod_var}")
    print(f"   - Variável de taxa: {taxa_var}")
    print(f"   - Variável de conversão: {conversion_var}")
    print(f"   - Variáveis categóricas: {categorical_vars}")
    
    # 1. Carregar dados
    all_periods = anomes_treino + anomes_teste
    df = load_and_prepare_data(all_periods)
    
    # 2. Validar variáveis categóricas
    if validate_vars:
        cat_vars = validate_categorical_vars(df, categorical_vars)
    else:
        cat_vars = categorical_vars
    
    # 3. Análise exploratória
    perform_hurdle_eda(df, prod_var, taxa_var, conversion_var, cat_vars)
    
    # 4. Preparar dados
    train_data, test_data = prepare_hurdle_data(
        df, prod_var, taxa_var, conversion_var, cat_vars,
        anomes_treino, anomes_teste
    )
    
    # 5. Construir modelo Hurdle
    hurdle_results = build_hurdle_model(
        train_data, test_data, prod_var, taxa_var,
        conversion_var, cat_vars, include_interactions
    )
    
    # 6. Calcular elasticidades
    elasticities = calculate_hurdle_elasticities(
        hurdle_results['conversion_model'],
        hurdle_results['production_model'],
        train_data, taxa_var, cat_vars
    )
    
    # 7. Diagnósticos
    hurdle_model_diagnostics(hurdle_results)
    
    # 8. Simular cenários
    scenarios_df = simulate_hurdle_scenarios(hurdle_results, taxa_var, cat_vars)
    
    print("\n" + "="*60)
    print("✅ ANÁLISE HURDLE CONCLUÍDA COM SUCESSO!")
    print("="*60)
    
    return {
        'hurdle_model': hurdle_results,
        'elasticities': elasticities,
        'scenarios': scenarios_df,
        'categorical_vars_used': cat_vars,
        'train_data': hurdle_results['train_data'],
        'test_data': hurdle_results['test_data']
    }

# ==========================================
# EXEMPLOS DE USO
# ==========================================

if __name__ == "__main__":
    # Definir períodos
    ANOMES_TREINO = [202501, 202502, 202503]
    ANOMES_TESTE = [202504]
    
    # ====== EXEMPLO 1: Configuração Padrão ======
    print("\n🔹 EXEMPLO 1: Modelo Hurdle Padrão")
    results = run_hurdle_elasticity_analysis(
        anomes_treino=ANOMES_TREINO,
        anomes_teste=ANOMES_TESTE
    )
    
    # ====== EXEMPLO 2: Customizar Variáveis ======
    print("\n🔹 EXEMPLO 2: Variáveis Customizadas")
    results = run_hurdle_elasticity_analysis(
        anomes_treino=ANOMES_TREINO,
        anomes_teste=ANOMES_TESTE,
        categorical_vars=['uf', 'seg_cliente'],
        include_interactions=True
    )
    
    # ====== EXEMPLO 3: Modelo Simplificado ======
    print("\n🔹 EXEMPLO 3: Modelo Sem Interações")
    results = run_hurdle_elasticity_analysis(
        anomes_treino=ANOMES_TREINO,
        anomes_teste=ANOMES_TESTE,
        categorical_vars=['rating_price'],
        include_interactions=False
    )
    
    # Acessar resultados:
    # results['hurdle_model'] - modelos de conversão e produção
    # results['elasticities'] - elasticidades calculadas
    # results['scenarios'] - cenários simulados
    # results['train_data'] - dados de treino com previsões
    # results['test_data'] - dados de teste com previsões