import pandas as pd
import numpy as np
import awswrangler as wr
import statsmodels.formula.api as smf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# =====================================================
# CONFIGURAÇÕES DO MODELO
# =====================================================

# Configurações principais - AJUSTE AQUI CONFORME NECESSÁRIO
CONFIG = {
    # Variáveis do modelo
    'prod_var': 'valor_prod',  # Variável de produção
    'taxa_var': 'taxa_',  # Variável de taxa
    
    # VARIÁVEIS CATEGÓRICAS - MODIFIQUE CONFORME SUA NECESSIDADE
    'cat_vars': ['uf', 'rating_price', 'seg_cliente'],  # Quebras da precificação
    
    # Parâmetros de agrupamento de taxa
    'n_faixas_taxa': 5,  # Número de faixas para quebrar a taxa
    
    # Parâmetros financeiros
    'taxa_captacao': 0.10,  # Taxa de captação anual (10%)
    'taxa_inadimplencia': 0.03,  # Taxa de inadimplência esperada (3%)
    'prazo_medio_default': 48,  # Prazo médio em meses
    
    # Parâmetros de simulação
    'rate_changes_simulation': [-20, -10, -5, 0, 5, 10, 20],  # Mudanças % para simular
    'mfl_target': 15,  # Target de MFL em %
}

# =====================================================
# 1. IMPORTAÇÃO E PREPARAÇÃO DOS DADOS
# =====================================================

def load_and_prepare_data():
    """Carrega e prepara os dados de financiamento"""
    
    query = '''
        SELECT *, CAST(safra_ajustado AS VARCHAR) AS anomes
        FROM tb_producao_veiculos_partic
        WHERE CAST(safra_ajustado AS INT) >= 202503 
            AND CAST(safra_ajustado AS INT) <= 202507
            AND segmento_pf_pj = 'F'
            AND flag2 = 'LEVES'
            AND anomesdia = 20250110
    '''
    
    df = wr.athena.read_sql_query(query, database='default')
    
    # Ajustando tipos de dados com base na configuração
    if CONFIG['taxa_var'] in df.columns:
        df[CONFIG['taxa_var']] = df[CONFIG['taxa_var']].astype(float)
    else:
        print(f"⚠️  Variável de taxa '{CONFIG['taxa_var']}' não encontrada!")
    
    if CONFIG['prod_var'] in df.columns:
        df[CONFIG['prod_var']] = df[CONFIG['prod_var']].astype(float)
    else:
        print(f"⚠️  Variável de produção '{CONFIG['prod_var']}' não encontrada!")
    
    # Criando variáveis adicionais (se não existirem)
    df['prazo_medio'] = df.get('prazo_medio', CONFIG['prazo_medio_default'])
    df['ltv'] = df.get('ltv', 0.8)  # Loan-to-value ratio
    
    # Validar variáveis categóricas
    print("\n📊 Validação das variáveis:")
    print(f"   Variáveis disponíveis no DataFrame: {list(df.columns)[:10]}...")  # Mostra primeiras 10
    
    cat_vars_valid = []
    cat_vars_missing = []
    
    for var in CONFIG['cat_vars']:
        if var in df.columns:
            cat_vars_valid.append(var)
            print(f"   ✅ '{var}' encontrada")
        else:
            cat_vars_missing.append(var)
            print(f"   ❌ '{var}' não encontrada")
    
    if cat_vars_missing:
        print(f"\n   ℹ️  Usando apenas as variáveis válidas: {cat_vars_valid}")
    
    return df

# =====================================================
# 2. ANÁLISE EXPLORATÓRIA
# =====================================================

def exploratory_analysis(df):
    """Realiza análise exploratória dos dados"""
    
    print("=" * 60)
    print("ANÁLISE EXPLORATÓRIA DOS DADOS")
    print("=" * 60)
    
    # Estatísticas básicas
    print("\n1. Distribuição por Safra:")
    print(df['anomes'].value_counts().sort_index())
    
    print("\n2. Estatísticas da Taxa de Juros:")
    print(df[CONFIG['taxa_var']].describe())
    
    print("\n3. Estatísticas da Produção:")
    print(df[CONFIG['prod_var']].describe())
    
    # Visualizações
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Distribuição da taxa
    axes[0, 0].hist(df[CONFIG['taxa_var']], bins=30, edgecolor='black')
    axes[0, 0].set_title('Distribuição das Taxas')
    axes[0, 0].set_xlabel('Taxa (%)')
    
    # Distribuição da produção
    axes[0, 1].hist(df[CONFIG['prod_var']], bins=30, edgecolor='black')
    axes[0, 1].set_title('Distribuição da Produção')
    axes[0, 1].set_xlabel('Valor Produzido (R$)')
    
    # Taxa vs Produção
    axes[1, 0].scatter(df[CONFIG['taxa_var']], df[CONFIG['prod_var']], alpha=0.5)
    axes[1, 0].set_title('Taxa vs Produção')
    axes[1, 0].set_xlabel('Taxa (%)')
    axes[1, 0].set_ylabel('Produção (R$)')
    
    # Produção por categoria (usa primeira variável categórica se existir)
    if CONFIG['cat_vars'] and CONFIG['cat_vars'][0] in df.columns:
        cat_var = CONFIG['cat_vars'][0]
        top_cats = df.groupby(cat_var)[CONFIG['prod_var']].sum().nlargest(10)
        axes[1, 1].bar(range(len(top_cats)), top_cats.values)
        axes[1, 1].set_xticks(range(len(top_cats)))
        axes[1, 1].set_xticklabels(top_cats.index, rotation=45)
        axes[1, 1].set_title(f'Top 10 {cat_var.upper()} por Produção')
    else:
        # Se não houver variáveis categóricas, mostrar distribuição do prazo ou outra métrica
        axes[1, 1].hist(df[CONFIG['prod_var']].apply(np.log), bins=30, edgecolor='black')
        axes[1, 1].set_title('Distribuição Log(Produção)')
        axes[1, 1].set_xlabel('Log(Produção)')
    
    plt.tight_layout()
    plt.show()

# =====================================================
# 3. MODELO DE ELASTICIDADE
# =====================================================

class ElasticityModel:
    """Classe para modelagem de elasticidade preço-demanda"""
    
    def __init__(self, df, config=None):
        """
        Inicializa o modelo de elasticidade
        
        Args:
            df: DataFrame com os dados
            config: Dicionário de configuração (usa CONFIG global se None)
        """
        self.df = df.copy()
        self.config = config or CONFIG
        self.prod_var = self.config['prod_var']
        self.taxa_var = self.config['taxa_var']
        self.cat_vars = self.config['cat_vars']
        self.n_faixas = self.config['n_faixas_taxa']
        self.model = None
        self.elasticities = {}
        
        # Validar se as variáveis existem no DataFrame
        self._validate_variables()
    
    def _validate_variables(self):
        """Valida se as variáveis configuradas existem no DataFrame"""
        missing_vars = []
        
        # Verificar variáveis principais
        if self.prod_var not in self.df.columns:
            missing_vars.append(f"Produção: '{self.prod_var}'")
        if self.taxa_var not in self.df.columns:
            missing_vars.append(f"Taxa: '{self.taxa_var}'")
        
        # Verificar variáveis categóricas
        for var in self.cat_vars:
            if var not in self.df.columns:
                missing_vars.append(f"Categórica: '{var}'")
        
        if missing_vars:
            print("⚠️  AVISO: Variáveis não encontradas no DataFrame:")
            for var in missing_vars:
                print(f"   - {var}")
            print("\nVariáveis disponíveis:", list(self.df.columns))
            print("\nAjuste a configuração CONFIG['cat_vars'] com as variáveis corretas.")
            
            # Usar apenas variáveis que existem
            self.cat_vars = [v for v in self.cat_vars if v in self.df.columns]
            if not self.cat_vars:
                print("   ℹ️  Nenhuma variável categórica válida. Modelo será ajustado sem quebras.")
        
    def prepare_data(self):
        """Prepara dados para modelagem"""
        
        # Criando faixas de taxa
        try:
            if self.n_faixas == 5:
                labels = ['Muito Baixa', 'Baixa', 'Média', 'Alta', 'Muito Alta']
            else:
                labels = [f'Faixa_{i+1}' for i in range(self.n_faixas)]
            
            self.df['range_taxa'] = pd.qcut(self.df[self.taxa_var], q=self.n_faixas, 
                                           labels=labels)
        except:
            # Se não conseguir criar as faixas, usar quartis
            self.df['range_taxa'] = pd.qcut(self.df[self.taxa_var], q=4, 
                                           labels=['Q1', 'Q2', 'Q3', 'Q4'])
        
        # Agregando dados - incluir apenas variáveis categóricas válidas
        if self.cat_vars:
            groupby_vars = self.cat_vars + ['range_taxa', 'anomes']
        else:
            groupby_vars = ['range_taxa', 'anomes']
        
        self.df_model = self.df.groupby(groupby_vars, observed=True).agg(
            qtd_obs=(self.prod_var, 'count'),
            taxa_media=(self.taxa_var, 'mean'),
            producao_media=(self.prod_var, 'mean'),
            producao_total=(self.prod_var, 'sum')
        ).reset_index()
        
        # Log-transformação para modelo log-log (elasticidade constante)
        self.df_model['log_producao'] = np.log(self.df_model['producao_total'] + 1)
        self.df_model['log_taxa'] = np.log(self.df_model['taxa_media'] + 1)
        
        print(f"\n✅ Dados preparados para modelagem:")
        print(f"   - Variável de produção: {self.prod_var}")
        print(f"   - Variável de taxa: {self.taxa_var}")
        print(f"   - Variáveis categóricas utilizadas: {self.cat_vars if self.cat_vars else 'Nenhuma'}")
        print(f"   - Total de observações agregadas: {len(self.df_model)}")
        
        return self.df_model
    
    def fit_model(self, use_log=True):
        """Ajusta modelo de regressão"""
        
        if use_log:
            # Modelo log-log para elasticidade constante
            y_var = 'log_producao'
            x_var = 'log_taxa'
        else:
            # Modelo linear simples
            y_var = 'producao_total'
            x_var = 'taxa_media'
        
        # Construindo fórmula do modelo com interações (apenas se houver cat_vars)
        formula = f'{y_var} ~ {x_var}'
        if self.cat_vars:
            for var in self.cat_vars:
                formula += f' + {x_var}:C({var})'
        
        print(f"\n📊 Fórmula do modelo: {formula}")
        
        # Ajustando modelo
        self.model = smf.ols(formula=formula, data=self.df_model).fit()
        
        print("\n" + "=" * 60)
        print("RESULTADOS DO MODELO DE ELASTICIDADE")
        print("=" * 60)
        print(self.model.summary())
        
        # Calculando elasticidades por segmento
        self.calculate_elasticities()
        
        return self.model
    
    def calculate_elasticities(self):
        """Calcula elasticidades por segmento"""
        
        # Elasticidade base (coeficiente do log_taxa)
        base_elasticity = self.model.params.get('log_taxa', 0)
        
        print("\n" + "=" * 60)
        print("ELASTICIDADES POR SEGMENTO")
        print("=" * 60)
        print(f"\nElasticidade Base: {base_elasticity:.3f}")
        print("(Interpretação: aumento de 1% na taxa reduz produção em {:.1f}%)\n".format(abs(base_elasticity)))
        
        # Elasticidades por categoria (apenas se houver cat_vars)
        if self.cat_vars:
            for var in self.cat_vars:
                print(f"\nElasticidades por {var.upper()}:")
                var_elasticities = {}
                
                # Obtendo categorias únicas
                categories = self.df_model[var].unique()
                
                for cat in categories:
                    # Procurando coeficiente de interação
                    interaction_term = f'log_taxa:C({var})[T.{cat}]'
                    interaction_coef = self.model.params.get(interaction_term, 0)
                    
                    # Elasticidade total para esta categoria
                    total_elasticity = base_elasticity + interaction_coef
                    var_elasticities[cat] = total_elasticity
                    
                    print(f"  {cat}: {total_elasticity:.3f}")
                
                self.elasticities[var] = var_elasticities
        else:
            print("ℹ️  Modelo sem variáveis categóricas - apenas elasticidade base aplicada.")
    
    def evaluate_model(self):
        """Avalia performance do modelo"""
        
        # Previsões
        self.df_model['y_pred'] = self.model.predict(self.df_model)
        
        # Métricas (em escala log se modelo log-log)
        y_true = self.df_model['log_producao'] if 'log_producao' in self.df_model.columns else self.df_model['producao_total']
        y_pred = self.df_model['y_pred']
        
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        print("\n" + "=" * 60)
        print("MÉTRICAS DE PERFORMANCE")
        print("=" * 60)
        print(f"MAE: {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"R²: {r2:.3f}")
        
        # Análise de resíduos
        self.df_model['residuo'] = y_pred - y_true
        self.df_model['residuo_percentual'] = (self.df_model['residuo'] / y_true) * 100
        
        # Visualizações
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Valores reais vs preditos
        axes[0, 0].scatter(y_true, y_pred, alpha=0.5)
        axes[0, 0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
        axes[0, 0].set_xlabel('Valor Real')
        axes[0, 0].set_ylabel('Valor Predito')
        axes[0, 0].set_title('Real vs Predito')
        
        # Distribuição dos resíduos
        axes[0, 1].hist(self.df_model['residuo'], bins=30, edgecolor='black')
        axes[0, 1].set_xlabel('Resíduo')
        axes[0, 1].set_title('Distribuição dos Resíduos')
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(self.df_model['residuo'], dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot')
        
        # Resíduos por categoria (exemplo: primeira variável categórica ou UF)
        if self.cat_vars:
            var_to_plot = self.cat_vars[0]  # Usa a primeira variável categórica
            top_cats = self.df_model.groupby(var_to_plot)['producao_total'].sum().nlargest(5).index
            df_top = self.df_model[self.df_model[var_to_plot].isin(top_cats)]
            axes[1, 1].boxplot([df_top[df_top[var_to_plot] == cat]['residuo_percentual'].values 
                               for cat in top_cats])
            axes[1, 1].set_xticklabels(top_cats, rotation=45)
            axes[1, 1].set_ylabel('Resíduo Percentual (%)')
            axes[1, 1].set_title(f'Resíduos por {var_to_plot.upper()} (Top 5)')
        else:
            # Se não houver variáveis categóricas, mostrar histograma de resíduos percentuais
            axes[1, 1].hist(self.df_model['residuo_percentual'], bins=20, edgecolor='black')
            axes[1, 1].set_xlabel('Resíduo Percentual (%)')
            axes[1, 1].set_title('Distribuição dos Resíduos Percentuais')
        
        plt.tight_layout()
        plt.show()

# =====================================================
# 4. SIMULAÇÃO DE CENÁRIOS
# =====================================================

class PricingScenarioSimulator:
    """Simulador de cenários de precificação"""
    
    def __init__(self, elasticity_model, config=None):
        self.model = elasticity_model
        self.config = config or CONFIG
        self.scenarios = {}
        
    def simulate_rate_change(self, rate_change_pct, segment_filters=None):
        """
        Simula mudança na taxa e impacto na produção
        
        Args:
            rate_change_pct: Mudança percentual na taxa (ex: 10 para +10%)
            segment_filters: Dict com filtros de segmento (ex: {'uf': 'SP'})
        """
        
        df_sim = self.model.df_model.copy()
        
        # Aplicar filtros se especificados
        if segment_filters:
            for key, value in segment_filters.items():
                if key in df_sim.columns:
                    df_sim = df_sim[df_sim[key] == value]
        
        # Produção atual
        producao_atual = df_sim['producao_total'].sum()
        
        # Estimar nova produção usando elasticidade
        if segment_filters and len(segment_filters) == 1:
            # Usar elasticidade específica do segmento
            var, val = list(segment_filters.items())[0]
            elasticity = self.model.elasticities.get(var, {}).get(val, -1.5)
        else:
            # Usar elasticidade média
            elasticity = self.model.model.params.get('log_taxa', -1.5)
        
        # Cálculo do impacto
        producao_change_pct = elasticity * rate_change_pct
        producao_nova = producao_atual * (1 + producao_change_pct / 100)
        
        resultado = {
            'taxa_change_pct': rate_change_pct,
            'elasticidade': elasticity,
            'producao_atual': producao_atual,
            'producao_nova': producao_nova,
            'producao_change_pct': producao_change_pct,
            'producao_change_abs': producao_nova - producao_atual,
            'segmento': segment_filters or 'Total'
        }
        
        return resultado
    
    def simulate_multiple_scenarios(self):
        """Simula múltiplos cenários de precificação"""
        
        print("\n" + "=" * 60)
        print("SIMULAÇÃO DE CENÁRIOS DE PRECIFICAÇÃO")
        print("=" * 60)
        
        # Cenários de mudança na taxa (da configuração)
        rate_changes = self.config['rate_changes_simulation']
        
        # Simulação geral
        print("\n1. CENÁRIO GERAL (TODOS OS SEGMENTOS)")
        print("-" * 40)
        
        results_general = []
        for change in rate_changes:
            result = self.simulate_rate_change(change)
            results_general.append(result)
            
            print(f"Taxa {change:+.0f}%: Produção {result['producao_change_pct']:+.1f}% "
                  f"(R$ {result['producao_change_abs']/1e6:+.1f}M)")
        
        # Simulação por categoria (se houver)
        if self.model.cat_vars:
            # Usar a primeira variável categórica para exemplo
            first_cat = self.model.cat_vars[0]
            top_categories = self.model.df_model.groupby(first_cat)['producao_total'].sum().nlargest(3).index
            
            print(f"\n2. CENÁRIOS POR {first_cat.upper()}")
            print("-" * 40)
            
            for cat in top_categories:
                print(f"\n{first_cat}: {cat}")
                for change in [-10, 0, 10]:
                    result = self.simulate_rate_change(change, {first_cat: cat})
                    print(f"  Taxa {change:+.0f}%: Produção {result['producao_change_pct']:+.1f}%")
        
        # Visualização dos cenários
        self.plot_scenarios(results_general)
        
        return results_general
    
    def plot_scenarios(self, results):
        """Visualiza cenários de precificação"""
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Extrair dados
        rate_changes = [r['taxa_change_pct'] for r in results]
        prod_changes_pct = [r['producao_change_pct'] for r in results]
        prod_values = [r['producao_nova']/1e6 for r in results]
        
        # Gráfico 1: Mudança percentual
        axes[0].plot(rate_changes, prod_changes_pct, 'b-o')
        axes[0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[0].axvline(x=0, color='r', linestyle='--', alpha=0.5)
        axes[0].set_xlabel('Mudança na Taxa (%)')
        axes[0].set_ylabel('Mudança na Produção (%)')
        axes[0].set_title('Elasticidade: Taxa vs Produção')
        axes[0].grid(True, alpha=0.3)
        
        # Gráfico 2: Valores absolutos
        axes[1].bar(range(len(rate_changes)), prod_values, 
                   color=['red' if x < 0 else 'green' for x in prod_changes_pct])
        axes[1].set_xticks(range(len(rate_changes)))
        axes[1].set_xticklabels([f'{x:+.0f}%' for x in rate_changes])
        axes[1].set_xlabel('Mudança na Taxa')
        axes[1].set_ylabel('Produção Total (R$ Milhões)')
        axes[1].set_title('Produção Total por Cenário')
        
        # Adicionar linha de produção atual
        prod_atual = results[rate_changes.index(0)]['producao_atual']/1e6
        axes[1].axhline(y=prod_atual, color='blue', linestyle='--', 
                       label=f'Produção Atual: R$ {prod_atual:.1f}M')
        axes[1].legend()
        
        plt.tight_layout()
        plt.show()

# =====================================================
# 5. CALCULADORA MFL (MARGEM FINANCEIRA LÍQUIDA)
# =====================================================

class MFLCalculator:
    """Calculadora de Margem Financeira Líquida"""
    
    def __init__(self, config=None):
        """
        Inicializa calculadora de MFL
        
        Args:
            config: Dicionário de configuração (usa CONFIG global se None)
        """
        self.config = config or CONFIG
        self.taxa_captacao = self.config['taxa_captacao']
        self.taxa_inadimplencia = self.config['taxa_inadimplencia']
        self.prazo_medio_default = self.config['prazo_medio_default']
        
    def calculate_mfl(self, producao, taxa_media, prazo_medio=None):
        """
        Calcula a MFL baseada na produção e parâmetros
        
        Args:
            producao: Volume de produção (R$)
            taxa_media: Taxa média praticada (anual)
            prazo_medio: Prazo médio dos contratos (meses)
        """
        
        if prazo_medio is None:
            prazo_medio = self.prazo_medio_default
        
        # Spread bruto
        spread_bruto = taxa_media - self.taxa_captacao
        
        # Receita de juros
        receita_juros = producao * taxa_media * (prazo_medio / 12)
        
        # Custo de captação
        custo_captacao = producao * self.taxa_captacao * (prazo_medio / 12)
        
        # Perda com inadimplência
        perda_inadimplencia = producao * self.taxa_inadimplencia
        
        # MFL
        mfl_total = receita_juros - custo_captacao - perda_inadimplencia
        mfl_percentual = (mfl_total / producao) * 100
        
        resultado = {
            'producao': producao,
            'taxa_media': taxa_media,
            'spread_bruto': spread_bruto,
            'receita_juros': receita_juros,
            'custo_captacao': custo_captacao,
            'perda_inadimplencia': perda_inadimplencia,
            'mfl_total': mfl_total,
            'mfl_percentual': mfl_percentual,
            'prazo_medio': prazo_medio
        }
        
        return resultado
    
    def optimize_pricing(self, elasticity_model, target_mfl_pct=None):
        """
        Otimiza precificação para atingir MFL target
        """
        
        if target_mfl_pct is None:
            target_mfl_pct = self.config['mfl_target']
        
        print("\n" + "=" * 60)
        print("OTIMIZAÇÃO DE PRECIFICAÇÃO PARA MFL TARGET")
        print("=" * 60)
        
        # Taxa atual média
        taxa_atual = elasticity_model.df_model['taxa_media'].mean()
        producao_atual = elasticity_model.df_model['producao_total'].sum()
        
        # Testar diferentes taxas
        best_result = None
        best_mfl = -np.inf
        
        results = []
        for taxa_change in range(-30, 31, 5):
            nova_taxa = taxa_atual * (1 + taxa_change/100)
            
            # Simular nova produção
            elasticity = elasticity_model.model.params.get('log_taxa', -1.5)
            nova_producao = producao_atual * (1 + elasticity * taxa_change/100)
            
            # Calcular MFL
            mfl_result = self.calculate_mfl(nova_producao, nova_taxa)
            mfl_result['taxa_change_pct'] = taxa_change
            
            results.append(mfl_result)
            
            if mfl_result['mfl_total'] > best_mfl:
                best_mfl = mfl_result['mfl_total']
                best_result = mfl_result
        
        print(f"\nTaxa Atual: {taxa_atual:.2%}")
        print(f"Produção Atual: R$ {producao_atual/1e6:.1f}M")
        print(f"\nMelhor Cenário:")
        print(f"  Mudança na Taxa: {best_result['taxa_change_pct']:+.0f}%")
        print(f"  Nova Taxa: {best_result['taxa_media']:.2%}")
        print(f"  Nova Produção: R$ {best_result['producao']/1e6:.1f}M")
        print(f"  MFL Total: R$ {best_result['mfl_total']/1e6:.1f}M")
        print(f"  MFL %: {best_result['mfl_percentual']:.1f}%")
        
        # Plotar resultados
        self.plot_mfl_optimization(results)
        
        return best_result, results
    
    def plot_mfl_optimization(self, results):
        """Visualiza otimização de MFL"""
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        taxa_changes = [r['taxa_change_pct'] for r in results]
        producao_values = [r['producao']/1e6 for r in results]
        mfl_total = [r['mfl_total']/1e6 for r in results]
        mfl_pct = [r['mfl_percentual'] for r in results]
        
        # Produção vs Taxa
        axes[0].plot(taxa_changes, producao_values, 'b-o')
        axes[0].set_xlabel('Mudança na Taxa (%)')
        axes[0].set_ylabel('Produção (R$ M)')
        axes[0].set_title('Produção vs Taxa')
        axes[0].grid(True, alpha=0.3)
        
        # MFL Total vs Taxa
        axes[1].plot(taxa_changes, mfl_total, 'g-o')
        axes[1].set_xlabel('Mudança na Taxa (%)')
        axes[1].set_ylabel('MFL Total (R$ M)')
        axes[1].set_title('MFL Total vs Taxa')
        axes[1].grid(True, alpha=0.3)
        
        # MFL % vs Taxa
        axes[2].plot(taxa_changes, mfl_pct, 'r-o')
        axes[2].set_xlabel('Mudança na Taxa (%)')
        axes[2].set_ylabel('MFL (%)')
        axes[2].set_title('MFL Percentual vs Taxa')
        axes[2].grid(True, alpha=0.3)
        axes[2].axhline(y=self.config['mfl_target'], color='green', linestyle='--', 
                       label=f'Target: {self.config["mfl_target"]}%')
        axes[2].legend()
        
        plt.tight_layout()
        plt.show()

# =====================================================
# 6. PIPELINE PRINCIPAL
# =====================================================

def main():
    """Pipeline principal de execução"""
    
    print("\n" + "=" * 60)
    print("MODELO DE ELASTICIDADE E PRECIFICAÇÃO")
    print("FINANCIAMENTO DE VEÍCULOS")
    print("=" * 60)
    
    # Mostrar configurações atuais
    print("\n📋 CONFIGURAÇÕES ATUAIS:")
    print("-" * 40)
    print(f"Variável de Produção: {CONFIG['prod_var']}")
    print(f"Variável de Taxa: {CONFIG['taxa_var']}")
    print(f"Variáveis Categóricas: {CONFIG['cat_vars']}")
    print(f"Taxa de Captação: {CONFIG['taxa_captacao']*100:.1f}%")
    print(f"Taxa de Inadimplência: {CONFIG['taxa_inadimplencia']*100:.1f}%")
    print(f"Prazo Médio: {CONFIG['prazo_medio_default']} meses")
    print("\n💡 Dica: Ajuste as variáveis no dicionário CONFIG no início do código")
    print("   Exemplo: CONFIG['cat_vars'] = ['uf', 'novo_campo', 'outro_campo']")
    
    # 1. Carregar dados
    print("\n[1] Carregando dados...")
    df = load_and_prepare_data()
    print(f"    Total de registros: {len(df):,}")
    
    # 2. Análise exploratória
    print("\n[2] Realizando análise exploratória...")
    exploratory_analysis(df)
    
    # 3. Modelo de elasticidade
    print("\n[3] Construindo modelo de elasticidade...")
    elasticity_model = ElasticityModel(df, config=CONFIG)
    df_model = elasticity_model.prepare_data()
    model = elasticity_model.fit_model(use_log=True)
    
    # 4. Avaliação do modelo
    print("\n[4] Avaliando performance do modelo...")
    elasticity_model.evaluate_model()
    
    # 5. Simulação de cenários
    print("\n[5] Simulando cenários de precificação...")
    simulator = PricingScenarioSimulator(elasticity_model, config=CONFIG)
    scenarios = simulator.simulate_multiple_scenarios()
    
    # 6. Calculadora MFL
    print("\n[6] Otimizando MFL...")
    mfl_calculator = MFLCalculator(config=CONFIG)
    best_scenario, all_scenarios = mfl_calculator.optimize_pricing(elasticity_model)
    
    # 7. Recomendações finais
    print("\n" + "=" * 60)
    print("RECOMENDAÇÕES ESTRATÉGICAS")
    print("=" * 60)
    
    print("\n1. ELASTICIDADE:")
    base_elasticity = model.params.get('log_taxa', -1.5)
    print(f"   - Elasticidade média: {base_elasticity:.2f}")
    print(f"   - Interpretação: Aumento de 10% na taxa reduz produção em {abs(base_elasticity*10):.1f}%")
    
    print("\n2. SEGMENTAÇÃO:")
    if elasticity_model.cat_vars:
        print("   - Considerar elasticidades diferentes por segmento")
        print("   - Foco em segmentos menos elásticos para aumento de taxa")
        print("   - Redução seletiva em segmentos mais elásticos")
        print(f"   - Variáveis utilizadas: {', '.join(elasticity_model.cat_vars)}")
    else:
        print("   - Nenhuma variável categórica foi utilizada")
        print("   - Considere adicionar segmentações em CONFIG['cat_vars']")
    
    print("\n3. OTIMIZAÇÃO:")
    print(f"   - Taxa ótima para MFL: {best_scenario['taxa_media']:.2%}")
    print(f"   - MFL esperada: R$ {best_scenario['mfl_total']/1e6:.1f}M ({best_scenario['mfl_percentual']:.1f}%)")
    
    print("\n4. PRÓXIMOS PASSOS:")
    print("   - Incorporar sazonalidade nos modelos")
    print("   - Adicionar variáveis macroeconômicas")
    print("   - Modelar inadimplência por segmento")
    print("   - Implementar otimização multiobjetivo (volume vs margem)")
    print("   - Testar diferentes variáveis categóricas em CONFIG['cat_vars']")
    
    return df, elasticity_model, simulator, mfl_calculator

# =====================================================
# EXEMPLOS DE USO E CUSTOMIZAÇÃO
# =====================================================

def update_config(**kwargs):
    """
    Função auxiliar para atualizar configurações facilmente
    
    Exemplos:
        update_config(cat_vars=['uf', 'tipo_veiculo'])
        update_config(taxa_captacao=0.12, taxa_inadimplencia=0.05)
    """
    for key, value in kwargs.items():
        if key in CONFIG:
            CONFIG[key] = value
            print(f"✅ Configuração atualizada: {key} = {value}")
        else:
            print(f"❌ Configuração '{key}' não reconhecida")
    
    return CONFIG

# Exemplo 1: Alterar variáveis categóricas
# update_config(cat_vars=['uf', 'tipo_veiculo', 'canal_venda'])

# Exemplo 2: Ajustar parâmetros financeiros
# update_config(taxa_captacao=0.12, taxa_inadimplencia=0.05)

# Exemplo 3: Modificar cenários de simulação
# update_config(rate_changes_simulation=[-30, -20, -10, 0, 10, 20, 30, 40])

# Exemplo 4: Usar apenas uma variável categórica
# update_config(cat_vars=['uf'])  # Apenas por estado

# Exemplo 5: Não usar variáveis categóricas (modelo simples)
# update_config(cat_vars=[])  # Modelo sem quebras

# Exemplo 6: Múltiplas configurações de uma vez
# update_config(
#     cat_vars=['uf', 'rating_price'],
#     taxa_captacao=0.11,
#     mfl_target=18,
#     n_faixas_taxa=4
# )

# Executar pipeline
if __name__ == "__main__":
    # Para customizar o modelo antes de executar, use:
    # update_config(cat_vars=['sua_var1', 'sua_var2', 'sua_var3'])
    # update_config(taxa_captacao=0.12, taxa_inadimplencia=0.04)
    
    # Executar o modelo
    df, elasticity_model, simulator, mfl_calculator = main()
    
    # Após executar, você pode fazer análises adicionais:
    # - elasticity_model.elasticities  # Ver elasticidades por segmento
    # - simulator.simulate_rate_change(15, {'uf': 'SP'})  # Simular cenário específico
    # - mfl_calculator.calculate_mfl(1000000, 0.18)  # Calcular MFL para valores específicos