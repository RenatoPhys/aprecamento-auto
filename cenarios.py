import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product

class SimuladorCenariosMultiSegmento:
    def __init__(self, modelo_elasticidade, colunas_segmento):
        """
        modelo_elasticidade: seu modelo treinado
        colunas_segmento: lista com nomes das colunas de segmentação
                         ex: ['regiao', 'produto', 'canal', 'perfil_risco']
        """
        self.modelo = modelo_elasticidade
        self.colunas_segmento = colunas_segmento
        self.resultados_cache = {}
    
    def preparar_dados_segmento(self, dados, valores_segmentos):
        """
        Prepara dados para um conjunto específico de valores de segmento
        valores_segmentos: dict com valores para cada coluna de segmento
                          ex: {'regiao': 'Sul', 'produto': 'Premium', ...}
        """
        df_filtrado = dados.copy()
        
        for coluna, valor in valores_segmentos.items():
            if coluna in self.colunas_segmento:
                df_filtrado = df_filtrado[df_filtrado[coluna] == valor]
        
        return df_filtrado
    
    def calcular_elasticidade_segmento(self, taxa, **kwargs_segmentos):
        """
        Calcula conversão para uma combinação específica de segmentos
        
        Exemplo de uso:
        calcular_elasticidade_segmento(0.10, regiao='Sul', produto='Premium')
        """
        
        # Criar feature vector para o modelo
        features = pd.DataFrame([{
            'taxa': taxa,
            **kwargs_segmentos
        }])
        
        # Se você tem um modelo sklearn, por exemplo:
        # conversao = self.modelo.predict(features)[0]
        
        # Exemplo simplificado com elasticidades diferentes por segmento
        elasticidade_base = -1.5
        
        # Ajustes por segmento (exemplo)
        ajustes = {
            'regiao': {'Sul': 0.1, 'Norte': -0.1, 'Centro': 0},
            'produto': {'Premium': -0.3, 'Standard': 0.2, 'Basic': 0.4},
            'canal': {'Digital': -0.2, 'Fisico': 0.1, 'Hibrido': 0},
            'perfil_risco': {'Baixo': -0.4, 'Medio': 0, 'Alto': 0.3}
        }
        
        elasticidade_ajustada = elasticidade_base
        
        for segmento, valor in kwargs_segmentos.items():
            if segmento in ajustes and valor in ajustes[segmento]:
                elasticidade_ajustada += ajustes[segmento][valor]
        
        # Cálculo da conversão
        taxa_base = 0.10
        conversao_base = 0.05
        
        variacao_taxa = (taxa - taxa_base) / taxa_base
        variacao_conversao = elasticidade_ajustada * variacao_taxa
        nova_conversao = conversao_base * (1 + variacao_conversao)
        
        return max(0, nova_conversao)
    
    def gerar_cenarios_completos(self, taxas, combinacoes_segmentos):
        """
        Gera cenários para todas as combinações de segmentos especificadas
        
        combinacoes_segmentos: lista de dicts, cada um com uma combinação
        ex: [
            {'regiao': 'Sul', 'produto': 'Premium', 'canal': 'Digital'},
            {'regiao': 'Norte', 'produto': 'Basic', 'canal': 'Fisico'},
        ]
        """
        
        resultados = []
        
        for taxa in taxas:
            for combinacao in combinacoes_segmentos:
                conversao = self.calcular_elasticidade_segmento(taxa, **combinacao)
                
                # Estimar volumes (você pode usar dados históricos aqui)
                volume_base = self.estimar_volume_base(combinacao)
                volume_convertido = volume_base * conversao
                receita = volume_convertido * taxa * 1000
                
                resultado = {
                    'taxa': taxa,
                    **combinacao,  # Desempacotar todas as colunas de segmento
                    'conversao': conversao,
                    'volume_base': volume_base,
                    'volume_convertido': volume_convertido,
                    'receita': receita,
                    'segmento_combinado': '_'.join([str(combinacao[col]) for col in sorted(combinacao.keys())])
                }
                
                resultados.append(resultado)
        
        return pd.DataFrame(resultados)
    
    def estimar_volume_base(self, combinacao):
        """Estima volume base para cada combinação de segmentos"""
        # Exemplo: volumes diferentes por combinação
        volumes_exemplo = {
            ('Sul', 'Premium', 'Digital'): 15000,
            ('Sul', 'Premium', 'Fisico'): 8000,
            ('Norte', 'Basic', 'Digital'): 25000,
            ('Norte', 'Basic', 'Fisico'): 12000,
        }
        
        # Criar chave da combinação
        chave = tuple(combinacao.get(col) for col in ['regiao', 'produto', 'canal'] if col in combinacao)
        
        return volumes_exemplo.get(chave, 10000)  # 10000 como default
    
    def analise_hierarquica(self, df_cenarios, hierarquia=['regiao', 'produto', 'canal']):
        """
        Análise hierárquica dos resultados por níveis de segmentação
        """
        
        analises = {}
        
        # Análise por cada nível
        for i in range(len(hierarquia)):
            nivel_atual = hierarquia[:i+1]
            
            # Agrupar por níveis atuais
            grupo = df_cenarios.groupby(nivel_atual + ['taxa']).agg({
                'conversao': 'mean',
                'volume_convertido': 'sum',
                'receita': 'sum'
            }).reset_index()
            
            analises[f'nivel_{i+1}_{"_".join(nivel_atual)}'] = grupo
        
        return analises
    
    def encontrar_taxas_otimas(self, df_cenarios, colunas_agrupamento=None):
        """
        Encontra taxa ótima para cada combinação de segmentos
        """
        
        if colunas_agrupamento is None:
            colunas_agrupamento = self.colunas_segmento
        
        # Encontrar taxa que maximiza receita para cada combinação
        idx_otimos = df_cenarios.groupby(colunas_agrupamento)['receita'].idxmax()
        cenarios_otimos = df_cenarios.loc[idx_otimos]
        
        # Criar resumo
        resumo = cenarios_otimos[colunas_agrupamento + ['taxa', 'conversao', 'receita']].copy()
        resumo['taxa_percentual'] = (resumo['taxa'] * 100).round(2)
        resumo['conversao_percentual'] = (resumo['conversao'] * 100).round(2)
        
        return resumo.sort_values('receita', ascending=False)
    

class VisualizadorCenarios:
    def __init__(self, df_cenarios, colunas_segmento):
        self.df = df_cenarios
        self.colunas_segmento = colunas_segmento
    
    def plot_elasticidade_por_segmento(self, coluna_principal, coluna_cor=None):
        """
        Plota curvas de elasticidade agrupadas por segmento
        """
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        if coluna_cor:
            # Criar grupos únicos
            grupos = self.df.groupby([coluna_principal, coluna_cor])
            
            colors = plt.cm.tab10(np.linspace(0, 1, self.df[coluna_cor].nunique()))
            color_map = dict(zip(self.df[coluna_cor].unique(), colors))
            
            for (valor_principal, valor_cor), grupo in grupos:
                grupo_taxa = grupo.groupby('taxa').agg({
                    'conversao': 'mean'
                }).reset_index()
                
                ax.plot(grupo_taxa['taxa'] * 100, 
                       grupo_taxa['conversao'] * 100,
                       marker='o',
                       label=f'{valor_principal}-{valor_cor}',
                       color=color_map[valor_cor],
                       linestyle='--' if valor_principal == self.df[coluna_principal].unique()[0] else '-')
        else:
            for valor in self.df[coluna_principal].unique():
                df_seg = self.df[self.df[coluna_principal] == valor]
                grupo_taxa = df_seg.groupby('taxa').agg({
                    'conversao': 'mean'
                }).reset_index()
                
                ax.plot(grupo_taxa['taxa'] * 100, 
                       grupo_taxa['conversao'] * 100,
                       marker='o',
                       label=valor)
        
        ax.set_xlabel('Taxa (%)')
        ax.set_ylabel('Conversão (%)')
        ax.set_title(f'Elasticidade por {coluna_principal}' + 
                    (f' e {coluna_cor}' if coluna_cor else ''))
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def criar_dashboard_interativo(self, df_cenarios):
        """
        Cria um dashboard com múltiplas visualizações
        """
        
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Heatmap de conversão média por dois segmentos principais
        ax1 = fig.add_subplot(gs[0, :2])
        if len(self.colunas_segmento) >= 2:
            pivot = df_cenarios.pivot_table(
                values='conversao',
                index=self.colunas_segmento[0],
                columns=self.colunas_segmento[1],
                aggfunc='mean'
            )
            sns.heatmap(pivot, annot=True, fmt='.2%', cmap='YlOrRd', ax=ax1)
            ax1.set_title(f'Conversão Média: {self.colunas_segmento[0]} vs {self.colunas_segmento[1]}')
        
        # 2. Distribuição de taxas ótimas
        ax2 = fig.add_subplot(gs[0, 2])
        taxas_otimas = self.encontrar_taxas_otimas_simples(df_cenarios)
        ax2.hist(taxas_otimas['taxa'] * 100, bins=15, edgecolor='black')
        ax2.set_xlabel('Taxa Ótima (%)')
        ax2.set_ylabel('Frequência')
        ax2.set_title('Distribuição de Taxas Ótimas')
        
        # 3. Top 10 combinações por receita
        ax3 = fig.add_subplot(gs[1, :])
        top_combinacoes = df_cenarios.nlargest(10, 'receita')
        x_labels = [f"{row['segmento_combinado'][:20]}..." if len(row['segmento_combinado']) > 20 
                   else row['segmento_combinado'] 
                   for _, row in top_combinacoes.iterrows()]
        
        bars = ax3.bar(range(len(top_combinacoes)), top_combinacoes['receita'])
        ax3.set_xticks(range(len(top_combinacoes)))
        ax3.set_xticklabels(x_labels, rotation=45, ha='right')
        ax3.set_ylabel('Receita')
        ax3.set_title('Top 10 Combinações de Segmentos por Receita')
        
        # Adicionar valores nas barras
        for i, (bar, val) in enumerate(zip(bars, top_combinacoes['receita'])):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'${val:,.0f}', ha='center', va='bottom', fontsize=8)
        
        # 4. Análise de sensibilidade
        ax4 = fig.add_subplot(gs[2, 0])
        for seg in df_cenarios[self.colunas_segmento[0]].unique()[:3]:  # Primeiros 3
            df_seg = df_cenarios[df_cenarios[self.colunas_segmento[0]] == seg]
            sensibilidade = df_seg.groupby('taxa').agg({
                'conversao': 'mean'
            }).reset_index()
            
            # Calcular elasticidade ponto a ponto
            elasticidade = np.gradient(sensibilidade['conversao']) / np.gradient(sensibilidade['taxa'])
            ax4.plot(sensibilidade['taxa'] * 100, elasticidade, marker='o', label=seg)
        
        ax4.set_xlabel('Taxa (%)')
        ax4.set_ylabel('Elasticidade')
        ax4.set_title('Análise de Sensibilidade')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Trade-off Volume vs Receita
        ax5 = fig.add_subplot(gs[2, 1])
        scatter = ax5.scatter(df_cenarios['volume_convertido'], 
                            df_cenarios['receita'],
                            c=df_cenarios['taxa'] * 100,
                            cmap='viridis',
                            alpha=0.6)
        ax5.set_xlabel('Volume Convertido')
        ax5.set_ylabel('Receita')
        ax5.set_title('Trade-off: Volume vs Receita')
        plt.colorbar(scatter, ax=ax5, label='Taxa (%)')
        
        # 6. Box plot de conversão por segmento
        ax6 = fig.add_subplot(gs[2, 2])
        if self.colunas_segmento:
            df_melt = df_cenarios.melt(id_vars=['conversao'], 
                                      value_vars=[self.colunas_segmento[0]])
            sns.boxplot(data=df_cenarios, x=self.colunas_segmento[0], 
                       y='conversao', ax=ax6)
            ax6.set_xticklabels(ax6.get_xticklabels(), rotation=45)
            ax6.set_ylabel('Conversão')
            ax6.set_title(f'Distribuição de Conversão por {self.colunas_segmento[0]}')
        
        plt.suptitle('Dashboard de Análise de Cenários', fontsize=16, y=1.02)
        plt.tight_layout()
        plt.show()
    
    def encontrar_taxas_otimas_simples(self, df):
        """Versão simplificada para encontrar taxas ótimas"""
        return df.loc[df.groupby('segmento_combinado')['receita'].idxmax()]
    

def gerar_todas_combinacoes(valores_por_segmento):
    """
    Gera todas as combinações possíveis de segmentos
    
    valores_por_segmento: dict com listas de valores possíveis
    ex: {
        'regiao': ['Sul', 'Norte', 'Centro'],
        'produto': ['Premium', 'Standard', 'Basic'],
        'canal': ['Digital', 'Fisico']
    }
    """
    
    # Obter todas as combinações possíveis
    chaves = valores_por_segmento.keys()
    valores = valores_por_segmento.values()
    
    combinacoes = []
    for combinacao in product(*valores):
        combinacoes.append(dict(zip(chaves, combinacao)))
    
    return combinacoes

# Exemplo de uso
valores_segmentos = {
    'regiao': ['Sul', 'Norte', 'Centro'],
    'produto': ['Premium', 'Standard', 'Basic'],
    'canal': ['Digital', 'Fisico'],
    'perfil_risco': ['Baixo', 'Medio', 'Alto']
}

todas_combinacoes = gerar_todas_combinacoes(valores_segmentos)
print(f"Total de combinações: {len(todas_combinacoes)}")

# Filtrar apenas combinações relevantes (se necessário)
combinacoes_relevantes = [
    comb for comb in todas_combinacoes 
    if not (comb['produto'] == 'Premium' and comb['perfil_risco'] == 'Alto')  # Exemplo de exclusão
]