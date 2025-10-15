# Modelo de demanda -  produção por simulação

# Importando dados
# O dataframe df será uma tabela analítica de todas as simulações no periodo filtrado
query = '''
        select *
        from
            tb_funil_veiculos
        where
            anomes in (202504)
        '''

df = wr.athena.read_sql(query)

# Ajustando dados
df['pct_txa_ofrt_simu_pmro_vers'] = df['pct_txa_ofrt_simu_pmro_vers'].astype(float)
df['valor_prod'] = df['valor_prod'].astype(float)

# Pequena análise exploratória (quantidade de linhas por safra)
print(df['anomes'].value_counts())

# Definindo variáveis da análise
prod_var = 'valor_prod' # valor produzido
taxa_var = 'pct_txa_ofrt_simu_pmro_vers' # variavel de taxa
cat_vars = ['uf', 'rating_price', 'seg_cliente'] # quebras da precificação

# Quebrando a variável de taxa em segmentos
df['range_taxa'] = pd.qcut(df[taxa_var], q=5)

# Montando o dataframe consolidado para modelagem
aux = df[[taxa_var] + [prod_var] + cat_vars + ['range_taxa']].groupby(cat_vars+['range_taxa'], 
                             observed=True).agg(
                                 qtd_obs = (prod_var, 'count'),
                                 taxa_ = (taxa_var, 'mean')
                                 producao_simulacao = (prod_var, 'mean'), # producao por simulacao
                                 producao_total = (prod_var, 'sum')
                             ).reset_index()


# Modelando produção por simulação - elasticidades
y = 'producao_simulacao'
string_model = f'{y} ~ {taxa_var}'

for var in cat_vars:
    string_model = string_model + f'+ {taxa_var}:C({var})'

# Ajustando variáveis categoricas
for variavel in cat_vars:
    aux[variavel] = aux[variavel].astype('category')

# Fit do modelo
model = smf.ols(formula = string_model, data = aux).fit()

print(model.summary)

# Análise de erro/performance
aux['y_pred'] = model.predict(aux)
aux['y'] = aux['producao_total']
aux['erro'] = aux['y_pred'] - aux['y']

# Cálculo de métricas
mae = mean_absolute_error(aux['y'], aux['y_pred'])
rmse = mean_squared_error(aux['y'], aux['y_pred'])
r2 = r2_score(aux['y'], aux['y_pred'])

# Box-Plot
aux['erro_relativo'] = aux['erro']/aux['y']
sns.boxplot(x=aux['erro_relativo'])


### CENÁRIOS DE PRECIFICAÇÃO
### COMO A MUDANÇA DA TAXA VAI AFETAR A PRODUÇÃO TOTAL POR SEGMENTO

