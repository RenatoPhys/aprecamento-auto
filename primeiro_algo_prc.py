# Importando dados
# O dataframe df será uma tabela analítica de todos os contratos de financiamento
query = '''
        select *, cast(safra_ajustado) as anomes
        from
            tb_producao_veiculos_partic
        where
            cast(safra_ajustado as int) >= 202503 and cast(safra_ajustado as int)<=202507
            and segmento_pf_pj = 'F'
            and flag2 = 'LEVES'
            and anomesdia = 20250110
        '''

df = wr.athena.read_sql(query)

# Ajustando dados
df['taxa_'] = df['taxa_'].astype(float)
df['valor_prod'] = df['valor_prod'].astype(float)

# Pequena análise exploratória (quantidade de linhas por safra)
print(df['anomes'].value_counts())

# Definindo variáveis da análise
prod_var = 'valor_prod' # valor produzido
taxa_var = 'taxa_' # variavel de taxa
cat_vars = ['uf', 'rating_price', 'seg_cliente'] # quebras da precificação

# Quebrando a variável de taxa em segmentos
df['range_taxa'] = pd.qcut(df[taxa_var], q=5)

# Montando o dataframe consolidado para modelagem
aux = df[[taxa_var] + [prod_var] + cat_vars + ['range_taxa']+
         ['anomes']].groupby(cat_vars+['range_taxa']+['anomes'], 
                             observed=True).agg(
                                 qtd_obs = (prod_var, 'count'),
                                 taxa_ = (taxa_var, 'mean')
                                 producao = (prod_var, 'mean'),
                                 producao_total = (prod_var, 'sum')
                             ).reset_index()


# Modelando produção total - elasticidades
y = 'producao_total'
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

