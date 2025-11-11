## Modeling Pipeline

from econml.dr import DRLearner
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Separate variables
Y = df['conversion'].values
T = df['interest_rate'].values
X = df[['segment', 'income', 'loan_value']]

# Preprocess categorical variables
preprocess = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), ['segment']),
    ('num', 'passthrough', ['income', 'loan_value'])
])

# Base learners
model_y = RandomForestClassifier(n_estimators=200, max_depth=5)
model_t = RandomForestRegressor(n_estimators=200, max_depth=5)
model_final = RandomForestRegressor(n_estimators=200, max_depth=5)

# DR Learner
dr = DRLearner(
    model_regression=model_y,
    model_propensity=model_t,
    model_final=model_final,
    featurizer=preprocess
)

dr.fit(Y, T, X=X)


# Step 4: Estimate the Causal Effect
# Average treatment effect at sample points
tau = dr.effect(X)
print(tau[:10])


import numpy as np
import matplotlib.pyplot as plt

# Predict expected Y for different interest rates
rates = np.linspace(df['interest_rate'].min(), df['interest_rate'].max(), 50)
x_sample = X.iloc[[0]]  # e.g., pick a representative client

preds = [dr.const_marginal_effect(x_sample) * (r - df['interest_rate'].mean()) for r in rates]

plt.plot(rates, preds)
plt.xlabel("Interest Rate")
plt.ylabel("Estimated Effect on Conversion Probability")
plt.title("Causal Price Response Curve")
plt.show()



# Step 5: Interpretability (Optional)

dr.effect_inference(X).summary_frame(alpha=0.05)


from econml.dr import LinearDRLearner
from sklearn.preprocessing import StandardScaler

learner = LinearDRLearner(
    model_regression=LogisticRegression(),
    model_propensity=LinearRegression(),
    featurizer=StandardScaler()
)

learner.fit(Y, T, X=X)
learner.coef_



# Step 1: Understand what the DRLearner gives you


# Step 2: Predict baseline probability of conversion

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from econml.dr import DRLearner

model_y = RandomForestClassifier(n_estimators=200, max_depth=5)
model_t = RandomForestRegressor(n_estimators=200, max_depth=5)
model_final = RandomForestRegressor(n_estimators=200, max_depth=5)

dr = DRLearner(
    model_regression=model_y,
    model_propensity=model_t,
    model_final=model_final
)
dr.fit(Y, T, X=X)


# Predict baseline conversion probability (fitted model)
proba_baseline = dr.model_regression_.predict_proba(np.column_stack([T, X]))[:, 1]


#Step 3: Adjust the probability using the causal effect

delta = dr.effect(X, T0=T, T1=T_new)


import numpy as np

T_new = T - 0.01  # simulate 1pp lower interest rate
delta = dr.effect(X, T0=T, T1=T_new)
proba_causal = np.clip(proba_baseline + delta, 0, 1)

#Now you have a "predict_proba under intervention", i.e.,
#“what the probability of conversion would be if the interest rate changed to T_new.”

# Step 4: Use this to fine-tune pricing

import matplotlib.pyplot as plt

rates = np.linspace(df['interest_rate'].min(), df['interest_rate'].max(), 20)
X_sample = X.iloc[[0]]

probs = []
for r in rates:
    delta = dr.effect(X_sample, T0=np.array([df['interest_rate'].mean()]), T1=np.array([r]))
    p = proba_baseline.mean() + delta  # approximate
    probs.append(p)

plt.plot(rates, probs)
plt.xlabel("Interest rate")
plt.ylabel("Conversion probability (causal-adjusted)")
plt.title("Causal Conversion Curve")
plt.show()


# Step 5: Optional – make it systematic

def predict_proba_causal(dr, X, T_base, T_new):
    """Predict conversion probability if interest rate changes from T_base to T_new."""
    base_pred = dr.model_regression_.predict_proba(np.column_stack([T_base, X]))[:, 1]
    delta = dr.effect(X, T0=T_base, T1=T_new)
    return np.clip(base_pred + delta, 0, 1)

p_new = predict_proba_causal(dr, X, T, T - 0.01)
