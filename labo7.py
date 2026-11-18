import copy
import warnings

warnings.filterwarnings("ignore")  # verbergt waarschuwingen over max_iter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score

np.random.seed(42)

# ---------- 1. Data inlezen ----------
df = pd.read_csv("spam.csv", encoding="latin-1")[["v1", "v2"]]
df.columns = ["label", "text"]
df["spam"] = (df["label"] == "spam").astype(int)  # spam = 1, ham = 0

# 20% testset, die gebruiken we nooit om te trainen
train, test = train_test_split(df, test_size=0.2, random_state=42)

# tekst omzetten naar getallen
vec = TfidfVectorizer(max_features=2000)
X_train = vec.fit_transform(train["text"])
y_train = train["spam"].values
X_test = vec.transform(test["text"])
y_test = test["spam"].values

# ---------- 2. Data verdelen over 5 clients ----------
# kans dat een spam-bericht bij client 1, 2, 3, 4 of 5 terechtkomt
kans_spam = [0.40, 0.30, 0.15, 0.10, 0.05]
kans_ham = [0.10, 0.15, 0.20, 0.25, 0.30]

# elk bericht krijgt willekeurig een client (0 t/m 4)
client_van_bericht = []
for y in y_train:
    if y == 1:
        client_van_bericht.append(np.random.choice(5, p=kans_spam))
    else:
        client_van_bericht.append(np.random.choice(5, p=kans_ham))
client_van_bericht = np.array(client_van_bericht)

# per client de berichten bijhouden
clients = []
for i in range(5):
    X_client = X_train[client_van_bericht == i]
    y_client = y_train[client_van_bericht == i]
    clients.append((X_client, y_client))
    print(
        "Client",
        i + 1,
        ":",
        len(y_client),
        "berichten,",
        round(y_client.mean() * 100),
        "% spam",
    )

# ---------- 3. Federated Learning ----------
scores = []  # (accuracy, precision, recall) na elke ronde

for ronde in range(1, 11):
    modellen = []
    groottes = []

    # elke client traint op zijn eigen data
    for X_client, y_client in clients:
        if ronde == 1:
            model = LogisticRegression(
                C=100, warm_start=True, max_iter=5
            )  # eerste ronde: van nul
        else:
            model = copy.deepcopy(globaal)  # daarna: start bij het globale model
        model.fit(X_client, y_client)
        modellen.append(model)
        groottes.append(len(y_client))

    # de server maakt het gemiddelde (client met meer berichten telt zwaarder)
    globaal = copy.deepcopy(modellen[0])
    globaal.coef_ = np.average([m.coef_ for m in modellen], axis=0, weights=groottes)
    globaal.intercept_ = np.average(
        [m.intercept_ for m in modellen], axis=0, weights=groottes
    )

    # testen op de testset
    voorspelling = globaal.predict(X_test)
    acc = accuracy_score(y_test, voorspelling)
    prec = precision_score(y_test, voorspelling)
    rec = recall_score(y_test, voorspelling)
    scores.append((acc, prec, rec))
    print(
        "Ronde",
        ronde,
        "- accuracy:",
        round(acc, 3),
        "precision:",
        round(prec, 3),
        "recall:",
        round(rec, 3),
    )

# ---------- 4. Centraal model (alle data samen, enkel ter vergelijking) ----------
centraal = LogisticRegression(C=100, max_iter=50)
centraal.fit(X_train, y_train)
voorspelling = centraal.predict(X_test)
centraal_scores = (
    accuracy_score(y_test, voorspelling),
    precision_score(y_test, voorspelling),
    recall_score(y_test, voorspelling),
)

print()
print("             Federated  Centraal")
print("accuracy    ", round(scores[-1][0], 3), "    ", round(centraal_scores[0], 3))
print("precision   ", round(scores[-1][1], 3), "    ", round(centraal_scores[1], 3))
print("recall      ", round(scores[-1][2], 3), "    ", round(centraal_scores[2], 3))

# ---------- 5. Grafieken ----------
plt.figure(figsize=(11, 4))

plt.subplot(1, 2, 1)
plt.plot([s[0] for s in scores], label="accuracy")
plt.plot([s[1] for s in scores], label="precision")
plt.plot([s[2] for s in scores], label="recall")
plt.xlabel("Ronde")
plt.title("Federated model per ronde")
plt.legend()

plt.subplot(1, 2, 2)
x = np.arange(3)
plt.bar(x - 0.2, scores[-1], 0.4, label="Federated")
plt.bar(x + 0.2, centraal_scores, 0.4, label="Centraal")
plt.xticks(x, ["accuracy", "precision", "recall"])
plt.title("Federated vs centraal")
plt.legend()

plt.tight_layout()
plt.savefig("resultaat.png")
plt.show()
