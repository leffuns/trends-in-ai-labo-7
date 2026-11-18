"""Maakt de figuren voor reflectie.md met de resultaten van labo7.py"""
import copy
import os
import runpy
import warnings

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs("figuren", exist_ok=True)

# labo7.py uitvoeren en zijn variabelen gebruiken
r = runpy.run_path("labo7.py")
clients, scores, centraal_scores = r["clients"], r["scores"], r["centraal_scores"]
X_test, y_test, globaal, centraal = r["X_test"], r["y_test"], r["globaal"], r["centraal"]
LR = r["LogisticRegression"]

# accuracy van elk lokaal model per ronde (zelfde training als in labo7.py)
client_acc = []
g = None
for ronde in range(10):
    modellen, groottes, accs = [], [], []
    for X, y in clients:
        m = LR(C=100, warm_start=True, max_iter=5) if g is None else copy.deepcopy(g)
        m.fit(X, y)
        modellen.append(m)
        groottes.append(len(y))
        accs.append(accuracy_score(y_test, m.predict(X_test)))
    g = copy.deepcopy(modellen[0])
    g.coef_ = np.average([m.coef_ for m in modellen], axis=0, weights=groottes)
    g.intercept_ = np.average([m.intercept_ for m in modellen], axis=0, weights=groottes)
    client_acc.append(accs)
client_acc = np.array(client_acc)
rondes = range(1, 11)

# 1. data per client
plt.figure(figsize=(7, 4))
namen = [f"Client {i+1}" for i in range(5)]
n_spam = [y.sum() for _, y in clients]
n_ham = [len(y) - y.sum() for _, y in clients]
plt.bar(namen, n_ham, label="ham")
plt.bar(namen, n_spam, bottom=n_ham, label="spam")
plt.ylabel("Aantal berichten")
plt.title("Verdeling van de data over de clients")
plt.legend()
plt.tight_layout()
plt.savefig("figuren/1_data_per_client.png", dpi=150)

# 2. metrieken per ronde
plt.figure(figsize=(7, 4))
for k, naam in enumerate(["accuracy", "precision", "recall"]):
    plt.plot(rondes, [s[k] for s in scores], marker="o", label=naam)
plt.xlabel("Ronde")
plt.ylabel("Score")
plt.ylim(0, 1.05)
plt.title("Globaal model per federated ronde")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("figuren/2_metrieken_per_ronde.png", dpi=150)

# 3. lokale modellen tegenover globaal model
plt.figure(figsize=(7, 4))
for i in range(5):
    plt.plot(rondes, client_acc[:, i], "--", alpha=0.7, label=f"Client {i+1} (lokaal)")
plt.plot(rondes, [s[0] for s in scores], "k-o", linewidth=2.5, label="Globaal (FedAvg)")
plt.xlabel("Ronde")
plt.ylabel("Accuracy op testset")
plt.title("Lokale modellen vs. globaal model")
plt.legend(fontsize=8)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("figuren/3_clients_vs_globaal.png", dpi=150)

# 4. federated vs centraal
plt.figure(figsize=(7, 4))
x = np.arange(3)
plt.bar(x - 0.2, scores[-1], 0.4, label="Federated")
plt.bar(x + 0.2, centraal_scores, 0.4, label="Centraal")
plt.xticks(x, ["accuracy", "precision", "recall"])
plt.ylim(0, 1.05)
plt.title("Federated vs centraal")
plt.legend()
plt.tight_layout()
plt.savefig("figuren/4_federated_vs_centraal.png", dpi=150)

# 5. confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(9, 4))
for ax, (titel, model) in zip(axes, [("Federated", globaal), ("Centraal", centraal)]):
    cm = confusion_matrix(y_test, model.predict(X_test))
    ConfusionMatrixDisplay(cm, display_labels=["ham", "spam"]).plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(titel)
plt.tight_layout()
plt.savefig("figuren/5_confusion_matrix.png", dpi=150)

print("Figuren opgeslagen in de map figuren")
