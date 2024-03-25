# %%
import numpy as np

# %%
def simulate_secretary_problem(N, R):
    candidates = np.random.permutation(N) + 1  # Génère N candidats avec des qualités uniques
    best_so_far = 0
    for i in range(N):
        if i < R:  # Phase de rejet
            continue
        if candidates[i] > best_so_far:  # Accepte le premier meilleur candidat après R
            return candidates[i] == N  # Retourne True si le meilleur candidat est choisi
        best_so_far = max(best_so_far, candidates[i])
    return False  # Dans le cas où aucun candidat n'est choisi

# %%
# Simulation
N = 1000  # Nombre de candidats
R = int(N / np.e)  # Nombre de candidats à rejeter selon la stratégie optimale
simulations = 10000  # Nombre de simulations

successes = sum(simulate_secretary_problem(N, R) for _ in range(simulations))
print(f"Probabilité de choisir le meilleur candidat : {successes / simulations:.4f}")

# %%
