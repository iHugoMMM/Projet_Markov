# %% CONTEXTE
"""On pose le problème de la "secretaire".
Un emloyeur cherche à employer quelqu'un pour un poste.
Il y a N candidats qui se présentent à l'entretien avec N fixé et connnu par l'employeur.
Les candidats sont interviewés l'un après l'autre.
Après chaque entretien, l'employeur decide si oui ou non on garde ce candidat. 
Si on ne garde pas ce candidat, celui-là ne peut plus se presenter.

Ici, on étudie le cas où l'employeur cherche à maximiser la probabilité de donner une offre
au meilleur candidat.

Formulation formelle du problème : 
    Une collection de N objets est rangée de 1 à N, avec le rang 1 étant le plus desirable.
Les vrais rangs sont inconnus pour la personne qui prend la décision.
La personne observe les objets un par un de manière aléatoire. 
Le personne peut, soit selectionner l'objet courant et finir la recherche
                  soit le rejeter et continuer la recherche avec l'objet suivant. 
L'objectif est de maximiser la probabilité de choisir l'objet avec rang 1.

On suppose que la décision relative des rangs sont consistent avec les vrai rangs,
ie. si A est inférieur numériquement en rang à B, alors la personne préfère A à B.
    Les décisions sont faites par observation de chaque objet, alors l'horizon N est le nombre d'objets.
L'espace d'état S' = {0,1}, 1 signifie que l'objet actuel est le meilleur (rang plus proche de 1), 0 si l'objet
d'avant est meilleur.
    Dans chaque état, l'action Q signifie selectionner l'objet actuel (donner une offre au candidat actuel),
et C signifie ne pas selectionner l'objet actuel et continuer à chercher (rejeter le candidat actuel et passer au suivant).
Les récompenses sont données une fois qu'on a stoppé, càd en choississant l'option Q. 
Notation: le coût de continuer f_{t}(s) = 0, s = 0,1 
          les recompenses quand on arrête, g_{t}(0) = 0
                                           g_{t}(1) = t/N
          la recompense terminal, h(0) = 0 et h(1) = 1 

g_{t} et h peuvent s'expliquer comme suit:
On suppose que, après avoir observer t objets, que la personne classe les objets présents 
comme le meilleur des ceux actuellement observés. 
Alors g_{t}(1), la probabilité que cet objet et meilleur que tous les autres, est determiner comme suit:
P({Le meilleur objet est en premier rang}) = (Nombre de sous parties {1,...,N} de taille t contenant 1) / (Nombre de sous parties de taille t)
                                           = (t-1 parmi N-1) / (t parmi N)
                                           = t/N

Si tous les objets one été observés, le dernier objet doit être choisi. Si l'objet est le meilleur,
alors s = 1, alors la probabilité de choisir le meilleur objet est 1, d'où h(1) = 1, sinon h(0) = 0.
    
    Les transitions de probabilités pour le système non contrôlé sont indépendentes de l'état du système, ie.
l'état de l'objet courant. Alors P_{t}(j|s) = P_{t}(j) pour s = {0,1}.
La probabilité que le subsequest (subsequest en anglais signifie qui suit) objet est le meilleur
parmis les premiers t+1, P_{t}(1|s) = 1/(t+1)
et la probabilité que le subsequest objet   n'est pas le meilleur  
parmis les premiers t+1, P_{t}(0|s) = t/(t+1)
"""

# %% CONTEXTE SOLUTION
"""On va maintenant résoudre le problème de la secretaire.
On a N candidats disponibles, N fixé et connu.
On note u_{t}^{*}(1) la probabilité maximale de choisir le meilleur candidat,
quand la candidat actuel a la plus grande probabilité relative parmi les premiers t
interviewé
        u_{t}^{*}(0) la probabilité maximale de choisir le meilleur candidat
overall si le candidat actuel n'a pas le rang maximal relative parmi les premiers
t interviewé.
On note delta l'état d'arrêt.

Alors u_{t}^{*} satisfait la relation de recurence suivante:
u_{N}^{*}(1) = 1
u_{N}^{*}(0) = h(0) = 0
u_{N}^{*}(delta) = 0

et pour t < N,
u_{t}^{*}(1) = max{g_{t}(1) + u_{t+1}^{*}(delta), -f_{t}(0) + p_{t}(1|1)u_{t+1}^{*}(1) + p_{t}(0|1)u_{t+1}^{*}(0)}
             = max{t/N, 1/(t+1)u_{t+1}^{*}(1) + t/(t+1)u_{t+1}^{*}(0)}

u_{t}^{*}(0) = max{g_{t}(0) + u_{t+1}^{*}(delta), -f_{t}(0) + p_{t}(1|0)u_{t+1}^{*}(1) + p_{t}(0|0)u_{t+1}^{*}(0)}
             = max{0, 1/(t+1)u_{t+1}^{*}(1) + t/(t+1)u_{t+1}^{*}(0)}    

et 

u_{t}^{*}(delta) = u_{t+1}^{*}(delta) = 0

On remarque que u_{t}^{*} >= 0 permet une simplification de la relation de recurence. Ainsi on a :
u_{t}^{*}(0) = 1/(t+1)u_{t+1}^{*}(1) + t/(t+1)u_{t+1}^{*}(0)   (4.6.6)
u_{t}^{*}(1) = max{t/N, u_{t}^{*}(0)}                          (4.6.7)

Les solutions (4.6.6) et (4.6.7) donne une solution optimal pour le problème.
Dans l'état 1 quand t/N > u_{t}^{*}(0), la solution optimal est d'arrêter. 
              quand t/N < u_{t}^{*}(0), l'action optimal est de continuer 
              quand t/N = u_{t}^{*}(0), les deux actions sont optimales.
Dans l'état 0, l'action optimal est de continuer.

Si on applique l'induction, ceci sugère que la solution optimal est de la forme :
"Observer les premiers tau candidats, après cela, selectionner le premier candidat qui est meilleur que tous les précédents."
Ceci se traduit par la stratégie:
pi^{*} = (d_{1}^{*}, d_{2}^{*},...,d_{N-1}^{*}) avec d_{t}^{*}(0) = C (ne pas selectionner le candidat t), 
d_{t}^{*}(1) = C si t <= tau
d_{t}^{*}(1) = Q (selectionner) si t > tau

Maintenant, on demontre que la stratégie optimal a cette forme en démontrant que si:
u_{tau}^{*}(1) > tau/N ou u_{tau}^{*}(1) = tau/N = u_{tau}^{*}(0) pour un certain tau, où c'est optimal de continuer,
alors u_{t}^{*}(1) > t/N pour t < tau pour que c'est optimal de continuer.

On suppose soit u_{tau}^{*}(1) > tau/N pour que de (4.6.7), u_{tau}^{*}(1) = u_{tau}^{*}(0) ou u_{tau}^{*}(1) = tau/N = u_{tau}^{*}(0) pour que:
u_{tau-1}^{*}(0) = (1/tau)u_{tau}^{*}(1) + (tau-1)/(tau) u_{tau}^{*}(0) = u_{tau}^{*}(0) >= tau/N
Thus
u_{tau}^{*}(1) = max{(tau-1)/(N), u_{tau-1}^{*}(0)} >= tau/N >= (tau-1)/N

En repetant cet argument avec tau-1 en remplaçant tau et ainsi de suite, on demontre que le résultat est vrai pour chaque t < tau.
Ceci implique que une stratégie optimal ne peut pas avoir la forme:
d_{t}^{*}(1) = C pour t <= t'
d_{t}^{*}(1) = Q pour t' < t <= t'' et
d_{t}^{*}(1) = C pour t > t''

Par définition, tau < N. Maintenant on montre que, si N > 2, tau >= 1. 
Alors pour tout t, u_{t}^{*}(1) = t/N pour que, par (4.6.6):
u_{t}^{*}(0) = (1/(t+1))((t+1)/N) + (t/(t+1))u_{t+1}^{*}(0) = 1/N + (t/(t+1))u_{t+1}^{*}(0)    (4.6.8)
En remarquant que u_{N}^{*}(0) = 0 et on résolvant par recurrence (4.6.8 inversément), on obtient:
u_{t}^{*}(0) = (t/N)(1/t + 1/(t+1) + ... + 1/(N-1)) pour tout 1 <= t < N       (4.6.9)
pour N>2.

Ceci implique que u_{1}^{*}(0) > 1/N = u_{1}^{*}(1) >= u_{1}^{*}(0). La deuxième ingégalité c'est une conséquence de (4.6.7).
Ceci est une contradiction, alors on inclue tau >= 1 quand N > 2
Alors, quand N > 2:
u_{t}^{*}(0) = u_{1}^{*}(1) = ... = u_{tau}^{*}(0) = u_{tau}^{*}(1)    (4.6.10)

u_{t}^{*}(1) = t/N pour t > tau et:
u_{t}^{*}(0) = t/N(1/t + 1/(t+1) + ... + 1/(N-1)) pour t > tau

Alors c'est optimal de continuer d'observer des candidats tant que u_{t}^{*}(0) > t/N, pour que:
tau = max_{t>1} {[1/t + 1/(t+1) + ... + 1/(N-1)] > 1}      (4.6.11)

Quand N <= 2, on choisit tau = 0, et dans ce cas, n'importe quelle stratégie est optimal.

Par exemple, on suppose que N = 4, vu que 1 + 1/2 + 1/3 > 1 et 1/2 + 1/3 < 1, tau = 1
Ceci signifie que, avec 4 candidats, la décision à prendre est d'observer le premier candidat et après choisir le candidat suivant 
qui a un rang relative de 1 
Figure 4.6.2 montre comment tau/N and u_{1}^{*}(1), la probabilité de choisir le meilleur candidat, varie en fonction de N.
Ceci sugère que, avec N grand, tau/N et u_{1}^{*}(1) approche les limites. 

Notons tau(N) la valeur optimal de tau avec N candidats. 

Quelques résultats : 
tau(1000) = 368, 
tau(10000) = 3679
tau(100000) = 36788

Ainsi, pour N suffisament grande:
1 environ= [1/tau(N) + 1/(tau(N) + 1) + ... + 1/(N-1)]/N environ = int_{tau(N)} ^{N} 1/x dx    (4.6.12)

alors pour N grand:
log(N/tau(N)) environ= 1

et donc lim_{N->infini} tau(N)/N = 1/e

et comme conséquence de (4.6.10), (4.6.11) et (4.6.12)
u_{1}^{*}(0) = u_{1}^{*}(1) = u_{tau(N)}^{*}(0) = tau(N)/N -> 1/e

Alors avec un grand nombre de candidats, l'observateur doit oberserver environ N/e candidats
ou 36.8% des candidats et subsequently choisir le premier candidat avec le plus haut rang relative.
La probabilité de choisir le meilleur candidat avec cette stratégie est environ 1/e = 0.368
D'où avec N = 100 000 
"""

# %% IMPORTS
import random
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

# %% PROCEDURE PAR MONTE CARLO
BEST_SCORE = 100
"""Les rangs des candidats sont dans les dans [1,100] et le meilleur candidat
est celui avec rang 100.
3 approches pour résoudre ce problème."""

# %%  1ER APPROCHE TEST D'ARRET
def check_secretary_list(secretaries, cutoff):
    best_test = np.max(secretaries[:cutoff])  # Best candidate in the 'learning set', i.e. before the cutoff
    if best_test == BEST_SCORE:
        return False  # If the best candidate is interviewed before the cutoff, there is no point to keep going
    # Now we must find a candidate that's better than all the candidates in the learning set
    for secretary in secretaries[cutoff:]:   
        if secretary < best_test:
            continue
        elif secretary == BEST_SCORE:  # True is returned only if the best candidate is met
            return True
        return False 

# %% SUCCESIVE NON CANDIDATE RULE
"""On separe les applicants en deux groupes, les candidats avant le cutoff et ceux après.
Les candidats sont des applicants qui est meilleur que tous les candidats avant le cutoff.
Pendant le procès, il peut y avoir plusieurs candidats.
Exemple : 
liste des applicants = [4,1,2,5,3,6]
les applicants sont de 1 à 6
Les candidats sont [4,5,6]"""
candidates = []
secretaries = list(range(1, 101))
random.shuffle(secretaries)
current_max = 0

non_candidate = True
for i, secretary in enumerate(secretaries):
    if secretary > current_max:
        current_max = secretary
        plt.plot(i, secretary, marker="o", markersize=5, color="red", label="Candidate" if len(candidates) == 0 else "_nolegend_")
        candidates.append(secretary)
    else:
        
        plt.plot(i, secretary, marker="x", markersize=3, color="blue", label="Non-candidate" if non_candidate else "_nolegend_")
        non_candidate = False

plt.xticks([])
plt.plot(secretaries, linewidth=1, linestyle='dashed')
plt.ylabel("Applicant quality")
plt.legend(loc="lower right")
plt.show()
print(f"List of candidates (marked with red): {candidates}")
# %% FONCTION POUR TROUVER LES CANDIDATS
def find_candidate(secretaries, non_candidates):
    current_streak, current_best = 0, 0
    for secretary in secretaries:
        if current_streak >= non_candidates and secretary > current_best:  
            #  Once the required number of successive non-candiates in encountered,
            #  function returns the first next candidate better than all previous candidates
            return secretary
        if secretary > current_best:
            #  Whenever a new best candidate is met,
            #  the streak of non-candidates is reset to zero.
            current_best = secretary  
            current_streak = 0
        else:
            #  The streak of non candidates increases as long as
            #  the candidates are inferior to the best candidate so far.
            current_streak += 1   
# %% 3EME APPROCHE : CANDIDATE COUNT RULE
"""On va tous simplement choisir le y-th candidat"""
def find_candidate_number(secretaries):
    counter = 0  # We keep track of the number of candidates so far.
    best_candidate = 0  # We also keep track of the current best candidate
    for secretary in secretaries:
        #  When the best candidate is found, function returns the parameter y that would allow us to hire this applicant.
        if secretary == BEST_SCORE:
            return counter
        if secretary > best_candidate:
            counter += 1
            best_candidate = secretary

def create_secretary_list():
    #  Helper function that does what it says.
    #  Creates a list of randomly shuffled numbers from 1 to 100
    #  It will be used as a list of applicants. The numbers represent the quality, or skill
    #  of the applicant.
    secretaries = np.arange(1, BEST_SCORE + 1, dtype=int)
    np.random.shuffle(secretaries)
    return secretaries

# %% SIMULATION CUT OFF RULE
ITERATIONS = 100_000
score = {i : 0 for i in range(1, BEST_SCORE + 1)}
for _ in range(ITERATIONS):
    secretaries = create_secretary_list()
    for cutoff in range(1, BEST_SCORE + 1):
        if check_secretary_list(secretaries, cutoff):
            score[cutoff] += 100 / ITERATIONS
X, y_cutoff_rule = score.keys(), score.values()

plt.plot(X, y_cutoff_rule)
plt.xlabel("Cutoff")
plt.ylabel("Probability")
plt.title("Secretary Problem - The cutoff rule")
plt.scatter([max(score, key=score.get)], [max(score.values())], color="red")
print(f"The best cutoff is {max(score, key=score.get)}")
plt.show()

# %% REMARQUES
print(f"Probability for a '20%' rule: {score[20]:.1f}%")
print(f"Probability for a '60%' rule: {score[60]:.1f}%")
max_probability = max(score.values())
best_cutoff = max(score, key=score.get)
print(f"The best probability of {round(max_probability)}% is achieved after always rejecting the first {best_cutoff}% of candidates")
# %% CANDIDATE COUNT RULE SIMULATION
ITERATIONS = 100_000

score = {i : 0 for i in range(0, BEST_SCORE + 1)}
#  key in the dict represents what candidate will be hired.
#  key can also be 0, in that case the first applicant from the list will be chosen.
for _ in range(ITERATIONS):
    secretaries = create_secretary_list()
    score[find_candidate_number(secretaries)] += 100 / ITERATIONS

X, y_ccr = score.keys(), score.values()

plt.plot(X, y_ccr)
plt.xlabel("Encountered candidate")
plt.ylabel("Probability")
plt.title("Secretary Problem - Candidate count rule")
plt.scatter([max(score, key=score.get)], [max(score.values())], color="red")
print(f"The best candidate to hire is {max(score, key=score.get)}", "\n", f"with a probability of {round(max(score.values()))}%")
plt.show()    

"""Avec cette stratégie, on va accepter le 4eme candidat.
Il faut faire attention à differentier candidats et applicants."""
# %% REMARQUES
max_probability = max(score.values())
best_candidate = max(score, key=score.get)
print(f"The best probability of {round(max_probability)}% is achieved after hiring the {best_candidate}th candidate in the list.")

# %% SNCR SIMULATION
ITERATIONS = 10_000

score = {i : 0 for i in range(0, BEST_SCORE + 1)}
for _ in range(ITERATIONS):
    secretaries = create_secretary_list()
    for non_candidates in range(BEST_SCORE):
        if find_candidate(secretaries, non_candidates) == BEST_SCORE:
            score[non_candidates] += 100 / ITERATIONS
            
X, y_sncr = score.keys(), score.values()

plt.plot(X, y_sncr)
plt.xlabel("Number of successive non-candidates")
plt.ylabel("Probability")
plt.title("Secretary Problem - Successive Non-Candidate Variation")
plt.scatter([max(score, key=score.get)], [max(score.values())], color="red")
print(f"The best number of successive non-candidates is {max(score, key=score.get)}","\n", f"with a probability of {round(max(score.values()))}%")
plt.show() 

# %% COMPARAISON
y_cutoff_rule = [0] + list(y_cutoff_rule)
# Zero is added so all the three y arrays have the same dimensionality

X = list(X)
if len(X) != len(y_cutoff_rule):
    min_length = min(len(X), len(y_cutoff_rule))
    X = list(X)[:min_length]
    y_cutoff_rule = y_cutoff_rule[:min_length]

plt.plot(X, y_cutoff_rule, label="The cutoff rule")
plt.plot(X, y_ccr, label="Candidate count rule")
plt.plot(X, y_sncr, label="Successive Non-Candidate Variation")

plt.xlabel("Heuristic parameter")
plt.ylabel("Probability")
plt.title("Combined Secretary Problem Plots")
plt.legend(loc="right")

plt.show()
# %%
