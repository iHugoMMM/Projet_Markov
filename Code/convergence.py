# %% IMPORTS
import numpy as np
import matplotlib.pyplot as plt
import time
# %%
n = 50  #total number of applicants
nplot = np.empty([1,1])
# %%
for k in range(2,n):
    m = 10000 #number of repeats 
    plot = np.empty([1,1])
    
    for j in range(1,k):    
        passed = 0
        for i in range (0,m): #multiple runs
            array = np.random.rand(k)
            picked = np.argmax(array[j:]>max(array[0:j])) + j
            best = np.argmax(array)
            if best == picked:
                passed = passed+1
        #print(passed/m)
        plot = np.append(plot,[passed/m])
    #print(plot)
    plot = plot[1:]
    x = range(1,k)
    y = plot
    #print("N = ",k)
    print("Check ",plot.argmax()+1,"if you have",k," applicants,", round(100* plot.max(),2),"% chance getting the best applicant")
    nplot =np.append(nplot,plot.max())
# %% PLOT
nplot = nplot[1:]
x = range(2,n)
y = nplot
plt.plot(x, y, 'o', color='black')
plt.xlabel("Number of Applicants")
plt.ylabel("Probability of Best Applicant")
plt.title("Probability of Finding Best Applicant")
plt.show()
# %%
