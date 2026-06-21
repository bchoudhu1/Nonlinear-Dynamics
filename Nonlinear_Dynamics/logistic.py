#Compute up to the n-th term of the logistic map
import matplotlib.pyplot as plt

def logistic(n, x0, R):
    res = []
    res.append(x0)
    for i in range(0, n):
        res.append(R*res[i]*(1-res[i]))
    return res 

x0=0.2
R=2.6

# Print up tp x3
print(logistic(3, x0, R))

#Print up to x10
print(logistic(10, x0, R))

#Plot up to n iterates of the logistic-map
n = 50
plt.scatter(range(0,n+1), logistic(n, 0.2,2.0))
plt.xlabel("n")
plt.ylabel("n-th iterate")
plt.show()