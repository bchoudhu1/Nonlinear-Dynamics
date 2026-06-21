import matplotlib.pyplot as plt

def logistic_nth(n, x0, R):
    res = []
    res.append(x0)
    for i in range(0, n):
        res.append(R*res[i]*(1-res[i]))
    return res[-1] 


def logistic(n, x0, R):
    res = []
    res.append(x0)
    for i in range(0, n):
        res.append(R*res[i]*(1-res[i]))
    return res


x0_0=0.2
R=3.72
x0_1 = 0.2000001

diff_array = []

for m in range(0,201):
    diff_array.append(abs(logistic_nth(m,x0_0,R)-logistic_nth(m,x0_1,R)))

plt.scatter(range(0,201),diff_array)
plt.show()

a = sum(abs(x-y) for x, y in zip(logistic(499999, x0_0, R),
                                 logistic(499999, x0_1, R)))


print(f"{a / 500000:.20f}")
