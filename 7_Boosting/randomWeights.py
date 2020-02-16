import numpy as np

select_from = list(range(5))
print("selecting from:", select_from, "\n")

### Implement 1/n weights (which is the np.random.choice default)
### Also note:
### replace = True (default, used for boot-strapping)
### size = len(array) - This will be the sample size used in our algorithms.

print("Equal Weights:\n",
    np.random.choice(select_from,
                size = len(select_from),
                replace = True,
                p = np.array([.2,.2,.2,.2,.2,])
                )
)

### Now, using uneven weights

print("\nWeights of [.9,.05,.03,.02,0]:\n",
    np.random.choice(select_from,
                size = len(select_from),
                p = np.array([.9,.05,.03,.02,0]))
)