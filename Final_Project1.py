#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from scipy.stats import cosine
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


N = 10000
a, b = (0,1.75)
x_min, x_max = (0, 1.0)
randx = np.random.uniform(x_min, x_max, N)
y = cosine.ppf(randx, a, b)
print(f'Real value to find: {cosine(a,b).cdf(.55)}')
print(f'Integral value:  {(x_max-x_min)*y.sum()/N}')
print(f'Calculation error: {np.sqrt((x_max-x_min)*(y*y).sum()/N - (x_max-x_min)*y.mean()**2)/np.sqrt(N)}')


# In[ ]:


randx = np.random.uniform(x_min,x_max, N)
y = cosine.ppf(randx, a, b)
randy = np.random.uniform(0,y.max(), N)
print(f'Integral value: {(x_max-x_min)*y.max()*(randy <= y).sum()/N}')


# In[ ]:


plt.figure(figsize=(8,5))
color = randy[:1000] <= y[:1000]
x = np.linspace(cosine.ppf(0.01),cosine.ppf(0.99), 100)
plt.plot(x, cosine.pdf(x),'r-', lw=5, alpha=0.6, label='cosine pdf')
plt.scatter(randx[:1000], randy[:1000], alpha=.2, c = color)
plt.xlabel('x')
plt.ylabel('y')
plt.show()


# In[ ]:




