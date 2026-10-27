import cma
import matplotlib.pyplot as plt
 
cma.plot()
plt.savefig('default_plot.png', dpi=300, bbox_inches='tight')
plt.show(block=True)
input()
