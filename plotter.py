import matplotlib.pyplot as plt

datax = [0,1,2,3,4,5]
datay = [5,6,7,8,9,19]
datayy = [3,4,5,6,11,34]
datayyy = [120,231,432,231,265,444]

fig, ax = plt.subplots(figsize=(8,5))
p1, = ax.plot(datax, datay, color='xkcd:red', label='Response time')
ax.set_xlabel('Aggregations')
ax.set_ylabel('Response time', color='xkcd:red')
ax2 = ax.twinx()
p2, = ax2.plot(datax, datayy, color='xkcd:pale rose', label='Vehicle density')
ax2.set_ylabel('Vehicle density', color='xkcd:pale rose')
ax3 = ax.twinx()
p3, = ax3.plot(datax, datayyy, color='xkcd:wine', label='Information value')
ax3.set_ylabel('Information value', color='xkcd:wine')
ax.legend(handles=[p1, p2, p3], loc='best')
ax3.spines['right'].set_position(('outward', 60))
fig.tight_layout()
#plt.show()
fig.savefig('figure.png')