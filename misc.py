import configparser
import os
l = list(range(36))
d = {}
cp = configparser.ConfigParser()
for i in l:
    filename = f'./new_simulation_results/config_{i}.ini'
    cp.read(filename)
    d[i] = f"wt{cp.getfloat('VEHICLE','veh_dataCollectionDistributionValue') / cp.getfloat('EDGE', 'poi_frequency')}d{cp['EDGE']['poi_diameter']}tt{cp['EDGE']['poi_frequency']}bw{cp['EDGE']['uplinkbandwidth']}"
    os.rename(f'./new_simulation_results/config_{i}.ini', f'./new_simulation_results/config_{d[i]}.ini')
    os.rename(f'./new_simulation_results/figure_{i}.pdf', f'./new_simulation_results/figure_{d[i]}.pdf')
    os.rename(f'./new_simulation_results/result_{i}.csv', f'./new_simulation_results/result_{d[i]}.csv')
print(d)

