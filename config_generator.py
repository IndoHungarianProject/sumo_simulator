import configparser

def setter_fun(endtime, rounds, veh_gen_end):
    cp.set('SIMULATION', 'simulation_endtime', f'{endtime}')
    cp.set('SIMULATION', 'simulationRounds', f'{rounds}')
    cp.set('SIMULATION', 'vehicle_generation_endtime', f'{veh_gen_end}')

global cp
cp = configparser.ConfigParser()
cp.read('config.ini')
diameter, frequency, uplinkbandwidth = [500, 1000, 2000, 4000], [1000, 10000, 100000], [10, 100, 1000]
i = 0
for d in diameter:
    cp.set('EDGE', 'poi_diameter', f'{d}')
    for f in frequency:
        cp.set('EDGE', 'poi_frequency', f'{f}')
        cp.set('VEHICLE', 'veh_dataCollectionDistributionValue', f'{0.1 * f}')
        if f == 1000:
            setter_fun(60, 10, 10)
        elif f == 10000:
            setter_fun(100, 10, 16)
        else:
            setter_fun(1000, 5, 10)
        for ub in uplinkbandwidth:
            cp.set('EDGE', 'uplinkbandwidth', f'{ub}')
            with open(f'config_{i}.ini', 'w') as configfile:
                cp.write(configfile)
                i += 1