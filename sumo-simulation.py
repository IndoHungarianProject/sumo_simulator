from __future__ import annotations
import math
import os, shutil
import sys
import traci
import traci.constants as tc
import time
import numpy as np
import configparser
import subprocess
import logging
import pandas as pd
import matplotlib.pyplot as plt
import datetime

rng = np.random.default_rng()

def info_function(x, y):
    try:
        return 2 * (x/math.sqrt(1 + x ** 2 + 600)) / math.log(y)
    except Exception as e:
        print(f'x is {x}, y is {y}')
        raise e

class Edge(traci.StepListener):
    def __init__(self, id: str) -> None:
        super().__init__()
        self.id = id
        self.logger = logging.getLogger(f'Edge {self.id}')
        self.trigger_interval = cp.getint('EDGE', 'poi_frequency')
        self.uplink_bandwidth = cp.getint('EDGE', 'uplinkbandwidth')
        self.diameter = cp.getint('EDGE', 'poi_diameter')
        self.current_time = -1 * simulation_step
        self.vehicles = {}
        self.vehicles_in_trigger_period, self.vehicles_successful, self.vehicles_unsuccessful = 0, 0, 0
        self.accumulated_data_current_interval = cp.getint('EDGE', 'poi_frequency')
        self.can_aggregate, self.aggregation_this_period = False, True
        self.trigger_no = 0
        traci.addStepListener(self)

    def register(self, v: Vehicle):
        self.vehicles[v.id] = v

    def deregister(self, v: Vehicle):
        del self.vehicles[v.id]
    
    def send_message(self, veh_id: str|int, data_size: float) -> bool:
        self.aggregation_this_period = False # because aggregation is only needed if at least one car is in range and it has sent info, we only set this value to True if aggregation happened
        if (self.accumulated_data_current_interval + (data_size * 8)) <= self.uplink_bandwidth / (1000 / (self.trigger_interval - self.current_time)):
            self.accumulated_data_current_interval += data_size * 8
            self.vehicles_successful += 1
            self.logger.info(f'Message from vehicle {veh_id} received')
            can_upload = True
        else:
            self.vehicles_unsuccessful += 1
            self.logger.info(f'Message from vehicle {veh_id} exceeds uplinkbandwidth')
            can_upload = False
        if self.vehicles_in_trigger_period == self.vehicles_unsuccessful + self.vehicles_successful:
            self.can_aggregate = True
        return can_upload

    def trigger(self):
        self.logger.info('Triggering...')
        for _, vehicle in self.vehicles.items():
            vehicle.start_data_collection(self.id)
        self.vehicles_in_trigger_period, self.vehicles_successful, self.vehicles_unsuccessful = len(self.vehicles.keys()), 0, 0
        self.accumulated_data_current_interval = 0
        self.trigger_no += 1

    def aggregate(self):
        aggregation_density, actual_density = float(self.vehicles_successful) / (poi_area := (self.diameter/2) ** 2 * math.pi), float(self.vehicles_in_trigger_period) / poi_area
        info_value = info_function(self.vehicles_successful, self.current_time)
        self.logger.info(f'Aggregating for vehicle density {aggregation_density}, full trigger density {actual_density}, info value {info_value} ...')
        for _, vehicle in self.vehicles.items():
            vehicle.receive_aggregated_message(self.id, aggregation_density, actual_density, info_value, self.trigger_no, self.current_time)
        self.can_aggregate, self.aggregation_this_period = False, True

    def step(self, t):
        if self.can_aggregate:
            self.aggregate()
        if self.current_time + simulation_step >= self.trigger_interval:
            if not self.aggregation_this_period:
                self.aggregate()
            self.trigger()
            self.current_time = 0
        else:
            self.current_time += simulation_step
        #vehicle_props = traci.poi.getContextSubscriptionResults(self.id)
        #print(list(map(lambda x: x.id, self.vehicles)))
        return True


class Vehicle(traci.StepListener):
    def __init__(self, id: str, pois_dict: dict[str, Edge]) -> None:
        super().__init__()
        self.id = id
        self.logger = logging.getLogger(f'Vehicle {self.id}')
        self.listener_id = 0
        self.current_edge = None
        self.pois_dict = pois_dict
        traci.vehicle.subscribeContext(
            self.id, tc.CMD_GET_POI_VARIABLE, cp.getint('EDGE', 'poi_diameter') / 2, [tc.VAR_POSITION])
        self.preparation_time = None
        self.triggered, self.send_data_this_period, self.accepted_data_this_period = False, False, False
        self.curr_response_time = 0
        self.response = {}

    def _register_listener(self):
        self.listener_id = traci.addStepListener(self)

    def _remove_listener(self):
        if self.current_edge is not None:
            self.pois_dict[self.current_edge].deregister(self)
        traci.removeStepListener(self.listener_id)

    def _get_random_quantity(self, distribution: str, dist_val: str):
        distribution = cp['VEHICLE'][distribution]
        dist_val = cp.getfloat('VEHICLE', dist_val)
        if distribution == 'exponential':
            return rng.exponential(dist_val)
        elif distribution == 'uniform':
            return rng.uniform(dist_val * 0.8, dist_val * 1.2)           
        elif distribution == 'normal':
            return rng.normal(dist_val, dist_val * 0.1)

    def _reset_values_for_new_training(self):
        self.triggered, self.send_data_this_period, self.accepted_data_this_period, self.curr_response_time = True, False, False, 0
        # we subtract the simulation step thus we solve the steplistener order problem
        self.preparation_time = self._get_random_quantity('veh_dataCollectionDistribution', 'veh_dataCollectionDistributionValue')
        self.logger.info(f'Preparation time: {self.preparation_time}')

    def start_data_collection(self, poi_id: str|int):
        self._reset_values_for_new_training()

    def receive_aggregated_message(self, poi_id: str|int, vehicle_density: float, trigger_density: float, info_value: float, trigger_no: int, waiting_time: float):
        self.logger.info(f'Received aggregation')
        if self.send_data_this_period:
            if self.accepted_data_this_period:
                # to solve step listener order problem
                #self.curr_response_time += simulation_step
                self.response[(poi_id, trigger_no)] = (self.curr_response_time, waiting_time, vehicle_density, trigger_density, info_value)
                self.logger.info(f'Current response time: {self.curr_response_time}')
        self.triggered = False
            
    def step(self, t):
        # position registry, currently as long as the previous POI is within range the cars stay registered to it
        # they only change when only the next (neighboring) one is within range
        # TODO modify if more complex vehicle registry-deregistry is needed between EDGEs (POIs)
        poi_ids = traci.vehicle.getContextSubscriptionResults(self.id)
        if poi_ids is not None and len(poi_ids) > 0:
            first_poi = list(poi_ids.keys())[0]
            if self.current_edge is None:
                self.current_edge = first_poi
                self.pois_dict[self.current_edge].register(self)
                self.logger.info(f'Arriving into edge {self.current_edge} area')
            elif self.current_edge != first_poi:
                self.pois_dict[self.current_edge].deregister(self)
                self.current_edge = first_poi
                self.pois_dict[self.current_edge].register(self)
                self.logger.info(f'Changing into edge {self.current_edge} area')
            elif self.triggered:
                if not self.send_data_this_period:
                    # maybe change to: current_waiting_time > self.simulation_step
                    if self.preparation_time > 0:
                        self.preparation_time -= simulation_step
                    else:
                        self.preparation_time = 0
                        # get random data size
                        data = self._get_random_quantity('uplinkDataDistribution', 'uplinkDataDistributionvalue')
                        self.logger.info(f'Sending {data} MBs')
                        self.accepted_data_this_period = self.pois_dict[self.current_edge].send_message(self.id, data)
                        self.send_data_this_period = True
                else:
                    self.curr_response_time += simulation_step
        
        elif self.current_edge is not None:
            self.logger.info(f'Leaving edge {self.current_edge}')      
            self.pois_dict[self.current_edge].deregister(self)
            self.current_edge = None      
        self.logger.debug(f'vehicle {self.id}: {poi_ids}')
        return True


class VehicleFactory(traci.StepListener):
    def __init__(self, pois: dict[str, Edge]):
        super().__init__()
        self.logger = logging.getLogger('VehicleFactory')
        self.active_vehicles = []
        self.new_vehicles_to_register = []
        self.old_vehicles_to_deregister = []
        self.pois_dict = pois
        self.response_times = {}

    def append_response_times(self, veh: Vehicle):
        for k, v in veh.response.items():
            if k not in self.response_times:
                self.response_times[k] = [v]
            else:
                self.response_times[k].append(v)

    def get_response_time_for_all_vehicles(self):
        for veh in self.active_vehicles:
            self.append_response_times(veh)
        multiidx = pd.MultiIndex.from_tuples(self.response_times.keys())
        df = pd.DataFrame([[np.mean([i[0] for i in v]), v[0][1], v[0][2], v[0][3], v[0][4]] for _, v in self.response_times.items()], index=multiidx)
        df.columns = ['response_time', 'waiting_time', 'vehicle_density', 'trigger_density','info_value']
        if len(df.index.get_level_values(0).unique()) == 1:
            df.index = df.index.droplevel(0)
        return df
        

    def adjust_vehicle_listeners(self):
        for veh in self.new_vehicles_to_register:
            veh._register_listener()
        for veh in self.old_vehicles_to_deregister:
            self.append_response_times(veh)
            veh._remove_listener()
        self.new_vehicles_to_register, self.old_vehicles_to_deregister = [], []

    def step(self, t):
        self.logger.debug(f'Active vehicles {[veh.id for veh in self.active_vehicles]}\nNumber of active vehicles: {len(self.active_vehicles)}')
        # remove vehicles which are no longer present in the simulation
        active_veh_ids = traci.vehicle.getIDList()
        previously_active_ids = [(idx, veh.id)
                                 for idx, veh in enumerate(self.active_vehicles)]
        idx_to_remove = [
            idx for idx, id in previously_active_ids if id not in active_veh_ids]
        for idx in sorted(idx_to_remove, reverse=True):
            old_veh = self.active_vehicles.pop(idx)
            self.old_vehicles_to_deregister.append(old_veh)
        # add vehicles which newly appeared
        if len(previously_active_ids) > 0:
            _, previously_active_ids = zip(*previously_active_ids)
        ids_to_add = [
            id for id in active_veh_ids if id not in previously_active_ids]
        for id in ids_to_add:
            new_veh = Vehicle(id, self.pois_dict)
            self.new_vehicles_to_register.append(new_veh)
            self.active_vehicles.append(new_veh)
        return True


def main():
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        sys.exit("please declare environment variable 'SUMO_HOME'")
    
    networkLoc = cp['LOCATION']['networkLocation']
    # run random vehicle generation
    if os.name == 'nt':
        pythonCommand = 'python'
    else:
        pythonCommand = 'python3'
    subprocess.run([pythonCommand, f"{os.environ['SUMO_HOME']}/tools/randomTrips.py", '-n', f'{networkLoc}/osm.net.xml', 
                    '-a', f'{networkLoc}/additionals.xml', '-e', cp['SIMULATION']['vehicle_generation_endtime'], '--random', '--random-depart-offset', '5', 
                    '--period', str(1/cp.getfloat('SIMULATION', 'vehicles_generated_per_second')), '--validate', '-r', f'{networkLoc}/complex_route.rou.xml'], shell=True)

    sumoBinary = cp['LOCATION']['sumoBin']
    sumoCmd = [sumoBinary, "-c", f'{networkLoc}/complex.sumocfg', "--step-length", cp['SIMULATION']['step_length'], '--quit-on-end']
    traci.start(sumoCmd)
    # add vehicle manager
    poi_ID = traci.poi.getIDList()
    print(poi_ID)
    pois = {}
    for i in poi_ID:
        e = Edge(i)
        pois[i] = e
    vf = VehicleFactory(pois)
    traci.addStepListener(vf)
    for step in range(cp.getint('SIMULATION', 'simulation_endtime') * int(1000 / simulation_step)):
        print("step", step)
        traci.simulationStep()
        vf.adjust_vehicle_listeners()
        time.sleep(cp.getfloat('SIMULATION', 'wait_between_steps'))
    traci.close()
    return vf.get_response_time_for_all_vehicles()



if __name__ == '__main__':
    global cp
    cp = configparser.ConfigParser()
    cp.read('./config.ini')
    global simulation_step
    simulation_step = cp.getfloat('SIMULATION', 'step_length') * 1000
    logging.basicConfig(level=cp.getint('SIMULATION', 'loglevel'))
    result_df = pd.DataFrame()
    for i in range(cp.getint('SIMULATION', 'simulationRounds')):
        concat_df = pd.concat([result_df, main()])
        result_df = concat_df.groupby(concat_df.index).mean()
        print(result_df.to_string())
    curr_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S.%f')
    result_df.to_csv(f'./simulation_results/result_{curr_time}.csv')
    shutil.copyfile('config.ini', f'./simulation_results/config_{curr_time}.ini')
    
    #plot
    fig, ax = plt.subplots()
    p1, = ax.plot(result_df.index, result_df['response_time'], color='xkcd:red', label='Response time (ms)')
    ax.set_xlabel('Aggregations') 
    ax.set_ylabel('Response time (ms)', color='xkcd:red')
    ax2 = ax.twinx()
    p2, = ax2.plot(result_df.index, result_df['vehicle_density'], color='xkcd:grey', label=f'Vehicle density (vehicle/km\N{SUPERSCRIPT TWO})')
    ax2.set_ylabel('Vehicle density (vehicle/km\N{SUPERSCRIPT TWO})', color='xkcd:grey')
    ax3 = ax.twinx()
    p3, = ax3.plot(result_df.index, result_df['info_value'], color='xkcd:blue', label='Information value')
    ax3.set_ylabel('Information value', color='xkcd:blue')
    ax.legend(handles=[p1, p2, p3], loc='best')
    ax3.spines['right'].set_position(('outward', 80))
    fig.tight_layout()
    fig.savefig(f'./simulation_results/figure_{curr_time}.pdf')