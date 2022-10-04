from __future__ import annotations
import os
import sys
import traci
import traci.constants as tc
import time
import numpy as np
import configparser
import subprocess
import logging

class Edge(traci.StepListener):
    def __init__(self, id: str) -> None:
        super().__init__()
        self.id = id
        self.logger = logging.getLogger(f'Edge {self.id}')
        self.aggregation_interval = cp.getint('EDGE', 'poi_frequency')
        self.uplink_bandwidth = cp.getint('EDGE', 'uplinkbandwidth')
        self.current_time = 0
        #self.radius = 50
        self.vehicles = {}
        self.accumulated_data_current_interval = 0
        #traci.poi.subscribeContext(self.id, tc.CMD_GET_VEHICLE_VARIABLE, 50, [
        #                           tc.VAR_COLOR, tc.VAR_ARRIVED_VEHICLES_IDS])
        traci.addStepListener(self)

    def register(self, v: Vehicle):
        self.vehicles[v.id] = v

    def deregister(self, v: Vehicle):
        del self.vehicles[v.id]
    
    def send_message(self, veh_id: str|int, data_size: float) -> bool:
        if (self.accumulated_data_current_interval + (data_size * 8)) <= self.uplink_bandwidth / (1000 / self.aggregation_interval):
            self.accumulated_data_current_interval += data_size * 8
            self.logger.info(f'Message from vehicle {veh_id} received')
            return True
        else:
            self.logger.info(f'Message from vehicle {veh_id} exceeds uplinkbandwidth')
            return False

    def aggregate(self):
        self.logger.info('Aggregating...')
        for _, vehicle in self.vehicles.items():
            vehicle.receive_aggregated_message(self.id)
        self.accumulated_data_current_interval = 0

    def step(self, t):
        if self.current_time + simulation_step >= self.aggregation_interval:
            self.aggregate()
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
        self.waiting_time = None
        self.send_data_this_period, self.accepted_data_this_period = False, False
        self.curr_latency = 0
        self.latencies = {}

    def _register_listener(self):
        self.listener_id = traci.addStepListener(self)

    def _remove_listener(self):
        if self.current_edge is not None:
            self.pois_dict[self.current_edge].deregister(self)
        traci.removeStepListener(self.listener_id)

    def _get_random_quantity(self, distribution: str, dist_val: str):
        distribution = cp['VEHICLE'][distribution]
        dist_val = cp.getint('VEHICLE', dist_val)
        if distribution == 'exponential':
            return np.random.exponential(dist_val)
        elif distribution == 'uniform':
            return np.random.uniform(dist_val * 0.8, dist_val * 1.2)           
        elif distribution == 'normal':
            return np.random.normal(dist_val, dist_val * 0.1)

    def _reset_values_for_new_training(self):
        self.send_data_this_period, self.accepted_data_this_period, self.curr_latency = False, False, 0
        # we subtract the simulation step thus we solve the steplistener order problem
        self.waiting_time = self._get_random_quantity('veh_dataCollectionDistribution', 'veh_dataCollectionDistributionValue') - simulation_step
        self.logger.info(f'Waiting time: {self.waiting_time}')

    def receive_aggregated_message(self, poi_id: str|int):
        self.logger.info(f'Received aggregation')
        if self.send_data_this_period:
            if self.accepted_data_this_period:
                # to solve step listener order problem
                self.curr_latency += simulation_step
                if poi_id not in self.latencies:
                    self.latencies[poi_id] = [self.curr_latency]
                else:
                    self.latencies[poi_id].append(self.curr_latency)
                self.logger.info(f'Current latency: {self.curr_latency} latencies average: {np.mean(self.latencies[poi_id])}')
            self._reset_values_for_new_training()
            
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
                self._reset_values_for_new_training()
                self.logger.info(f'Arriving into edge {self.current_edge} area')
            elif self.current_edge != first_poi:
                self.pois_dict[self.current_edge].deregister(self)
                self.current_edge = first_poi
                self.pois_dict[self.current_edge].register(self)
                self._reset_values_for_new_training()
                self.logger.info(f'Changing into edge {self.current_edge} area, waiting time: {self.waiting_time}')
            else:
                if not self.send_data_this_period:
                    # maybe change to: current_waiting_time > self.simulation_step
                    if self.waiting_time - simulation_step > 0:
                        self.waiting_time -= simulation_step
                    else:
                        self.waiting_time = 0
                        # get random data size
                        data = self._get_random_quantity('uplinkDataDistribution', 'uplinkDataDistributionvalue')
                        self.logger.info(f'Sending {data} MBs')
                        self.accepted_data_this_period = self.pois_dict[self.current_edge].send_message(self.id, data)
                        self.send_data_this_period = True
                else:
                    self.curr_latency += simulation_step
        
        elif self.current_edge is not None:
            self.logger.info(f'Leaving edge {self.current_edge}')      
            self.pois_dict[self.current_edge].deregister(self)
            self.current_edge = None      
        print(f'vehicle {self.id}: {poi_ids}\n')
        return True


class VehicleFactory(traci.StepListener):
    def __init__(self, pois: dict[str, Edge]):
        super().__init__()
        self.logger = logging.getLogger('VehicleFactory')
        self.active_vehicles = []
        self.new_vehicles_to_register = []
        self.old_vehicles_to_deregister = []
        self.pois_dict = pois

    def adjust_vehicle_listeners(self):
        for veh in self.new_vehicles_to_register:
            veh._register_listener()
        for veh in self.old_vehicles_to_deregister:
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
    subprocess.run([pythonCommand, f"{os.environ['SUMO_HOME']}/tools/randomTrips.py", '-n', f'{networkLoc}/simple.net.xml', 
                    '-a', f'{networkLoc}/addition.xml', '-e', cp['SIMULATION']['endtime'], '--random', '--fringe-factor', '10', 
                    '--period', str(1/cp.getfloat('SIMULATION', 'vehicles_generated_per_second')), '--validate', '-r', f'{networkLoc}/simple_route.rou.xml'], shell=True)

    sumoBinary = cp['LOCATION']['sumoBin']
    sumoCmd = [sumoBinary, "-c", f'{networkLoc}/simple.sumocfg', "--step-length", cp['SIMULATION']['step_length']]
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
    for step in range(cp.getint('SIMULATION', 'endtime') * int(1000 / simulation_step)):
        print("step", step)
        traci.simulationStep()
        vf.adjust_vehicle_listeners()
        time.sleep(cp.getfloat('SIMULATION', 'wait_between_steps'))
    
    traci.close()


if __name__ == '__main__':
    global cp
    cp = configparser.ConfigParser()
    cp.read('./config.ini')
    global simulation_step
    simulation_step = cp.getfloat('SIMULATION', 'step_length') * 1000
    logging.basicConfig(level=cp.getint('SIMULATION', 'loglevel'))
    main()
