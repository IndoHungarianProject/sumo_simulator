from __future__ import annotations
import os
import sys
from turtle import color
import traci
import traci.constants as tc
import time
import configparser

class Edge(traci.StepListener):
    def __init__(self, id: str) -> None:
        super().__init__()
        self.id = id
        self.aggregation_interval = 5
        self.radius = 50
        self.vehicles = []
        traci.poi.subscribeContext(self.id, tc.CMD_GET_VEHICLE_VARIABLE, 50, [
                                   tc.VAR_COLOR, tc.VAR_ARRIVED_VEHICLES_IDS])
        traci.addStepListener(self)

    def register(self, v: Vehicle):
        self.vehicles.append(v)

    def deregister(self, v: Vehicle):
        self.vehicles.remove(v)

    def step(self, t):
        vehicle_props = traci.poi.getContextSubscriptionResults(self.id)
        print(list(map(lambda x: x.id, self.vehicles)))
        return True


class Vehicle(traci.StepListener):
    def __init__(self, id: str, pois_dict: dict[str, Edge]) -> None:
        super().__init__()
        self.id = id
        self.listener_id = 0
        self.current_edge = None
        self.pois_dict = pois_dict
        traci.vehicle.subscribeContext(
            self.id, tc.CMD_GET_POI_VARIABLE, 49, [tc.VAR_POSITION])

    def register_listener(self):
        self.listener_id = traci.addStepListener(self)

    def remove_listener(self):
        self.pois_dict[self.current_edge].deregister(self)
        traci.removeStepListener(self.listener_id)

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
            elif self.current_edge != first_poi:
                self.pois_dict[self.current_edge].deregister(self)
                self.current_edge = first_poi
                self.pois_dict[self.current_edge].register(self)
        print(f'vehicle {self.id}: {poi_ids}\n')
        return True


class VehicleFactory(traci.StepListener):
    def __init__(self, pois: dict[str, Edge]):
        super().__init__()
        self.active_vehicles = []
        self.new_vehicles_to_register = []
        self.old_vehicles_to_deregister = []
        self.pois_dict = pois

    def adjust_vehicle_listeners(self):
        for veh in self.new_vehicles_to_register:
            veh.register_listener()
        for veh in self.old_vehicles_to_deregister:
            veh.remove_listener()
        self.new_vehicles_to_register, self.old_vehicles_to_deregister = [], []

    def step(self, t):
        print(f'vehiclefactory: {self.active_vehicles}')
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


def main(cp: configparser.ConfigParser):
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        sys.exit("please declare environment variable 'SUMO_HOME'")

    sumoBinary = cp['LOCATION']['sumoBin']
    networkLoc = cp['LOCATION']['networkLocation']
    sumoCmd = [sumoBinary, "-c", networkLoc, "--step-length", "0.1"]
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
    for step in range(1000):
        print("step", step)
        traci.simulationStep()
        vf.adjust_vehicle_listeners()
        time.sleep(0.5)
    traci.close()


if __name__ == '__main__':
    cp = configparser.ConfigParser()
    cp.read('./config.ini')
    main(cp)
