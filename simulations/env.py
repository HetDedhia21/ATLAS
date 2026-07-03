from simulations.config import JUNCTION
import random

class TrafficEnv:
    def __init__(self):
        self.lanes = JUNCTION["lanes"]
        self.phases = JUNCTION["phases"]
        self.queues = {lane: 0 for lane in self.lanes}
        self.current_phase = "NS"

    def reset(self):
        for lane in self.queues:
            self.queues[lane] = 0
        self.current_phase = "NS"
        return self.get_state()

    def step(self, action):
        self.current_phase = list(self.phases.keys())[action]

        outgoing = {lane: 0 for lane in self.queues}

        # Random arrivals
        for lane in self.queues:
            self.queues[lane] += random.randint(0, 2)

        # Vehicles pass on green
        for lane in self.phases[self.current_phase]:
            passed = min(3, self.queues[lane])
            self.queues[lane] -= passed
            outgoing[lane] = passed

        reward = -sum(self.queues.values())

        return self.get_state(), reward, outgoing

    def get_state(self):
        return list(self.queues.values())