from simulations.env import TrafficEnv

class MultiAgentEnv:
    def __init__(self):
        self.junctions = {
            "J0": TrafficEnv(),
            "J1": TrafficEnv(),
            "J2": TrafficEnv(),
            "J3": TrafficEnv(),
            "J4": TrafficEnv(),
        }

        self.connections = {
            "J1": {"S": "J0"},
            "J2": {"E": "J0"},
            "J3": {"W": "J0"},
            "J4": {"N": "J0"},
            "J0": {
                "N": "J1",
                "S": "J4",
                "E": "J3",
                "W": "J2"
            }
        }

    def reset(self):
        states = {}
        for j in self.junctions:
            states[j] = self.junctions[j].reset()
        return states

    def step(self, actions):
        next_states = {}
        rewards = {}
        all_outgoing = {}

        # Step each junction
        for j in self.junctions:
            state, reward, outgoing = self.junctions[j].step(actions[j])
            next_states[j] = state
            rewards[j] = reward
            all_outgoing[j] = outgoing

        # Move vehicles between junctions
        for j in self.connections:
            for direction, target in self.connections[j].items():
                cars = all_outgoing[j][direction]

                # Add cars to target junction (opposite lane)
                if direction == "N":
                    self.junctions[target].queues["S"] += cars
                elif direction == "S":
                    self.junctions[target].queues["N"] += cars
                elif direction == "E":
                    self.junctions[target].queues["W"] += cars
                elif direction == "W":
                    self.junctions[target].queues["E"] += cars

        return next_states, rewards