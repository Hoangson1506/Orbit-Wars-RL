import multiprocessing as mp
from typing import Callable, Any
from env import OrbitWarsEnv, StepResult, TurnBatch

def _worker(
    remote: mp.connection.Connection,
    parent_remote: mp.connection.Connection,
    env_fn: Callable[[], OrbitWarsEnv]
) -> None:
    parent_remote.close()
    try:
        env = env_fn()
        while True:
            cmd, data = remote.recv()
            if cmd == "step":
                remote.send(env.step(data))
            elif cmd == "reset":
                remote.send(env.reset(seed=data))
            elif cmd == "sync":
                if hasattr(env.opponent, "policy"):
                    env.opponent.policy.load_state_dict(data)
                remote.send(True) 
            elif cmd == "close":
                remote.close()
                break

    except KeyboardInterrupt:
        pass

class SubprocVectorEnv:
    """Runs multiple OrbitWarsEnv instances in parallel processes."""
    def __init__(self, env_fns: list[Callable[[], OrbitWarsEnv]]):
        self.num_envs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(self.num_envs)])
        self.processes = []

        for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes, env_fns):
            process = mp.Process(target=_worker, args=(work_remote, remote, env_fn))
            process.daemon = True
            process.start()
            self.processes.append(process)
            work_remote.close()

    def step(self, actions: list[list[list[float | int]]]) -> list[StepResult]:
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", action))
        return [remote.recv() for remote in self.remotes]

    def reset(self, seeds: list[int]) -> list[TurnBatch]:
        for remote, seed in zip(self.remotes, seeds):
            remote.send(("reset", seed))
        return [remote.recv() for remote in self.remotes]
    
    def reset_one(self, env_idx: int, seed: int) -> Any:
        """Resets a single environment and returns its initial observation batch."""
        self.remotes[env_idx].send(("reset", seed))
        return self.remotes[env_idx].recv()
    
    def sync_opponents(self, state_dict: dict) -> None:
        """Sends the latest neural network weights to all subprocess opponents."""
        # Move state dict to CPU to ensure safe serialization across processes
        cpu_state_dict = {k: v.cpu() for k, v in state_dict.items()}
        for remote in self.remotes:
            remote.send(("sync", cpu_state_dict))
        # Wait for all processes to acknowledge before continuing
        for remote in self.remotes:
            remote.recv()
    
    def close(self) -> None:
        for remote in self.remotes:
            remote.send(("close", None))
        for process in self.processes:
            process.join()

            