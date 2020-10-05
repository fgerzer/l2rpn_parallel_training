import abc
import signal
from queue import Queue

import numpy as np
import pandas
from grid2op.Exceptions.Grid2OpException import Grid2OpException

from utils import StepResult, WorkerCommunication

from contextlib import contextmanager


class WorkerInterrupted(BaseException):
    pass


def raise_interrupt(signum, stack):
    raise WorkerInterrupted


@contextmanager
def sigusr_raise():
    prev_handler = signal.signal(signal.SIGUSR1, raise_interrupt)
    try:
        yield
    finally:
        signal.signal(signal.SIGUSR1, prev_handler)


@contextmanager
def sigusr_ignored():
    prev_handler = signal.signal(signal.SIGUSR1, signal.SIG_IGN)
    try:
        yield
    finally:
        signal.signal(signal.SIGUSR1, prev_handler)


class EnvironmentWorker(abc.ABC):
    def __init__(self, data_path, env_name, train, action_space, initial_curriculum,
                 communication: WorkerCommunication, fast_forward, skip_steps=None, seed=None):
        self.data_path = data_path
        self.env_name = env_name
        self.action_space = action_space
        self.train = train
        self.skip_steps = skip_steps
        self.fast_forward = fast_forward
        self.random_state = np.random.RandomState(seed)
        self.comms = communication
        self.env, self.action_processor, self.observer = None, None, None
        self.update_curriculum(initial_curriculum)

    def communicate_model(self, observation, reward, done, info, is_reset, action_chosen, n_steps):
        if self.comms.is_interrupted():
            return False
        step_result = StepResult(
            observation=observation, reward=reward, done=done, info=info, is_reset=is_reset,
            action_chosen=action_chosen, n_steps=n_steps
        )
        self.comms.put_observation(step_result)
        output = self.comms.get_model_output(block=True)
        return output

    def update_curriculum(self, curriculum):
        with sigusr_ignored():
            self._update_curriculum(curriculum)

    def _update_curriculum(self, curriculum):
        raise NotImplementedError

    def quit_worker(self):
        pass

    def loop_single(self):
        n_ids = len(self.env.chronics_handler.real_data.subpaths)
        env_id = self.random_state.randint(n_ids)
        self.env.set_id(env_id)
        observation = self.env.reset()
        if self.fast_forward:
            max_timesteps = self.env.chronics_handler.max_timestep() - 5
            self.env.fast_forward_chronics(self.random_state.randint(max_timesteps))

        self.comms.done_reset()
        reward = 0
        done = False
        is_reset = True
        action_chosen = 0
        info = {"reset"}
        # print(os.getpid(), self.train, self.curriculum)
        n_steps = 0
        actions_remaining = Queue()
        while not done:
            if actions_remaining.empty():
                processed_observation = self.observer(observation)

                output = self.communicate_model(processed_observation, reward, done, info, is_reset, action_chosen, n_steps)
                reward = 0
                actions, action_chosen = self.action_processor(observation, processed_observation, output)
                [actions_remaining.put(a) for a in actions]
            observation, this_reward, done, info = self.env.step(actions_remaining.get())
            reward += this_reward
            n_steps += 1
            is_reset = False
        processed_observation = self.observer(observation)
        self.communicate_model(processed_observation, reward, done, info, is_reset, action_chosen, n_steps)

    def loop_infinite(self):
        while not self.comms.is_quit():
            try:
                with sigusr_raise():
                    self.comms.done_reset()
                    curriculum = self.comms.get_curriculum()
                    if curriculum is not None:
                        self.update_curriculum(curriculum)
                    self.loop_single()
            except (WorkerInterrupted, pandas.errors.ParserError):
                pass
            except Grid2OpException as e:
                print(f"{e}, but continuing.")
