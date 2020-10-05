import multiprocessing as mp
import os
import queue
import signal

import torch

class BaseCallback:
    def __init__(self, verbose: int = 0):
        self.model = None
        self.verbose = verbose

    def init_callback(self, model):
        self.model = model

    def on_training_start(self):
        pass

    def on_step(self, sim_step, learn_step, **kwargs):
        # If False, break off Training
        return True

    def on_training_end(self):
        pass

    def on_evaluation_end(self, sim_step, learn_step, **kwargs):
        return True

    def update_curriculum(self, curriculum):
        pass

    def on_episode_done(self, episode_results):
        pass


class EpisodeResults:
    def __init__(self):
        self.reward = 0
        self.done = False
        self.n_steps = 0
        self.infos = []

    def update(self, step_results):
        self.reward += step_results.reward
        self.done = step_results.done
        self.n_steps = step_results.n_steps
        self.infos.append(step_results.info)


class WorkerCommunication:
    def __init__(self):
        self._model_output_queue = mp.Queue()
        self._observation_queue = mp.Queue()
        self._curriculum_queue = mp.Queue()
        self._event_reset = mp.Event()
        self._event_quit = mp.Event()
        self.pid = None

    def is_interrupted(self):
        return self.is_reset() or self.is_quit()

    def is_reset(self):
        return self._event_reset.is_set()

    def done_reset(self):
        self._event_reset.clear()

    def put_reset(self):
        os.kill(self.pid, signal.SIGUSR1)
        self._event_reset.set()

    def get_curriculum(self, block=False):
        curriculum = None
        try:
            while True:
                curriculum = self._curriculum_queue.get(block=block)
        except queue.Empty:
            pass
        return curriculum

    def put_update_curriculum(self, curriculum):
        self._curriculum_queue.put(curriculum)
        self.put_reset()

    def put_observation(self, observation):
        self._observation_queue.put(observation)

    def get_observation(self, block=True):
        return self._observation_queue.get(block=block)

    def _convert_to_numpy(self, container):
        if isinstance(container, list):
            return [self._convert_to_numpy(c) for c in container]
        elif isinstance(container, dict):
            return {k: self._convert_to_numpy(v) for k, v in container.items()}
        elif isinstance(container, torch.Tensor):
            return container.detach().cpu().numpy()
        return container

    def put_model_output(self, output):
        self._model_output_queue.put(self._convert_to_numpy(output))

    def get_model_output(self, block=True):
        return self._model_output_queue.get(block=block)

    def put_quit(self):
        return self._event_quit.set()

    def is_quit(self):
        return self._event_quit.set()


class StepResult:
    def __init__(self, observation, is_reset, action_chosen, reward, done, info, n_steps):
        self.observation = observation
        self.is_reset = is_reset
        self.action_chosen = action_chosen
        self.reward = reward
        self.done = done
        self.info = info
        self.n_steps = n_steps



class EmptyLogger:
    def add_scalar(self, name, value, step, steps_added=1):
        pass

    def add_mean_scalar(self, name, value, step, save_every):
        pass
