import multiprocessing as mp
import os
import queue
import time
import numpy as np

from utils import WorkerCommunication, EmptyLogger, EpisodeResults, BaseCallback


class ParallelTraining:
    def __init__(self, model, start_worker, n_workers, data_path, env_name, action_space, initial_curriculum,
                 train_steps, eval_interval, model_dir, num_eval_episodes, callbacks=None, max_eval_steps=None,
                 logger=None):
        self.sim_steps = 0
        self.learn_steps = 0
        self.last_eval_time = 0
        self.num_eval_episodes = num_eval_episodes
        self.n_workers = n_workers
        self.logger = logger or EmptyLogger()
        self.callbacks = callbacks or BaseCallback()
        self.train_steps = train_steps
        self.model_dir = model_dir
        self.max_eval_steps = max_eval_steps
        self.best_eval_score = -np.inf
        self.train_communication = [WorkerCommunication() for _ in range(n_workers)]
        self.train_processes = [mp.Process(target=start_worker, args=(
            comm, action_space, data_path, env_name, True, initial_curriculum
        )) for comm in self.train_communication]
        self.test_communication = [WorkerCommunication() for _ in range(n_workers)]
        self.test_processes = [mp.Process(target=start_worker, args=(
            comm, action_space, data_path, env_name, False, initial_curriculum
        )) for comm in self.test_communication]

        self.eval_interval = eval_interval
        self.model = model
        print(f"{len(self.train_processes)} training processes; {len(self.test_processes)} test processes; ")
        self.callbacks.init_callback(self)

    def kill_workers(self):
        for comm in self.train_communication + self.test_communication:
            comm.put_quit()

        for p in self.train_processes + self.test_processes:
            p.terminate()
            p.join()

    def start_workers(self):
        [p.start() for p in self.train_processes]
        [p.start() for p in self.test_processes]
        for process, comms in zip(self.train_processes, self.train_communication):
            comms.pid = process.pid
        for process, comms in zip(self.test_processes, self.test_communication):
            comms.pid = process.pid
        time.sleep(2)

    def get_observations(self, test, skip_wait=False):
        indices = []
        results = []
        if test:
            comms = self.test_communication
        else:
            comms = self.train_communication
        while len(results) == 0:
            for i, comm in enumerate(comms):
                try:
                    obs = comm.get_observation(block=False)
                    obs.observation = obs.observation.to_tensors()
                    results.append(obs)
                    indices.append(i)
                except queue.Empty:
                    pass
            if skip_wait:
                break
        return indices, results

    def update_curriculum(self, curriculum):
        self.callbacks.update_curriculum(curriculum)
        for comm in self.train_communication + self.test_communication:
            comm.put_update_curriculum(curriculum)
        self.max_eval_steps = curriculum.get("max_eval_steps", self.max_eval_steps)

    def evaluate(self, overwrite_maxlength=-1):
        start_time = time.perf_counter()
        n_workers = len(self.test_processes)
        episodes_finished = []
        [comm.put_reset() for comm in self.test_communication]
        cur_episodes = [EpisodeResults() for _ in range(self.n_workers)]
        valid_workers = np.ones(n_workers, dtype=bool)
        if overwrite_maxlength != -1:
            max_eval_steps = overwrite_maxlength
        else:
            max_eval_steps = self.max_eval_steps
        while len(episodes_finished) < self.num_eval_episodes:
            indices, cur_step_results = self.get_observations(test=True)
            observations = [s.observation for s in cur_step_results]
            observations = observations[0].to_batch(observations)
            logits = self.model.get_logits(observations)
            logits = logits.split_batch()
            for i, queue_idx in enumerate(indices):
                sr = cur_step_results[i]
                stop_eval_early = max_eval_steps is not None and sr.n_steps >= max_eval_steps
                if valid_workers[queue_idx]:
                    cur_episodes[queue_idx].update(sr)
                    if sr.done or stop_eval_early:
                        episodes_finished.append(cur_episodes[queue_idx])
                        cur_episodes[queue_idx] = EpisodeResults()
                        if n_workers + len(episodes_finished) > self.num_eval_episodes:
                            valid_workers[queue_idx] = False
                if stop_eval_early:
                    self.test_communication[queue_idx].put_reset()
                else:

                    model_output = logits[i].to_numpy()
                    self.test_communication[queue_idx].put_model_output(model_output)
        steps_by_episode = [e.n_steps for e in episodes_finished]
        reward_by_episode = [e.reward for e in episodes_finished]
        mean_return = np.mean(reward_by_episode)
        self.logger.add_scalar("reward/eval", mean_return, self.sim_steps)
        self.logger.add_scalar("n_steps/eval", np.mean(steps_by_episode), self.sim_steps)
        if mean_return > self.best_eval_score:
            self.best_eval_score = mean_return
            self.model.save_models(os.path.join(self.model_dir, 'best'))
        end_time = time.perf_counter()
        print('-' * 60)
        print(f'Evaluation@Sim step: {self.sim_steps:>5}  Learn step: {self.learn_steps:>5} '
              f'Episodes evaluated: {len(episodes_finished)} '
              f'mean return: {mean_return:>5.1f} '
              f'mean steps/episode: {np.mean(steps_by_episode):>5.1f} ({np.min(steps_by_episode)}--{np.max(steps_by_episode)}) '
              f'{end_time - start_time:.2f}s eval; {start_time - self.last_eval_time:.2f}s training')
        print(self.create_diagram(episodes_finished, binsize=500))
        print('-' * 60)
        self.last_eval_time = time.perf_counter()
        return episodes_finished

    def create_diagram(self, episodes, binsize):
        chars = "·▁▂▃▄▅▆▇█"
        lengths = [e.n_steps for e in episodes]
        bins = list(range(0, 8100, binsize))
        if bins[-1] < 8100:
            bins.append(8100)
        histogram, _ = np.histogram(lengths, bins=bins, density=False)
        max_val = np.max(histogram)
        relatives = histogram / max_val
        binned = np.ceil(relatives * 8).astype(int)
        diagram = "".join([chars[b] for b in binned])
        return diagram

    def model_loop(self):
        self.callbacks.on_training_start()
        episode_results = [EpisodeResults() for _ in range(self.n_workers)]
        self.episodes = 0
        self.last_eval_time = time.perf_counter()
        last_seen_states = {}
        [comm.put_reset() for comm in self.train_communication]
        while self.sim_steps <= self.train_steps:
            indices, cur_step_results = self.get_observations(test=False, skip_wait=True)
            if len(indices) > 0:
                observations = [s.observation for s in cur_step_results]
                observations = observations[0].to_batch(observations)
                logits = self.model.get_logits(observations)
                self.sim_steps += logits.n_batch.item()
                logits = logits.split_batch()
                if not self.callbacks.on_step(self.sim_steps, self.learn_steps, step_results=cur_step_results):
                    break
                for i, queue_idx in enumerate(indices):
                    model_output = logits[i].to_numpy()
                    self.train_communication[queue_idx].put_model_output(model_output)
                for i, queue_idx in enumerate(indices):
                    sr = cur_step_results[i]
                    if queue_idx in last_seen_states:
                        old_state = last_seen_states[queue_idx]
                        these_results = episode_results[queue_idx]
                        these_results.update(sr)
                        if sr.done:
                            self.logger.add_mean_scalar("reward/train", these_results.reward, self.sim_steps, save_every=100)
                            self.logger.add_mean_scalar("episode_steps/train", these_results.n_steps, self.sim_steps, save_every=100)
                            self.callbacks.on_episode_done(these_results)
                            episode_results[queue_idx] = EpisodeResults()
                            self.episodes += 1
                        if sr.is_reset:
                            del last_seen_states[queue_idx]
                        else:
                            self.model.add_memory(old_state, sr.action_chosen, sr.reward, sr.observation, sr.done)
                    last_seen_states[queue_idx] = sr.observation
            self.learn_steps += self.model.potentially_learn()
            if (self.sim_steps % self.eval_interval < len(indices)) and (self.sim_steps > len(indices)):
                evaluated_episodes = self.evaluate()
                self.model.save_models(os.path.join(self.model_dir, 'latest'))
                if not self.callbacks.on_evaluation_end(self.sim_steps, self.learn_steps, evaluated_episodes=evaluated_episodes):
                    break
        self.callbacks.on_training_end()
        print("final:", self.sim_steps)
