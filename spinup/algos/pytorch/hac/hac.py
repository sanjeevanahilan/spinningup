import numpy as np
import torch
import gym
import time
import spinup.algos.pytorch.hac.core as core
from spinup.utils.logx import EpochLogger


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for DDPG agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in batch.items()}


def hac(env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0,
         steps_per_epoch=4000, epochs=100, replay_size=int(1e6), gamma=0.99,
         polyak=0.995, pi_lr=1e-3, q_lr=1e-3, batch_size=100, start_steps=10000,
         update_after=1000, update_every=50, act_noise=0.1, num_test_episodes=10,
         max_ep_len=1000, logger_kwargs=dict(), save_freq=1):
    """
    Deep Deterministic Policy Gradient (DDPG)


    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with an ``act``
            method, a ``pi`` module, and a ``q`` module. The ``act`` method and
            ``pi`` module should accept batches of observations as inputs,
            and ``q`` should accept a batch of observations and a batch of
            actions as inputs. When called, these should return:

            ===========  ================  ======================================
            Call         Output Shape      Description
            ===========  ================  ======================================
            ``act``      (batch, act_dim)  | Numpy array of actions for each
                                           | observation.
            ``pi``       (batch, act_dim)  | Tensor containing actions from policy
                                           | given observations.
            ``q``        (batch,)          | Tensor containing the current estimate
                                           | of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object
            you provided to DDPG.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs)
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs to run and train agent.

        replay_size (int): Maximum length of replay buffer.

        gamma (float): Discount factor. (Always between 0 and 1.)

        polyak (float): Interpolation factor in polyak averaging for target
            networks. Target networks are updated towards main networks
            according to:

            .. math:: \\theta_{\\text{targ}} \\leftarrow
                \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

            where :math:`\\rho` is polyak. (Always between 0 and 1, usually
            close to 1.)

        pi_lr (float): Learning rate for policy.

        q_lr (float): Learning rate for Q-networks.

        batch_size (int): Minibatch size for SGD.

        start_steps (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration.

        update_after (int): Number of env interactions to collect before
            starting to do gradient descent updates. Ensures replay buffer
            is full enough for useful updates.

        update_every (int): Number of env interactions that should elapse
            between gradient descent updates. Note: Regardless of how long
            you wait between updates, the ratio of env steps to gradient steps
            is locked to 1.

        act_noise (float): Stddev for Gaussian exploration noise added to
            policy at training time. (At test time, no noise is added.)

        num_test_episodes (int): Number of episodes to test the deterministic
            policy at the end of each epoch.

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    torch.manual_seed(seed)
    np.random.seed(seed)

    env, test_env = env_fn(), env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape[0]

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = env.action_space.high[0]

    # obs bounds and offset
    # obs_bounds = np.array([0.9, 0.07])
    # state_bounds = torch.FloatTensor(state_bounds_np.reshape(1, -1)).to(device)
    # state_offset =  np.array([-0.3, 0.0])
    # state_offset = torch.FloatTensor(state_offset.reshape(1, -1)).to(device)
    # state_clip_low = np.array([-1.2, -0.07])
    # state_clip_high = np.array([0.6, 0.07])

    # Create actor-critic module and target networks
    k_level = 1
    acs = [actor_critic(obs_dim[0], act_dim, act_limit, **ac_kwargs)]
    hac_agents = [core.DDPG(acs[0], pi_lr, q_lr, gamma, polyak, logger)]
    hac_buffers = [ReplayBuffer(obs_dim=tuple([obs_dim[0]]), act_dim=act_dim, size=replay_size)]
    for i in range(1, k_level):
        if i == k_level-1:
            od = obs_dim[0]
        else:
            od = 2*obs_dim[0]
        acs.append(actor_critic(od, obs_dim[0], act_limit, **ac_kwargs))
        hac_agents.append(core.DDPG(acs[i], pi_lr, q_lr, gamma, polyak, logger))
        hac_buffers.append(ReplayBuffer(obs_dim=tuple([od]), act_dim=obs_dim, size=replay_size))

    # agent = core.DDPG(ac, pi_lr, q_lr, gamma, polyak, logger)
    # threshold = np.array([0.01, 0.02])
    class HAC:
        def __init__(self, hac_agents, dur, threshold, logger):
            self.hac_agents = hac_agents
            # self.exploration_noise = exploration_noise
            self.k_level = len(hac_agents)
            self.dur = dur
            self.threshold = 0.1
            self.timestep = 0
            self.logger = logger
            self.ep_ret = 0
            self.ep_len = 0

        def check_goal(self, obs, goal):
            for i in range(obs.shape[0]):
                if abs(obs[i] - goal[i]) > self.threshold:
                    return 0
            return 1

        # def update(self, data_agents):
        #     for data in data_agents:
        #         self.hac_agents[i].update(data)

        def run(self, env, i_level, hac_buffers, o, goal=()):
                # Until start_steps have elapsed, randomly sample actions
                # from a uniform distribution for better exploration. Afterwards,
                # use the learned policy (with some noise, via act_noise).

                # Step the env
                for _ in range(self.dur):
                    og = np.concatenate((o, goal))
                    if self.timestep > 1000:
                        a = self.hac_agents[i_level].get_action(og, act_noise)
                    else:
                        a = env.action_space.sample()
                    if i_level > 0:
                        o2, r, d = self.run(env, i_level-1,  hac_buffers, o, goal=a)
                    else:
                        o2, r, d, _ = env.step(a)
                        self.timestep += 1
                        self.ep_ret += r
                        self.ep_len += 1
                    if i_level < k_level-1:
                        goal_achieved = self.check_goal(o2, goal)
                    else:
                        goal_achieved = r+1

                    og2 = np.concatenate((o2, goal))

                    # r = -1 if goal_achieved else 0

                    # Store experience to replay buffer
                    hac_buffers[i_level].store(og, a, r, og2, d)

                    # Super critical, easy to overlook step: make sure to update
                    # most recent observation!
                    o = o2

                    # # Ignore the "done" signal if it comes from hitting the time
                    # # horizon (that is, when it's an artificial terminal signal
                    # # that isn't based on the agent's state)
                    # d = False if ep_len == max_ep_len else d
                    if agent.ep_len == max_ep_len or d:
                        break

                return o2, r, d


    # Prepare for interaction with environment
    agent = HAC(hac_agents, dur=1, threshold=10, logger=logger)
    total_steps = steps_per_epoch * epochs
    start_time = time.time()
    o, agent.ep_ret, ep_len = env.reset(), 0, 0
    delta_update, delta_epoch = 0, 0
    # Main loop: collect experience in env and update/log each epoch
    while agent.timestep < total_steps:
        t0 = agent.timestep
        o2, r, d, = agent.run(env, len(hac_agents)-1, hac_buffers, o)
        o = o2
        t = agent.timestep
        # ep_len += t-t0
        delta_update += t-t0
        delta_epoch += t-t0

        # End of trajectory handling

        if d or (agent.ep_len == max_ep_len):
            logger.store(EpRet=agent.ep_ret, EpLen=ep_len)
            o, agent.ep_ret, agent.ep_len = env.reset(), 0, 0
    #
        # Update handling
        if t >= update_after and delta_update > update_every:
            delta_update = 0
            for _ in range(update_every):
                for i, replay_buffer in enumerate(hac_buffers):
                    batch = replay_buffer.sample_batch(batch_size)
                    agent.hac_agents[i].update(data=batch)

        # End of epoch handling
        if (delta_epoch + 1) > steps_per_epoch:
            delta_epoch = 0
            epoch = (t + 1) // steps_per_epoch

            # Save model
            if (epoch % save_freq == 0) or (epoch == epochs):
                logger.save_state({'env': env}, None)

        # # Test the performance of the deterministic version of the agent.
        # test_agent()
#
            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            # logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            # logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', t)
            logger.log_tabular('QVals', with_min_and_max=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossQ', average_only=True)
            logger.log_tabular('Time', time.time() - start_time)
            logger.dump_tabular()
    # # Set up model saving
    # logger.setup_pytorch_saver(acs[-1])

    # def test_agent():
    #     for j in range(num_test_episodes):
    #         o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
    #         while not (d or (ep_len == max_ep_len)):
    #             # Take deterministic actions at test time (noise_scale=0)
    #             o, r, d, _ = test_env.step(agent.get_action(o, 0))
    #             ep_ret += r
    #             ep_len += 1
    #         logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)




if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='ddpg')
    args = parser.parse_args()

    from spinup.utils.run_utils import setup_logger_kwargs

    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    hac(lambda: gym.make(args.env), actor_critic=core.MLPActorCritic,
         ac_kwargs=dict(hidden_sizes=[args.hid] * args.l),
         gamma=args.gamma, seed=args.seed, epochs=args.epochs,
         logger_kwargs=logger_kwargs)
