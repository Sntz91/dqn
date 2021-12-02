import torch as T
import torch.nn as nn
import torch.nn.functional as F


def vanilla_dqn_loss(batch, policy_net, target_net, 
                     gamma, device, rescale_q_target=False):
    states, actions, rewards, dones, next_states = batch
    
    states_v = T.from_numpy(states).to(device)
    next_states_v = T.from_numpy(next_states).to(device)
    actions_v = T.from_numpy(actions).to(device)
    rewards_v = T.from_numpy(rewards).to(device)
    dones_v = T.BoolTensor(dones).to(device)

    state_action_values = policy_net(states_v)\
            .gather(1, actions_v.unsqueeze(-1))\
            .squeeze(1)

    with T.no_grad():
        next_state_values = target_net(next_states_v).max(1)[0]
        next_state_values[dones_v] = 0.0
    
    next_state_values = next_state_values.detach()
    expected_state_action_values = \
            (next_state_values * gamma) + rewards_v

    if rescale_q_target:
        expected_state_action_values = \
                _rescale_q_target(expected_state_action_values)

    return F.smooth_l1_loss(state_action_values,
                            expected_state_action_values)


def double_dqn_loss(batch, policy_net, target_net,
                   gamma, device, rescale_q_target=False): 
    states, actions, rewards, dones, next_states = batch

    states_v = T.from_numpy(states).to(device)
    next_states_v = T.from_numpy(next_states).to(device)
    actions_v = T.from_numpy(actions).to(device)
    rewards_v = T.from_numpy(rewards).to(device)
    dones_v = T.BoolTensor(dones).to(device)

    state_action_values = policy_net(states_v)\
            .gather(1, actions_v.unsqueeze(-1))\
            .squeeze(1)

    with T.no_grad():
        next_actions = policy_net(next_states_v).max(1)[1]
        next_state_values = target_net(next_states_v)\
                .gather(1, next_actions.unsqueeze(-1)).squeeze(-1)
        next_state_values[dones_v] = 0.0
    next_state_values = next_state_values.detach()
    expected_state_action_values = \
            (next_state_values * gamma)+ rewards_v
    if rescale_q_target:
        expected_state_action_values = \
                _rescale_q_target(expected_state_action_values)

    return F.smooth_l1_loss(state_action_values, 
                            expected_state_action_values)



def _rescale_q_target(val, eps=0.01):
    return T.sign(val) * (T.sqrt(T.abs(val) + 1) - 1) + eps*val
