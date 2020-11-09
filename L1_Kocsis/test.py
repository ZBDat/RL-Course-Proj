import numpy as np


def create_transition_matrix(reward_map: np.ndarray):
    directions = np.array([[-1, 0],
                           [1, 0],
                           [0, -1],
                           [0, 1]])

    direction_movement_probs = np.array([[0.8, 0, 0.1, 0.1],
                                         [0, 0.8, 0.1, 0.1],
                                         [0.1, 0.1, 0.8, 0],
                                         [0.1, 0.1, 0, 0.8]])

    index_tensor = np.indices(reward_map.shape).transpose(1, 2, 0)

    movement_tensor = list()
    for direction_vector in directions:
        # Generate indices pointing to the next block
        movement_tensor_in_direction = index_tensor + direction_vector[None, None:]

        # Check environment borders
        movement_tensor_in_direction = \
            np.minimum(
                np.maximum(movement_tensor_in_direction,
                           0),
                np.array(reward_map.shape)[None, None, :] - 1)

        # Check obstacles
        mask_bumped_into_obstacle = np.isnan(
            reward_map[tuple(movement_tensor_in_direction.transpose(2, 0, 1).tolist())])
        movement_tensor_in_direction[mask_bumped_into_obstacle] = index_tensor[mask_bumped_into_obstacle]

        # Set the transition
        movement_tensor.append(movement_tensor_in_direction)

    movement_tensor = np.array(movement_tensor)

    # Transition matrix:
    #  dim 0: action   - direction to move
    #  dim 1: state(0) - coordinate of the current state
    #  dim 2: state(1) - coordinate of the current state
    #  dim 3: next(0)  - coordinate of the next state
    #  dim 4: next(1)  - coordinate of the next state
    #  dim 5: prob     - probability of the next state
    transition_tensor = np.zeros((4, *reward_map.shape, *reward_map.shape))

    for direction in range(transition_tensor.shape[0]):
        for curr_x in range(transition_tensor.shape[1]):
            for curr_y in range(transition_tensor.shape[2]):
                for resulting_dir in range(transition_tensor.shape[0]):
                    transition_tensor[
                        tuple([direction, curr_x, curr_y] + movement_tensor[resulting_dir, curr_x, curr_y].tolist())] += \
                    direction_movement_probs[direction, resulting_dir]
    return transition_tensor, index_tensor


def exercise_1(reward_map: np.ndarray, transition_tensor: np.ndarray):
    """
    Implements the policy iteration for the given reward_map
    :param reward_map: The reward map of the environment
    :param transition_tensor: The transition tensor of the environment
    """
    index_tensor = np.indices(reward_map.shape).transpose(1, 2, 0)

    # Parameter definitions
    threshold = 0.01
    gamma = 0.9

    print("Exercise 1: Policy Iteration")
    policy_map = np.random.randint(1, 4, reward_map.shape)

    value_map = np.zeros_like(reward_map)

    cycle = 0
    maximal_change = threshold
    while not maximal_change < threshold:
        previous_value_map = value_map
        # Policy evaluation
        probs_given_policy = transition_tensor[
            tuple(np.append(policy_map[..., None], index_tensor, axis=2).transpose((2, 0, 1)).tolist())]
        value_map = reward_map + gamma * np.tensordot(probs_given_policy, np.nan_to_num(value_map))

        # Policy improvement
        # Note: the reward_map and gamma is independent of the action
        policy_map = np.argmax(np.tensordot(transition_tensor, np.nan_to_num(value_map)), axis=0)
        cycle += 1
        maximal_change = abs(np.max(np.nan_to_num(value_map) - np.nan_to_num(previous_value_map)))
        print(f"\r{cycle}", end="")

    # Change the notation back for the action
    policy_map += 1
    policy_map = policy_map.astype("float")
    policy_map[np.isnan(reward_map)] = np.NaN

    print(f"\rPolicy iteration finished after {cycle} iteration!")
    print(f"Value map:\n{value_map}\n")
    print(f"Policy map:\n{policy_map}\n")


def exercise_2(reward_map: np.ndarray, transition_tensor: np.ndarray):
    """
    Implements the action-value iteration for the given reward_map
    :param reward_map: The reward map of the environment
    :param transition_tensor: The transition tensor of the environment
    """
    # Parameter definitions
    threshold = 0.01
    gamma = 0.9

    print("Exercise 2: Action-value Iteration")
    # Action value tensor:
    #  dim 0: action   - direction to move
    #  dim 1: state(0) - coordinate of the current state
    #  dim 2: state(1) - coordinate of the current state
    #  dim 3: value    - the value of the given action
    action_value_map = np.zeros((4, *reward_map.shape))

    cycle = 0
    maximal_change = threshold
    while not maximal_change < threshold:
        previous_action_value_map = action_value_map
        # Evaluation and improvement
        approx_value_map = np.max(action_value_map, axis=0)
        action_value_map = reward_map + gamma * np.tensordot(transition_tensor, np.nan_to_num(approx_value_map))

        cycle += 1
        maximal_change = abs(np.max(np.nan_to_num(action_value_map) - np.nan_to_num(previous_action_value_map)))
        print(f"\r{cycle}", end="")

    policy_map = np.argmax(action_value_map, axis=0).astype("float") + 1
    policy_map[np.isnan(reward_map)] = np.NaN

    print(f"\rAction-value iteration finished after {cycle} iteration!")
    for idx in range(4):
        print(f"Action-value map for direction {idx + 1}:\n{action_value_map[idx]}\n")

    print(f"Policy map:\n{policy_map}\n")


if __name__ == "__main__":
    reward_map = np.array([[0, 0, 0, 1],
                           [0, np.NaN, 0, -100],
                           [0, 0, 0, 0]])

    transition_tensor, index_tensor = create_transition_matrix(reward_map)

    exercise_1(reward_map, transition_tensor)
    print()
    exercise_2(reward_map, transition_tensor)
