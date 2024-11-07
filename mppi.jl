using MuJoCo
using LinearAlgebra
using Random
using Statistics
using Base.Threads

model = load_model("models/unitree_go1/scene.xml")
data = init_data(model)

const K = 100  # Number of samples
const H = 30    # Horizon length
const λ = 0.2   # Temperature
const Σ = 0.3   # Noise standard deviation
const nx = model.nq + model.nv
const nu = model.nu
const U_global = zeros(nu, H)

function get_gait_reference(t)
    # Simple trotting pattern (diagonal legs move together)
    freq = 2.0  # Gait frequency
    phase = (t * freq) % 1.0

    if phase < 0.5
        return [0.0, 0.9, -1.8, # FR
            0.0, 0.6, -1.2, # FL
            0.0, 0.6, -1.2, # RR
            0.0, 0.9, -1.8] # RL
    else
        return [0.0, 0.6, -1.2, # FR
            0.0, 0.9, -1.8, # FL
            0.0, 0.9, -1.8, # RR
            0.0, 0.6, -1.2] # RL
    end
end

function cost(qpos, qvel, ctrl)
    # Weights
    w_height = 100.0    # Height tracking
    w_orientation = 100.0 # Orientation tracking
    w_vel_x = 50.0      # Forward velocity tracking
    w_vel_y = 100.0     # Lateral velocity penalty
    w_vel_z = 50.0      # Vertical velocity penalty
    w_ctrl = 0.1        # Control cost
    w_ctrl_smooth = 1.0 # Control smoothness
    w_foot_force = 10.0 # Foot force distribution
    w_gait = 10.0

    # Target states
    target_height = 0.35     # Desired standing height
    target_vel_x = 0.5      # Desired forward velocity (m/s)

    gait_ref = get_gait_reference(t)
    gait_tracking_cost = w_gait * sum((ctrl - gait_ref) .^ 2)


    # Extract current states
    pos = qpos[1:3]         # Base position
    quat = qpos[4:7]        # Base orientation quaternion
    joint_pos = qpos[8:end] # Joint positions
    vel = qvel[1:3]         # Base linear velocity
    ang_vel = qvel[4:6]     # Base angular velocity

    # Cost components
    # 1. Height tracking
    height_cost = w_height * (pos[3] - target_height)^2

    # 2. Orientation tracking (keep trunk level)
    orientation_cost = w_orientation * (sum(quat[2:4] .^ 2))  # Penalize non-zero roll and pitch

    # 3. Velocity tracking
    vel_cost = w_vel_x * (vel[1] - target_vel_x)^2 +  # Track forward velocity
               w_vel_y * vel[2]^2 +                    # Penalize lateral velocity
               w_vel_z * vel[3]^2                      # Penalize vertical velocity

    # 4. Angular velocity penalty (stability)
    ang_vel_cost = w_orientation * sum(ang_vel .^ 2)

    # 5. Control costs
    ctrl_cost = w_ctrl * sum(ctrl .^ 2)

    # 6. Symmetric leg positions (encourage coordinated movement)
    # Define diagonal pairs
    FL_FR = (joint_pos[4:6] + joint_pos[1:3]) .^ 2
    RL_RR = (joint_pos[10:12] + joint_pos[7:9]) .^ 2
    symmetry_cost = sum(FL_FR) + sum(RL_RR)

    # Total cost
    total_cost = height_cost +
                 orientation_cost +
                 vel_cost +
                 ang_vel_cost +
                 ctrl_cost +
                 0.1 * symmetry_cost +
                 gait_tracking_cost

    return total_cost
end



function old_cost(qpos, qvel, ctrl)
    # Weights
    w_pos = 100.0    # Position tracking
    w_height = 100.0 # Height tracking
    w_vel = 10.0      # Velocity tracking
    w_ctrl = 0.1     # Control cost

    # Target state
    target_height = 0.35  # Desired standing height
    target_vel_x = 0.5   # Desired forward velocity (m/s)

    # Current states
    current_pos = qpos[1:3]
    current_vel = qvel[1:3]

    # Costs
    # 1. Track desired height
    height_cost = w_height * (current_pos[3] - target_height)^2

    # 2. Track desired forward velocity
    vel_cost = w_vel * (current_vel[1] - target_vel_x)^2

    # 3. Penalize lateral motion
    lateral_cost = w_pos * (current_pos[2]^2 + current_vel[2]^2)

    # 4. Control cost
    ctrl_cost = w_ctrl * sum(ctrl .^ 2)

    # Total cost
    total_cost = height_cost + vel_cost + lateral_cost + ctrl_cost

    return total_cost
end

function rollout(m::Model, d::Data, U::Matrix{Float64}, noise::Array{Float64,3})
    costs = zeros(K)
    @threads for k in 1:K
        d_copy = init_data(m)
        d_copy.qpos .= d.qpos
        d_copy.qvel .= d.qvel
        cost_sum = 0.0

        for t in 1:H
            current_ctrl = vec(U[:, t] + noise[:, t, k])
            d_copy.ctrl .= clamp.(current_ctrl, -10.0, 10.0)
            mj_step(m, d_copy)
            cost_sum += cost(d_copy.qpos, d_copy.qvel, d_copy.ctrl)
        end
        costs[k] = cost_sum
    end
    return costs
end

function mppi_update!(m::Model, d::Data)
    noise = randn(nu, H, K) * Σ
    costs = rollout(m, d, U_global, noise)

    β = minimum(costs)
    weights = exp.(-1 / λ * (costs .- β))
    weights ./= sum(weights) + 1e-10

    for t in 1:H
        weighted_noise = sum(weights[k] * noise[:, t, k] for k in 1:K)
        U_global[:, t] .= clamp.(U_global[:, t] + weighted_noise, -10.0, 10.0)
    end

    d.ctrl .= U_global[:, 1]
    U_global[:, 1:end-1] .= U_global[:, 2:end]
    U_global[:, end] .= 0.0
end


# Initialize visualization
init_visualiser()
data.qpos .= [0.0, 0.0, 0.35, 1.0, 0.0, 0.0, 0.0, 0.0, 0.9, -1.8, 0.0, 0.9, -1.8, 0.0, 0.9, -1.8, 0.0, 0.9, -1.8]

visualise!(model, data; controller=mppi_update!)
