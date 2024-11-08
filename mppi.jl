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

function cost(qpos, qvel, ctrl)
    # Weights
    w_pos = 100.0    # Position tracking
    w_height = 100.0 # Height tracking
    w_vel = 10.0      # Velocity tracking
    w_ctrl = 0.1     # Control cost

    # Target state
    target_height = 0.5  # Desired standing height
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


init_visualiser()
visualise!(model, data; controller=mppi_update!)
