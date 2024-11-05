using MuJoCo
using LinearAlgebra
using Random
using Statistics
using Base.Threads


model = load_model("models/unitree_go1/scene.xml")
data = init_data(model)

const K = 50
const H = 30
const λ = 0.1
const Σ = 0.1
const nx = model.nq + model.nv
const nu = model.nu
const U_global = zeros(nu, H)

# Target height for the torso site
const TARGET_HEIGHT = 0.32

function cost(qpos, qvel, ctrl)
    cost = 0.0

    # Get the torso site position
    torso_pos = data.site_xpos[1, :]  # Assuming it's the first site

    # Height error using site position
    height_error = (torso_pos[3] - TARGET_HEIGHT)^2
    cost += 500.0 * height_error

    # Penalize horizontal movement
    xy_movement = torso_pos[1]^2 + torso_pos[2]^2
    cost += 200.0 * xy_movement

    # Velocity costs
    cost += 200.0 * sum(qvel.^2)

    # Control cost
    cost += 0.1 * sum(ctrl.^2)

    # Extra penalty for angular velocities
    angular_vel = qvel[4:6]
    cost += 6000.0 * sum(angular_vel.^2)

    # Penalty for foot movement
    foot_joints_vel = qvel[7:end]
    cost += 50.0 * sum(foot_joints_vel.^2)

    return cost
end

# Rest of the code remains the same...
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
