
using MuJoCo
using LinearAlgebra
using Random
using Statistics
using Base.Threads

model = load_model("models/cartpole.xml")
data = init_data(model)

const K = 30    # num sample trajectories
const H = 100   # horizon
const λ = 1.0   # temperature
const Σ = 1.0   # control noise for exploration

const nx = 2 * model.nv
const nu = model.nu

function cost(qpos, qvel, ctrl)
    cost = 0.0

    # Extract torso position and orientation
    torso_pos = qpos[1:3]  # x, y, z position
    torso_quat = qpos[4:7] # quaternion orientation

    # Desired forward velocity (x-direction)
    target_vel_x = 0.5  # meters per second
    current_vel_x = qvel[1]

    # Penalize deviation from target forward velocity
    cost += 1.0 * (current_vel_x - target_vel_x)^2

    # Penalize lateral velocity (y-direction)
    cost += 2.0 * qvel[2]^2

    # Penalize deviation from target height
    #cost += 3.0 * (torso_pos[3] - TARGET_HEIGHT)^2

    # Penalize torso rotation (try to keep it upright)
    # Convert quaternion to roll and pitch
    roll = atan(2(torso_quat[1] * torso_quat[2] + torso_quat[3] * torso_quat[4]),
        1 - 2(torso_quat[2]^2 + torso_quat[3]^2))
    pitch = asin(2(torso_quat[1] * torso_quat[3] - torso_quat[4] * torso_quat[2]))

    cost += 2.0 * (roll^2 + pitch^2)

    # Penalize excessive joint velocities
    cost += 0.1 * sum(qvel[7:end] .^ 2)

    # Penalize control effort
    cost += 0.01 * sum(ctrl .^ 2)

    return cost
end
function running_cost(x_pos, theta, x_vel, theta_vel, control)
    cart_pos_cost = 1.0 * x_pos^2
    pole_pos_cost = 20.0 * (cos(theta) - 1.0)^2  # Changed to use angle directly
    cart_vel_cost = 0.1 * x_vel^2
    pole_vel_cost = 0.1 * theta_vel^2
    ctrl_cost = 0.01 * control[1]^2
    return cart_pos_cost + pole_pos_cost + cart_vel_cost + pole_vel_cost + ctrl_cost
end


# makes it not kiss the corners
function terminal_cost(x_pos, theta, x_vel, theta_vel)
    return 10.0 * running_cost(x_pos, theta, x_vel, theta_vel, zeros(nu))
end

# init controls
const U_global = zeros(nu, H)

function rollout(m::Model, d::Data, U::Matrix{Float64}, noise::Array{Float64,3})
    costs = zeros(K)
    # thanks claude san for making this multi thread?
    @threads for k in 1:K
        d_copy = init_data(m)
        d_copy.qpos .= d.qpos
        d_copy.qvel .= d.qvel
        cost = 0.0
        for t in 1:H
            # Apply control with noise
            d_copy.ctrl .= U[:, t] + noise[:, t, k]
            mj_step(m, d_copy)
            # Extract state information
            x_pos = d_copy.qpos[1]
            theta = d_copy.qpos[2]
            x_vel = d_copy.qvel[1]
            theta_vel = d_copy.qvel[2]
            # Compute running cost
            cost += running_cost(x_pos, theta, x_vel, theta_vel, d_copy.ctrl)
        end
        # Add terminal cost
        costs[k] = cost + terminal_cost(
            d_copy.qpos[1], d_copy.qpos[2],
            d_copy.qvel[1], d_copy.qvel[2]
        )
    end
    return costs
end

function mppi_step!(m::Model, d::Data)
    noise = randn(nu, H, K) * Σ
    costs = rollout(m, d, U_global, noise)
    β = minimum(costs)
    weights = exp.(-1 / λ * (costs .- β))
    weights ./= sum(weights)
    # update controls
    for t in 1:H
        U_global[:, t] .= U_global[:, t] + sum(weights[k] * noise[:, t, k] for k in 1:K)
    end
end

function mppi_controller!(m::Model, d::Data)
    mppi_step!(m, d)
    d.ctrl .= U_global[:, 1]
    # shifting controls
    U_global[:, 1:end-1] .= U_global[:, 2:end]
    U_global[:, end] .= 0.1 * U_global[:, end-1]  # Smaller decay factor
end

# woohoooo
init_visualiser()
visualise!(model, data; controller=mppi_controller!)
