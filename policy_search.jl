using MuJoCo: Visualiser
using BSON: @save, @load
using MuJoCo
using LinearAlgebra
using Random
using Statistics
using Base.Threads
using Flux
using Flux: DataLoader

model = load_model("models/cartpole.xml")
data = init_data(model)


const POLICY_PATH = "cartpole_mppi.bson"


const K = 30    # num sample trajectories
const H = 100   # horizon
const λ = 1.0   # temperature
const Σ = 1.0   # control noise
const nx = 2 * model.nv
const nu = model.nu
const U_global = zeros(nu, H)



function running_cost(x_pos, theta, x_vel, theta_vel, control)
    cart_pos_cost = 1.0 * x_pos^2
    pole_pos_cost = 20.0 * (cos(theta) - 1.0)^2
    cart_vel_cost = 0.1 * x_vel^2
    pole_vel_cost = 0.1 * theta_vel^2
    ctrl_cost = 0.01 * control[1]^2
    return cart_pos_cost + pole_pos_cost + cart_vel_cost + pole_vel_cost + ctrl_cost
end

function terminal_cost(x_pos, theta, x_vel, theta_vel)
    return 10.0 * running_cost(x_pos, theta, x_vel, theta_vel, zeros(nu))
end

function rollout(m::Model, d::Data, U::Matrix{Float64}, noise::Array{Float64,3})
    costs = zeros(K)
    @threads for k in 1:K
        d_copy = init_data(m)
        d_copy.qpos .= d.qpos
        d_copy.qvel .= d.qvel
        cost = 0.0
        for t in 1:H
            d_copy.ctrl .= U[:, t] + noise[:, t, k]
            mj_step(m, d_copy)
            cost += running_cost(d_copy.qpos[1], d_copy.qpos[2],
                d_copy.qvel[1], d_copy.qvel[2], d_copy.ctrl)
        end
        costs[k] = cost + terminal_cost(d_copy.qpos[1], d_copy.qpos[2],
            d_copy.qvel[1], d_copy.qvel[2])
    end
    return costs
end

function mppi_step!(m::Model, d::Data)
    noise = randn(nu, H, K) * Σ
    costs = rollout(m, d, U_global, noise)
    β = minimum(costs)
    weights = exp.(-1 / λ * (costs .- β))
    weights ./= sum(weights)
    for t in 1:H
        U_global[:, t] .= U_global[:, t] + sum(weights[k] * noise[:, t, k] for k in 1:K)
    end
end

function mppi_controller!(m::Model, d::Data)
    mppi_step!(m, d)
    d.ctrl .= U_global[:, 1]
    U_global[:, 1:end-1] .= U_global[:, 2:end]
    U_global[:, end] .= 0.1 * U_global[:, end-1]
end

function collect_data(num_episodes=100, steps_per_episode=200)
    states = []
    actions = []


    for episode in 1:num_episodes
        data.qpos .= randn(model.nq) * 0.1
        data.qvel .= randn(model.nv) * 0.1

        for step in 1:steps_per_episode
            state = vcat(data.qpos, data.qvel)
            mppi_controller!(model, data)
            push!(states, state)
            push!(actions, copy(data.ctrl))
            mj_step(model, data)

        end

    end

    return hcat(states...), hcat(actions...)
end

# Neural Network Policy
policy = Chain(
    Dense(nx => 64, tanh),
    Dense(64 => 32, tanh),
    Dense(32 => nu)
)

# Training
function train_policy(states, actions, epochs=100)
    dataset = DataLoader((Float32.(states), Float32.(actions)), batchsize=128, shuffle=true)
    opt = Adam(1e-3)
    params = Flux.params(policy)
    losses = []

    for epoch in 1:epochs
        epoch_losses = []
        for (x, y) in dataset
            loss, grads = Flux.withgradient(() -> sum(abs2, policy(x) - y), params)
            push!(epoch_losses, loss)
            Flux.Optimise.update!(opt, params, grads)  # Corrected update line
        end

        avg_loss = mean(epoch_losses)
        push!(losses, avg_loss)

        if epoch % 10 == 0
            println("Epoch: $epoch, Average Loss: $avg_loss")
        end
    end

    @save POLICY_PATH saved_policy = policy
    println("saved policy to $POLICY_PATH")

    return losses
end

println("Collecting data...")
states, actions = collect_data()
println("Training policy...")
train_policy(states, actions)

# Test learned policy
function neural_controller!(m::Model, d::Data)
    state = Float32.(vcat(d.qpos, d.qvel))
    d.ctrl .= policy(state)
end

# Visualize learned policy
init_visualiser()
visualise!(model, data; controller=neural_controller!)
