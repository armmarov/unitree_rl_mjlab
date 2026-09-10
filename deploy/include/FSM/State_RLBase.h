// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "FSMState.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include "isaaclab/envs/mdp/terminations.h"

class State_RLBase : public FSMState
{
public:
    State_RLBase(int state_mode, std::string state_string);
    
    void enter()
    {
        // set gain
        // BUGFIX (see unitreerobotics/unitree_rl_mjlab#57): this loop used to write kp/kd/dq/tau
        // via the raw loop index `i` instead of joint_ids_map[i], while run() (below, in the
        // .cpp) correctly maps q() through joint_ids_map[i]. For any robot with a non-identity
        // joint_ids_map (R1: gaps at 14/20/21), this meant position targets landed on the right
        // joint but gains landed on a DIFFERENT joint -- several arm joints got another joint's
        // gain, and the last few joints (whose loop index exceeds the highest real slot count
        // touched here) never got written at all, silently keeping whatever gain the previous
        // FSM state had left behind. Confirmed present verbatim in this exact file; fixed to
        // route through joint_ids_map[i] consistently with run()'s q() write, and with
        // isaaclab::mdp::joint_pos_rel's own state read (unitree_articulation.h uses
        // motor_state()[joint_ids_map[i]]).
        for (int i = 0; i < env->robot->data.joint_stiffness.size(); ++i)
        {
            const int slot = static_cast<int>(env->robot->data.joint_ids_map[i]);
            lowcmd->msg_.motor_cmd()[slot].kp() = env->robot->data.joint_stiffness[i];
            lowcmd->msg_.motor_cmd()[slot].kd() = env->robot->data.joint_damping[i];
            lowcmd->msg_.motor_cmd()[slot].dq() = 0;
            lowcmd->msg_.motor_cmd()[slot].tau() = 0;
        }

        env->robot->update();
        // Start policy thread
        policy_thread_running = true;
        policy_thread = std::thread([this]{
            using clock = std::chrono::high_resolution_clock;
            const std::chrono::duration<double> desiredDuration(env->step_dt);
            const auto dt = std::chrono::duration_cast<clock::duration>(desiredDuration);

            // Initialize timing
            auto sleepTill = clock::now() + dt;
            env->reset();

            while (policy_thread_running)
            {
                env->step();

                // Sleep
                std::this_thread::sleep_until(sleepTill);
                sleepTill += dt;
            }
        });
    }

    void run();
    
    void exit()
    {
        policy_thread_running = false;
        if (policy_thread.joinable()) {
            policy_thread.join();
        }
    }

private:
    std::unique_ptr<isaaclab::ManagerBasedRLEnv> env;

    std::thread policy_thread;
    bool policy_thread_running = false;
};

REGISTER_FSM(State_RLBase)
