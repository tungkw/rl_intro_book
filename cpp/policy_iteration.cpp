#include "policy_iteration.h"
#include <iostream>
#include <cmath>
using namespace std;


// finiteMDP::finiteMDP(int state_size, int action_size, int reward_size, vector<vector<vector<vector<float>>>> p_sr_sa)
finiteMDP::finiteMDP(int state_size, int action_size, int reward_size, float (*dynamics) (int s_, float r, int s, int a))
{
    this->state_size = state_size;
    this->action_size = action_size;
    this->reward_size = reward_size;

    // this->v_s = new float[this->state_size];
    // this->v_s = vector<float>(state_size);
    cout << "bug" << endl;
    this->q_sa = new float[state_size*action_size];
    // this->p_sr_sa = p_sr_sa;
    this->dynamics = dynamics;
    this->policy = new float[state_size*action_size];

    for(int i = 0 ; i < state_size; i++)
    {
        for(int j = 0 ; j < action_size; j++)
        {
            this->policy[i*action_size + j] = 1/action_size;
        }
    }
    cout << "bug" << endl;
}

void finiteMDP::state_value_iter()
{
    cout << "bug" << endl;
    float *v_last = new float[this->state_size];
    for(int i  = 0 ; i < this->state_size; i++) v_last[i] = this->v_s[i];

    float diff = 1;
    while(diff > 0.001)
    {
        for(int s = 0 ; s < this->state_size; s++)
        {
            float new_vs = 0;
            for(int a = 0 ; a < this->action_size; a++)
            {
                float new_q_sa = 0;
                for(int s_ = 0 ; s_ < this->state_size; s_++)
                {
                    for(int r = 0 ; r < reward_size; r++)
                    {
                        // new_q_sa += p_sr_sa[s][a][s_][r];
                        new_q_sa += dynamics(s_, r, s, a);
                    }
                }
                this->q_sa[s*action_size+a] = new_q_sa;
                new_vs += this->policy[s*action_size+a] * this->q_sa[s*action_size+a];
            }
            this->v_s[s] = new_vs;
        }

        diff = 0;
        for(int i  = 0 ; i < this->state_size; i++)
        {
            float d = abs(v_last[i] - this->v_s[i]);
            if(d > diff) diff = d;
        }
        cout << diff << endl;
        for(int i  = 0 ; i < this->state_size; i++) v_last[i] = this->v_s[i];
    }
    delete [] v_last;
}

void finiteMDP::policy_improvment()
{
    float *p_last = new float[this->state_size*this->action_size];
    for(int i  = 0 ; i < state_size * action_size; i++) p_last[i] = this->policy[i];

    float diff = 1;
    while(diff > 0.001)
    {
        for(int s = 0 ; s < state_size; s++)
        {
            int max_a = 0;
            float max_q = 0;
            int max_cnt = 0;
            for(int a = 0 ; a < action_size; a++)
            {
                float new_q_sa = this->q_sa[s*action_size+a];

                if(a==0)
                {
                    max_q = new_q_sa;
                    max_cnt = 1;
                }
                else
                {
                    if(new_q_sa > max_q)
                    {
                        max_a = a;
                        max_q = new_q_sa;
                        max_cnt = 1;
                    }
                    if(new_q_sa == max_q)
                    {
                        max_cnt += 1;
                    }
                }
            }
            for(int a = 0 ; a < action_size; a++)
            {
                float new_q_sa = this->q_sa[s*action_size+a];
                if(new_q_sa == max_q)
                {
                    this->policy[s*action_size+a] = 1/max_cnt;
                }
                else
                {
                    this->policy[s*action_size+a] = 0;
                }
            }
        }

        diff = 0;
        for(int i  = 0 ; i < state_size * action_size; i++)
        {
            float d = abs(p_last[i] - this->policy[i]);
            if(d > diff) diff = d;
        }
        cout << diff << endl;
        for(int i  = 0 ; i < state_size * action_size; i++) p_last[i] = this->policy[i];
    }
    
    delete [] p_last;
}


void finiteMDP::release()
{
    // delete [] this->v_s;
    delete [] this->q_sa;
    delete [] this->policy;
}