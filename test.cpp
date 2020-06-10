#include <iostream>
#include <cmath>
#include <vector>
#include "policy_iteration.h"
using namespace std;

int fac(int n)
{
    int result = 1;
    for(int i = 1 ; i <= n;i++)
    {
        result *= i;
    }
    return result;
}

float *poisson_m;
float poisson(int lambda, int n)
{
    return pow(lambda, n) / fac(n) * exp(-lambda);
}

float dynamics(int s_, float r, int s, int a)
{
    int l_size = 20+1;

    int location1 = s/l_size - a;
    int location2 = s%l_size + a;
    
    int location1_ = s_/l_size;
    int location2_ = s_%l_size;

    // possible request num
    float r_request = (-10+r) + abs(-5+a) * 2;
    if(abs(r_request - int(r_request)) > 0.0001 || int(r_request) % 10 !=0)
        return 0;
    int requests = int(r_request) / 10;
    if(requests > (location1 + location2))
        return 0;

    // location1 - requests1 + return1 = location1_
    float p = 0;
    int location1_diff = location1_ - location1;
    int location2_diff = location2_ - location2;
    for(int requests1 = 0; requests1 <= location1; requests1++)
    {
        int requests2 = requests - requests1;
        if(requests2 > location2 || requests2 < 0) continue;
        // cout << "requests1 " << requests1 <<  " " << "requests2 " << requests2 << endl;

        int return1 = location1_diff + requests1;
        int return2 = location2_diff + requests2;
        if(return1 < 0 || return2 < 0) continue;
        float p_;
        p_ = poisson_m[0*21+requests1];
        p_ *= poisson_m[1*21+requests2];
        p_ *= poisson_m[2*21+return1];
        p_ *= poisson_m[3*21+return2];
        p += p_;
    }
    return p;
}

int main()
{
    int lambda[4] = {3,4,3,2};
    poisson_m = new float[4*20];
    for(int i = 0 ; i < 4;i++)
    {
        for(int j = 0 ; j < 21;j++)
        {
            poisson_m[i*21+j] = poisson(lambda[i], j);
        }
    }

    int l_size = 20+1;
    int a_size = 5;

    int state_size = l_size * l_size;
    int action_size = 2*a_size + 1;
    int reward_size = 400+10+1;

    // int location1 = 10;
    // int location2 = 5;
    // int s = location1*l_size + location2;

    // int p_size = state_size * action_size * state_size * reward_size;
    // vector<vector<vector<vector<float>>>> p_sr_sa;
    
    // for(int s = 0 ; s < state_size; s++)
    // {
    //     cout << s << endl;
    //     vector<vector<vector<float>>> p_a(action_size);
        // for(int a = 0; a < action_size; a++)
        // {
        //     vector<vector<float>> p_s_;
        //     float t = 0;
        //     for(int s_ = 0; s_ < state_size; s_++)
        //     {
        //         vector<float> p_r;
        //         for(int r = 0; r < reward_size; r++)
        //         {
        //             float tmp = dynamics(s_, r, s, a);
        //             p_r.push_back(tmp);
        //             t += tmp;
        //         }
        //         p_s_.push_back(p_r);
        //     }
        //     for(int s_ = 0; s_ < state_size; s_++)
        //     {
        //         for(int r = 0; r < reward_size; r++)
        //         {
        //             p_s_[s_][r] /= t;
        //         }
        //     }
        //     p_a.push_back(p_s_);
        // }
    //     p_sr_sa.push_back(p_a);
    // }


    // finiteMDP mdp(state_size, action_size, reward_size, p_sr_sa);
    finiteMDP mdp(state_size, action_size, reward_size, dynamics);
    mdp.state_value_iter();
    // mdp.policy_improvment();
    mdp.release();
}