#ifndef __POLICY_ITER
#define __POLICY_ITER
#include <vector>
using namespace std;

class finiteMDP
{
    int state_size;
    int action_size;
    int reward_size;
    vector<float> v_s;
    // float *v_s;
    float *q_sa;
    float *policy;
    // vector<vector<vector<vector<float>>>> p_sr_sa;
    float (*dynamics) (int s_, float r, int s, int a);
public:

    // finiteMDP(int state_size, int action_size, int reward_size, vector<vector<vector<vector<float>>>> p_sr_sa);
    finiteMDP(int state_size, int action_size, int reward_size, float (*dynamics) (int s_, float r, int s, int a));
    void state_value_iter();
    void policy_improvment();
    void release();
};

#endif