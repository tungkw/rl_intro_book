# policy iteration

policy evaluation + policy improvement

- Agent
  
  element: 
  - ```v[float]```
  - ```p[int]```

  func:
  - ```state_value(state)```->```value```
  - ```policy(state, action)```->```$\pi$(a|s)```
  - ```print()```