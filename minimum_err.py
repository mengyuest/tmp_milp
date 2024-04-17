import numpy as np
import gurobipy as gp

def nn_addMVar_addConstr(model, nn_input, weights, operator, thres, BIG_M, prefix=""):
    bs = nn_input.shape[0]
    
    a_list = []
    z_list = []
    
    n_layers = len(weights) // 2 - 1
    prev_layer_input = nn_input
    for layer_i in range(n_layers):
        n_hidden_units = weights[layer_i*2+1].shape[0]
        nn_a = model.addMVar((bs, n_hidden_units), vtype=gp.GRB.BINARY, name="%s_a%d"%(prefix, layer_i))
        nn_z = model.addMVar((bs, n_hidden_units), lb=-0, ub=BIG_M, name="%s_z%d"%(prefix, layer_i))
        a_list.append(nn_a)
        z_list.append(nn_z)
        
        # model linear layer forward pass with ReLU activation
        # z = ReLU(wx+b)
        nn_w = weights[layer_i*2]
        nn_b = weights[layer_i*2+1]
        model.addConstr(nn_z <= prev_layer_input @ nn_w.T + nn_b + BIG_M * (1-nn_a))
        model.addConstr(nn_z >= prev_layer_input @ nn_w.T + nn_b)
        model.addConstr(nn_z <= BIG_M * nn_a)
        
        prev_layer_input = nn_z        
    
    # last layer
    nn_w = weights[-2]
    nn_b = weights[-1]
    assert operator in [">=", "<=", "=="]
    if operator==">=":
        model.addConstr(prev_layer_input @ nn_w.T + nn_b >= thres)
    elif operator=="<=":
        model.addConstr(prev_layer_input @ nn_w.T + nn_b <= thres)
    else:
        model.addConstr(prev_layer_input @ nn_w.T + nn_b == thres)
    return a_list, z_list


def milp_test():
    # Create a new model
    model = gp.Model("example")
        
    network_weights = np.load("network_weights.npz", allow_pickle=True)
    
    # hyper-param & configs
    nt = 10
    u_max = 0.1
    
    goal0_x = 3
    goal0_y = -2
    goal1_x = -3
    goal1_y = 2
    
    BIG_M = 1000
    coll_thres = 0.15
    reach_thres = 0.3   
    
    # Create variables
    # create initial variables
    q0s = model.addMVar((nt+1, 4), lb=-np.pi/2, ub=np.pi/2, name="q0s")
    q1s = model.addMVar((nt+1, 4), lb=-np.pi/2, ub=np.pi/2, name="q1s")
    
    u0s = model.addMVar((nt+1, 4), lb=-u_max, ub=u_max, name="u0s")
    u1s = model.addMVar((nt+1, 4), lb=-u_max, ub=u_max, name="u1s")
    
    nn_input0 = model.addMVar((nt+1, 4+4+4+4), lb=-BIG_M, ub=BIG_M, name="nn_input0")
    nn_input1_0 = model.addMVar((1, 1+4+2), lb=-BIG_M, ub=BIG_M, name="nn_input1_0")
    nn_input1_1 = model.addMVar((1, 1+4+2), lb=-BIG_M, ub=BIG_M, name="nn_input1_1")
    
    # Set objective
    model.setObjective(0, sense=gp.GRB.MINIMIZE)
    
    # Add constraints
    # initial state
    model.addConstr(q0s[0, :]==0)
    model.addConstr(q1s[0, :]==0)
    
    # dynamics constraint
    model.addConstr(q0s[1:, :]-q0s[:-1,:] == u0s[:-1, :])
    model.addConstr(u0s[-1, :] == 0)
    model.addConstr(q1s[1:, :]-q1s[:-1,:] == u1s[:-1, :])
    model.addConstr(u1s[-1, :] == 0)
    
    # form the neural network input (concatenation)
    model.addConstr(nn_input0[:, 0:4] == q0s)
    model.addConstr(nn_input0[:, 4:8] == u0s)
    model.addConstr(nn_input0[:, 8:12] == q1s)
    model.addConstr(nn_input0[:, 12:16] == u1s)
    
    model.addConstr(nn_input1_0[:, 0] == 0)
    model.addConstr(nn_input1_0[:, 1:5] == q0s[-1:, :])
    model.addConstr(nn_input1_0[:, 5] == goal0_x)
    model.addConstr(nn_input1_0[:, 6] == goal0_y)
    
    model.addConstr(nn_input1_1[:, 0] == 1)
    model.addConstr(nn_input1_1[:, 1:5] == q1s[-1:, :])
    model.addConstr(nn_input1_1[:, 5] == goal1_x)
    model.addConstr(nn_input1_1[:, 6] == goal1_y)
    
    # neural constraints
    a0_list, z0_list = nn_addMVar_addConstr(model, nn_input0, network_weights["nn0"], ">=", coll_thres, BIG_M, prefix="nn0")
    a1_list, z1_list = nn_addMVar_addConstr(model, nn_input1_0, network_weights["nn1"], "<=", reach_thres, BIG_M, prefix="nn1_0")
    a2_list, z2_list = nn_addMVar_addConstr(model, nn_input1_1, network_weights["nn1"], "<=", reach_thres, BIG_M, prefix="nn1_1")
    
    # Optimize model
    model.optimize()

    # print out necessary info
    print("Optimal solution:")
    print("Optimal objective value:", model.objVal)
    print(q0s.X)
    print(q1s.X)
    

if __name__ == "__main__":
    milp_test()