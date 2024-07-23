- t_coh seems to have an impact only in the distillation cases
- int((2/parameters["p_swap"])**(swaps+dists) * (1/parameters["p_gen"])) seems to be not bad approximation

p_gen: 0.1, p_swap: 0.1, w0: 0.9, t_coh: 10000

Protocol: 0, 
        Analytical Bound: 20000, Experimental Trunc: 670

Protocol: (1, 0), 
        Analytical Bound: 22200, Experimental Trunc: 1020

Protocol: (1, 1, 0), 
        Analytical Bound: 23900, Experimental Trunc: 1448

Protocol: (1, 1, 1, 0), 
        Analytical Bound: 25200, Experimental Trunc: 1952

Protocol: (0, 0), 
        Analytical Bound: 400000, Experimental Trunc: 9960

Protocol: (1, 0, 0), 
        Analytical Bound: 518400, Experimental Trunc: 15280

Protocol: (1, 1, 0, 0), 
        Analytical Bound: 868700, Experimental Trunc: 21760

Protocol: (1, 1, 1, 0, 0), 
        Analytical Bound: 1701800, Experimental Trunc: 29440

- If I double t_coh, nothing changes, 
- If I divide it by 10, very small changes

p_gen: 0.1, p_swap: 0.1, w0: 0.99, t_coh: 10000

Protocol: 0, 
        Analytical Bound: 20000, Experimental Trunc: 670

Protocol: (1, 0), 
        Analytical Bound: 20001, Experimental Trunc: 924

Protocol: (1, 1, 0), 
        Analytical Bound: 20003, Experimental Trunc: 1208

Protocol: (1, 1, 1, 0), 
        Analytical Bound: 20005, Experimental Trunc: 1520

Protocol: (0, 0), 
        Analytical Bound: 400000, Experimental Trunc: 9960

Protocol: (1, 0, 0), 
        Analytical Bound: 400002, Experimental Trunc: 13840

Protocol: (1, 1, 0, 0), 
        Analytical Bound: 400004, Experimental Trunc: 18240

Protocol: (1, 1, 1, 0, 0), 
        Analytical Bound: 400006, Experimental Trunc: 23040
        
p_gen: 0.1, p_swap: 0.1, w0: 0.99, t_coh: 1000

Protocol: 0, 
        Analytical Bound: 20000, Experimental Trunc: 670

Protocol: (1, 0), 
        Analytical Bound: 39999, Experimental Trunc: 928

Protocol: (1, 1, 0), 
        Analytical Bound: 80000, Experimental Trunc: 1232

Protocol: (1, 1, 1, 0), 
        Analytical Bound: 160000, Experimental Trunc: 1584

Protocol: (0, 0), 
        Analytical Bound: 400000, Experimental Trunc: 9960

Protocol: (1, 0, 0), 
        Analytical Bound: 800000, Experimental Trunc: 13920

Protocol: (1, 1, 0, 0), 
        Analytical Bound: 1600000, Experimental Trunc: 18560

Protocol: (1, 1, 1, 0, 0), 
        Analytical Bound: 3200000, Experimental Trunc: 23680