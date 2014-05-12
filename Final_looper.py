# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy as np
import copy
import matplotlib.pyplot as plt
import seaborn as sns
import operator
import csv
from IPython import display
from multiprocessing import Pool
import pickle
import scipy as sp
from collections import Counter

# <markdowncell>

# ###Framework

# <codecell>

#define fibb series
def fib(n): # return Fibonacci series up to n
    result = []
    a, b = 1, 1
    while b < n:
        result.append(b)
        if b < 10:
            a, b = b, a+b
        else:
            b = b+10
    return result

#get data from CSV
def get_csv_data(ticker, ndays=365*15):
    stock_data = []
    volume_data = []
    
    with open('stock_data/long_SP500/'+ticker+'.csv', 'rb') as f: 
        reader = csv.reader(f) 
        days = 0
        
        #skip first row 
        next(reader) 
        for row in reader: 
            stock_data.append(float(row[-1]))
            volume_data.append(float(row[-2]))
            days += 1
            
            #get only the first ndays data
            if days == ndays:
                break
            
    #reverse data
    stock_data = np.array(stock_data[::-1])
    volume_data = np.array(volume_data[::-1])
    return stock_data, volume_data

#precalulcate data streams
def precalculate(N_max, max_window, day_st, stock_data, volume_data):

    max_windex = len(fib(max_window))
    n_categories = 11 #14+len(fib(30))
    
    data_streams = []+[[np.nan]*max_windex for i in range(n_categories)]
    data_streams[0] = [
                       stock_data[day_st:day_st+N_max],
                       volume_data[day_st:day_st+N_max]
                       ]
    #add difference
    for window in fib(30):
        data_streams[0].append(np.concatenate(([np.nan]*window,np.diff(data_streams[0][0], window))))
    
    #generate moving window calculations (do in fibonacci)
    for w_index, window in enumerate(fib(max_window)):
        
        #init array of zeros for all operations for this particular window
        for j in range(1,n_categories):
            blank = np.empty(N_max)
            blank[:] = np.nan
            data_streams[j][w_index] = blank #np.zeros(N_max - window)

        #calculate
        for i in range(window,N_max):
            data_streams[1][w_index][i] = np.mean(data_streams[0][0][i-window:i])
            data_streams[2][w_index][i] = np.std(data_streams[0][0][i-window:i])
            data_streams[3][w_index][i] = min(data_streams[0][0][i-window:i])
            data_streams[4][w_index][i] = max(data_streams[0][0][i-window:i])
            
            data_streams[6][w_index][i] = np.mean(data_streams[0][1][i-window:i])
            data_streams[7][w_index][i] = np.std(data_streams[0][1][i-window:i])
            data_streams[8][w_index][i] = min(data_streams[0][1][i-window:i])
            data_streams[9][w_index][i] = max(data_streams[0][1][i-window:i])
            
            #for j in range(len(fib(30))):
            #    data_streams[11+j][w_index][i] = np.mean(data_streams[0][2+j][i-window:i])
            #    data_streams[12+j][w_index][i] = np.std(data_streams[0][2+j][i-window:i])
            #    data_streams[13+j][w_index][i] = min(data_streams[0][2+j][i-window:i])
            #    data_streams[14+j][w_index][i] = max(data_streams[0][2+j][i-window:i])
        
        #lag
        
        data_streams[5][w_index][(window):] = data_streams[0][0][:(-window)]
        data_streams[10][w_index][(window):] = data_streams[0][1][:(-window)]
        
    n_categories = len(data_streams)
    n_rawdatastreams = len(data_streams[0])
        
    return data_streams, n_categories, n_rawdatastreams

# <codecell>

#define operations

#trim
def trim(ts1,ts2):
    len2 = len(ts2)
    len1 = len(ts1)
    
    if len1 >= len2:
        return ts1[(-len2):], ts2[(-len2):]
    else:
        return ts1[(-len1):], ts2[(-len1):]

#add
def op_add(c, ts1, ts2):
    ts1, ts2 = trim(ts1,ts2)
    return ts1 + ts2

#subtract
def op_sub(c, ts1, ts2):
    ts1, ts2 = trim(ts1,ts2)
    return ts1 - ts2

#norm
def op_norm(c, ts1, ts2):
    ts1, ts2 = trim(ts1,ts2)
    return abs(ts1 - ts2)

#mulitply
def op_mult(c, ts1, ts2):
    ts1, ts2 = trim(ts1,ts2)
    return c*ts1

#divide
def op_div(c,ts1,ts2):
    ts1, ts2 = trim(ts1,ts2)
    return ts1/ts2

#less than
def op_lt(c,ts1,ts2):
    ts1, ts2 = trim(ts1,ts2)
    return ts1<ts2

#more than
def op_gt(c,ts1,ts2):
    ts1, ts2 = trim(ts1,ts2)
    return ts1>ts2

operations = {
              #real-valued outputs
              1: op_add,
              2: op_sub,
              3: op_norm,
              4: op_mult, 
              5: op_div,
              #boolean outputs
              6: op_lt,
              7: op_gt
              }

bool_input_number = 6
const_functions = [4]
one_ts_functions = [4]

# <codecell>

def process(obj, start_date, end_date, data_streams):
    
    #if the object is just a timeseries represented by tuple, return it
    if len(obj) == 2:
        ts = data_streams[obj[0]][obj[1]][start_date:end_date]
        #return only series without nans
        return ts[~np.isnan(ts)]
    
    #if the object is just a timeseries, return it
    if len(obj) > 4:
        return obj
    
    #otherwise, object is a procedure. Perfrom the indicated operation on the two input TSs
    return operations[obj[0]](obj[1], process(obj[2], start_date, end_date, data_streams), 
                              process(obj[3], start_date, end_date, data_streams) )

# <codecell>

def fitness((buy_rule,sell_rule), start_date, end_date, data_streams, splits = 4, risk_adj = False):
    
    bool_timeseries_buy = process(buy_rule, start_date, end_date, data_streams)
    bool_timeseries_sell = process(sell_rule, start_date, end_date, data_streams)
    ts = data_streams[0][0][start_date:end_date]
    
    #add zeros to the bools
    addon = len(ts) - len(bool_timeseries_buy)
    bool_timeseries_buy = np.concatenate((np.zeros(addon),bool_timeseries_buy))
    addon = len(ts) - len(bool_timeseries_sell)
    bool_timeseries_sell = np.concatenate((np.zeros(addon),bool_timeseries_sell))
    
    #split arrays
    buy_set = np.array_split(bool_timeseries_buy, splits)
    sell_set = np.array_split(bool_timeseries_sell, splits)
    ts_set = np.array_split(ts, splits)

    fitness_set = np.zeros(splits)
    min_fitness = (np.inf,0,[])
        
    #loop over splits
    for en, (buy_i, sell_i, ts_i) in enumerate(zip(buy_set, sell_set, ts_set)):
    
        #run through buying and selling
        profit = 0
        state = "buy"
        n_transactions = 0
        bs = np.zeros(len(ts_i))
        first_buy  = None
        last_sell = None
        
        for i in range(len(buy_i)):
            if (buy_i[i] and state == "buy"):
                buy_price = ts_i[i]
                if first_buy == None:
                    first_buy = ts_i[i]
                state = "sell"
                n_transactions += 1
                bs[i] = -1
            elif (sell_i[i] and state == "sell" and ts_i[i] > buy_price):
                state = "buy"
                profit += ts_i[i] - buy_price
                last_sell = ts_i[i]
                n_transactions += 1
                bs[i] = 1
        
        #subtract out buy and hold method if there is at least one buy and one sell
        if risk_adj and (n_transactions > 1):
            profit -= last_sell - first_buy
        
        if profit < min_fitness[0]:
            min_fitness = (np.array([profit]), np.array([n_transactions]), bs)

    return min_fitness[0]*splits, min_fitness[1]*splits, min_fitness[2], 

# <markdowncell>

# ###Mutations and Recbominations

# <codecell>

#define mutations

#mutate ts input
def mutate_ts((category,period), cat_mu, period_mu, n_categories, n_rawdatastreams):
    
    #mutate type of time series
    if np.random.rand() < cat_mu:
        category = np.random.randint(n_categories)
        
    #these are raw data streams
    if (category == 0):
        return (0, np.random.randint(n_rawdatastreams))
    
    #mutate the window
    if (np.random.rand() < period_mu):
        #choose up or down
        step = np.random.choice([-1,1])
        #make sure window is within bounds
        if (period + step) < len(fib(365)) and (period + step) >= 0:
            period += step
    return (category,period)
        
#mutate operation
def mutate_op(operation, constant, op_mu, const_mu, const_step):
    
    #mutate operation type
    if (np.random.rand() < op_mu):
        if (operation < bool_input_number):
            operation = np.random.randint(1,bool_input_number)
        else:
            operation = np.random.randint(bool_input_number, len(operations)+1)
    
    if ((operation in const_functions) and (np.random.rand() < const_mu)):
        constant += np.random.randn()*const_step
    
    return operation, constant

def mutate_rule(obj, cat_mu, period_mu, op_mu, const_mu, const_step, n_categories, n_rawdatastreams):
    
    #if the object is just a timeseries represented by tuple, mutate it
    if len(obj) == 2:
        obj = mutate_ts(obj, cat_mu, period_mu, n_categories, n_rawdatastreams)
        return obj
    
    #otherwise, object is a procedure. mutate operation
    obj[0], obj[1] = mutate_op(obj[0], obj[1], op_mu, const_mu, const_step)
    
    #mutate the ts
    obj[2] = mutate_rule(obj[2], cat_mu, period_mu, op_mu, const_mu, const_step, n_categories, n_rawdatastreams)
    obj[3] = mutate_rule(obj[3], cat_mu, period_mu, op_mu, const_mu, const_step, n_categories, n_rawdatastreams)
    
    return obj

def mutate_member(mem, cat_mu, period_mu, op_mu, const_mu, const_step, n_categories, n_rawdatastreams):
    return (mutate_rule(mem[0], cat_mu, period_mu, op_mu, const_mu, const_step, n_categories, n_rawdatastreams), 
            mutate_rule(mem[1], cat_mu, period_mu, op_mu, const_mu, const_step, n_categories, n_rawdatastreams))

#swap buy or sell rules with other members
def recombine1(mem1, mem2):
    #choose between buy or sell rule:
    c = np.random.randint(2)
    if c == 1:
        return (mem1[0], mem2[1]), (mem2[0], mem1[1])
    else:
        return (mem2[0], mem1[1]), (mem1[0], mem2[1])

#traverse and swap
def traverse(obj, traverse_index = 0, index_list = []):

    #obj is a rule
    if len(obj) == 4:
        
        #store this subtree
        index_list.append( obj )
        traverse_index += 1
        
        #count right branch
        traverse_index, index_list = traverse(obj[2], traverse_index, index_list)
        
        #count left branch
        traverse_index, index_list = traverse(obj[3], traverse_index, index_list)
            
    #if the object is just a timeseries represented by tuple
    else:
        index_list.append( obj )
        traverse_index += 1
        return traverse_index, index_list
    
    return traverse_index, index_list

def traverse_replace(curr_obj, traverse_index = 0, replace_index = -1, replacement = []):

    obj = copy.deepcopy( curr_obj )
    
    #obj is a rule
    if len(obj) == 4:
        
        #replace this
        if traverse_index == replace_index:
            obj = replacement
            traverse_index += 1
            return traverse_index, obj

        traverse_index += 1
        
        #count right branch
        traverse_index, obj[2] = traverse_replace(obj[2], traverse_index, replace_index, replacement)
        
        #count left branch
        traverse_index, obj[3] = traverse_replace(obj[3], traverse_index, replace_index, replacement)

        return traverse_index, obj
            
    #if the object is just a timeseries represented by tuple
    else:
        
        #replace this
        if traverse_index == replace_index:
            obj = replacement
            traverse_index += 1
            return traverse_index, obj
        
        traverse_index += 1     
        
        return traverse_index, obj
    
def recombine2(mem1, mem2):
    
    mem1 = list(mem1)
    mem2 = list(mem2)
    
    #choose between buy or sell rule:
    c = np.random.randint(2)
    rule1 = mem1[c]
    rule2 = mem2[c]
    
    #traverse both
    mem1_len, mem1_tree = traverse(rule1, traverse_index = 0, index_list = [])
    mem2_len, mem2_tree = traverse(rule2, traverse_index = 0, index_list = [])
    
    #limit size of tree to 20
    if mem1_len < 20 and mem2_len < 20: 
    
        #choose one of each randomly (0 is not eligible because it is boolean always)
        mem1_choice = np.random.randint(1,mem1_len)
        mem2_choice = np.random.randint(1,mem2_len)
        
        mem1[c] = traverse_replace(rule1, traverse_index = 0, replace_index = mem1_choice, replacement = mem2_tree[mem2_choice])[1]
        mem2[c] = traverse_replace(rule2, traverse_index = 0, replace_index = mem2_choice, replacement = mem1_tree[mem1_choice])[1]
        
    #replace and return
    return (tuple(mem1), tuple(mem2))
    

# <markdowncell>

# ##Genetic Algorithm

# <codecell>

#create Bollinger bands
ubb = [1,1,
    #get standard deviation for 20 days, multiply by 2
    [4, 2, (2,6), (0,0)] ,
    #get mean
    (1,6)
    ]

lbb = [2,
       1, 
       (1,6),
       [4, 1, (2,6), (0,0)]
    ]

ubb_bool = [7,1, (0,0), ubb]
lbb_bool =[6,1, (0,0), lbb]

bollinger = (lbb_bool, ubb_bool)

# <codecell>

#%
def gp(stock_data, volume_data, pop_size = 300, generations = 250, 
       cat_mu = 0.05,               #type of aggregated time series
       period_mu = 0.02,            #period of aggregation
       op_mu = 0.02,                #type of operation
       const_mu = 0.4,              #constant used in operation
       const_step = 0.5, 
       recombine1_n = 100,          #number of pairs to recombine1
       recombine2_n = 200,          #number of pairs to recombine2
       initial_mem = bollinger,
       train_start = 0,             #training period start day
       train_end = 365*10,          #training period end day
       test_start = 365*10,         #test period start day
       test_end = 365*15,            #test period end day
       update_window = 25,
       trans_cost = 0.007,
       cross_val_splits = 4
       ):
    
    #generate initial population
    g0 = []
    for i in range(pop_size): 
        newmember = copy.deepcopy(initial_mem)
        if not i==0:
            mutate_member(newmember, cat_mu, period_mu, op_mu, const_mu, const_step, n_categories, n_rawdatastreams)
        g0.append(newmember)
    
    #recombine1
    for pair in np.random.choice(pop_size,(recombine1_n,2)):
        g0[pair[0]], g0[pair[1]] = recombine1(g0[pair[0]], g0[pair[1]])
        
    #recombine2
    for pair in np.random.choice(pop_size,(recombine2_n,2)):
        g0[pair[0]], g0[pair[1]] = recombine2(g0[pair[0]], g0[pair[1]])
    
    #init
    generation_fitness = np.zeros(generations)
    avg_generation_fitness = np.zeros(generations)
    test_generation_fitness = np.zeros(generations)
    test_avg_generation_fitness = np.zeros(generations)
    polyclonality = np.zeros(generations)
    
    #put placeholder for optimal solution
    optimal_sol = initial_mem
    optimal_fitness, opt_n_trans, opt_bs = fitness(optimal_sol, train_start, train_end, data_streams, 1)
    optimal_sols = []
    optimal_sol_date = []
    optimal_sols_history = []
    optimal_sols_history.append(optimal_sol)
        
    g_curr = copy.deepcopy(g0)
    plt.figure(figsize=(24,12))
    
    for z in range(generations):
        
        #don't mutate on first generation
        if not z==0:
            
            #recombine1
            for pair in np.random.choice(pop_size,(recombine1_n,2)):
                g_curr[pair[0]], g_curr[pair[1]] = recombine1(g_curr[pair[0]], g_curr[pair[1]])
                
            #recombine2
            for pair in np.random.choice(pop_size,(recombine2_n,2)):
                g_curr[pair[0]], g_curr[pair[1]] = recombine2(g_curr[pair[0]], g_curr[pair[1]])
            
            #mutate
            for en, mem in enumerate(g_curr):
                g_curr[en] = mutate_member(mem, cat_mu, period_mu, op_mu, const_mu, const_step, n_categories, n_rawdatastreams)
        
        #calculate fittness
        [fit_curr, n_trans, bs] = np.array( [fitness(memi, train_start, train_end, data_streams, cross_val_splits) for memi in g_curr] ).T
        [test_fit_curr, test_n_trans, test_bs] = np.array( [fitness(memi, test_start, test_end, data_streams, 1) for memi in g_curr]).T 
        
        #turn on transaction cost
        if trans_cost > 0:
            fit_curr -= n_trans*trans_cost
            test_fit_curr -= test_n_trans*trans_cost
        
        #calculate clonality
        polyclonality[z] = len(np.unique(fit_curr))
        
        generation_fitness[z] = np.nanmax(fit_curr)
        avg_generation_fitness[z] = fit_curr.mean()
            
        test_generation_fitness[z] = test_fit_curr[fit_curr==generation_fitness[z]][0]
        test_avg_generation_fitness[z] = test_fit_curr.mean()
        
        #save optimal_fitness
        if generation_fitness[z] > optimal_fitness:
            optimal_sol = [ g_curr[i] for i in range(pop_size) if fit_curr[i] == generation_fitness[z] ][0]
            optimal_sols_history.append(optimal_sol)
            optimal_fitness = generation_fitness[z]
            optimal_sols.append(np.nanmax(fit_curr))
            optimal_sol_date.append(z)
        
        #floor current fitnesses at min to make all positive fitnesses
        fit_curr_floored = fit_curr - np.nanmin(fit_curr)
        #fit_curr_floored[np.isnan(fit_curr_floored)] = np.nanmin(fit_curr_floored)
        marg_fit = fit_curr_floored.cumsum()/fit_curr_floored.sum()
    
        #keep clonality up
        if polyclonality[z] > 20.:
            #reproduce
            cum_sum = 0
            r = np.random.rand(pop_size)
            g_new = []
            
            #count number of offspring per member of population
            for i in range(pop_size):
                num_offspring = np.sum([(r > cum_sum )*(r <=(marg_fit[i]))])
                cum_sum = marg_fit[i]
                
                for j in range(num_offspring):
                    g_new.append(copy.deepcopy(g_curr[i]))
        else:
            #generate population from optimal solution
            g_new = []
            for i in range(pop_size):
                newmember = copy.deepcopy(optimal_sol)
                g_new.append(newmember)
    
        #replace current population
        g_curr = copy.deepcopy(g_new)
        
        if (((z+1) % update_window == 0)):
            plt.clf()

            plt.subplot(3,1,1)
            plt.plot(generation_fitness[:z], label='Train Fitness')
            plt.plot(avg_generation_fitness[:z], label='Average Train Fitness')
            plt.plot(test_generation_fitness[:z], label='Test Fitness')
            plt.plot(test_avg_generation_fitness[:z], label='Average Test Fitness')
            plt.plot(optimal_sol_date, optimal_sols, 'ro')
            plt.xlabel('Generation')
            plt.ylabel('Best Fitness in Generation')
            plt.title('Best of Current Generation: '+str(generation_fitness[z])+'\n Best so far:'+str(optimal_fitness))
            plt.legend()
            
            plt.subplot(3,1,2)
            plt.plot(polyclonality[:z])
            plt.xlabel('Generation')
            plt.ylabel('Colonies')
            
            #plt.subplot(3,1,3)
            #plt.plot(avg_generation_fitness[:z] - test_avg_generation_fitness[:z])
            #plt.xlabel('Generation')
            #plt.ylabel('Average Train Fitness - Test Fitness')
    
            display.clear_output()
            display.display(plt.gcf())
        
        
    display.clear_output()
    
    return optimal_sol

# <codecell>

def viz(rule,
            train_start = 0,             #training period start day
           train_end = 365*10,          #training period end day
           test_start = 365*10,         #test period start day
           test_end = 365*15            #test period end day
        ):
    
    plt.plot(data_streams[0][0], label="Price")
    
    #training period
    trained = fitness(rule, train_start, train_end, data_streams, 1, True)
    tr_sell_mask = trained[2] > 0
    tr_buy_mask = trained[2] < 0
    
    plt.axvline(test_start)
    
    #test period
    tested = fitness(rule, test_start, test_end, data_streams, 1, True)
    ts_sell_mask = tested[2] > 0
    ts_buy_mask = tested[2] < 0
    
    sell_mask = np.concatenate((tr_sell_mask, ts_sell_mask))
    buy_mask = np.concatenate((tr_buy_mask, ts_buy_mask))
    
    plt.plot(np.arange(len(data_streams[0][0]))[sell_mask], data_streams[0][0][sell_mask], 'ro', alpha=1)
    plt.plot(np.arange(len(data_streams[0][0]))[buy_mask], data_streams[0][0][buy_mask], 'go', alpha=1)
    
    plt.xlim([train_start,test_end])
    plt.legend()
    
    print trained[0] , tested[0]


# <markdowncell>

# ##Looping over many stocks

# <codecell>

#make this fast
def gp_fast(stock_data, volume_data,
            data_streams, n_categories, n_rawdatastreams,
       pop_size = 300, generations = 250, 
       cat_mu = 0.05,               #type of aggregated time series
       period_mu = 0.02,            #period of aggregation
       op_mu = 0.02,                #type of operation
       const_mu = 0.4,              #constant used in operation
       const_step = 0.5, 
       recombine1_n = 100,          #number of pairs to recombine1
       recombine2_n = 200,          #number of pairs to recombine2
       initial_mem = bollinger,
       train_start = 0,             #training period start day
       train_end = 365*10,          #training period end day
       trans_cost = 0.007,
       cross_val_splits = 4
       ):
    
    #generate initial population
    g0 = []
    for i in range(pop_size): 
        newmember = copy.deepcopy(initial_mem)
        if not i==0:
            mutate_member(newmember, cat_mu, period_mu, op_mu, const_mu, const_step, n_categories, n_rawdatastreams)
        g0.append(newmember)
    
    #recombine1
    for pair in np.random.choice(pop_size,(recombine1_n,2)):
        g0[pair[0]], g0[pair[1]] = recombine1(g0[pair[0]], g0[pair[1]])
        
    #recombine2
    for pair in np.random.choice(pop_size,(recombine2_n,2)):
        g0[pair[0]], g0[pair[1]] = recombine2(g0[pair[0]], g0[pair[1]])
    
    #init
    generation_fitness = np.zeros(generations)
    polyclonality = np.zeros(generations)
    
    #put placeholder for optimal solution
    optimal_sol = []
    optimal_fitness = -np.inf
        
    g_curr = copy.deepcopy(g0)
    
    for z in range(generations):
        
        #don't mutate on first generation
        if not z==0:
            
            #recombine1
            for pair in np.random.choice(pop_size,(recombine1_n,2)):
                g_curr[pair[0]], g_curr[pair[1]] = recombine1(g_curr[pair[0]], g_curr[pair[1]])
                
            #recombine2
            for pair in np.random.choice(pop_size,(recombine2_n,2)):
                g_curr[pair[0]], g_curr[pair[1]] = recombine2(g_curr[pair[0]], g_curr[pair[1]])
            
            #mutate
            for en, mem in enumerate(g_curr):
                g_curr[en] = mutate_member(mem, cat_mu, period_mu, op_mu, const_mu, const_step, n_categories, n_rawdatastreams)
        
        #calculate fittness
        [fit_curr, n_trans, bs] = np.array( [fitness(memi, train_start, train_end, data_streams, cross_val_splits) for memi in g_curr] ).T
        
        #turn on transaction cost
        if trans_cost > 0:
            fit_curr -= n_trans*trans_cost
        
        #calculate clonality
        polyclonality[z] = len(np.unique(fit_curr))
        
        generation_fitness[z] = np.nanmax(fit_curr)
        
        #save optimal_fitness
        if generation_fitness[z] > optimal_fitness:
            optimal_sol = [ g_curr[i] for i in range(pop_size) if fit_curr[i] == generation_fitness[z] ][0]
            optimal_fitness = generation_fitness[z]

        #floor current fitnesses at min to make all positive fitnesses
        fit_curr_floored = fit_curr - np.nanmin(fit_curr)
        marg_fit = fit_curr_floored.cumsum()/fit_curr_floored.sum()
    
        #keep clonality up
        if polyclonality[z] > 20.:
            #reproduce
            cum_sum = 0
            r = np.random.rand(pop_size)
            g_new = []
            
            #count number of offspring per member of population
            for i in range(pop_size):
                num_offspring = np.sum([(r > cum_sum )*(r <=(marg_fit[i]))])
                cum_sum = marg_fit[i]
                
                for j in range(num_offspring):
                    g_new.append(copy.deepcopy(g_curr[i]))
        else:
            #generate population from optimal solution
            g_new = []
            for i in range(pop_size):
                newmember = copy.deepcopy(optimal_sol)
                g_new.append(newmember)
    
        #replace current population
        g_curr = copy.deepcopy(g_new)
    
    return optimal_sol

# <codecell>

#functionalize the enture process
def complete_GP(ticker):
    Total_days = 365*15

    #get S&P and stock Data
    stock_data, volume_data = get_csv_data(ticker, ndays=Total_days)
    
    #precalculate data streams
    data_streams, n_categories, n_rawdatastreams = precalculate(Total_days, 365, 0, stock_data, volume_data)

    print "Doing:", ticker
    
    opti_sol =  gp_fast(stock_data, volume_data, data_streams, n_categories, n_rawdatastreams, 
                pop_size = 300, generations = 500, 
               cat_mu = 0.1,               #type of aggregated time series
               period_mu = 0.05,            #period of aggregation
               op_mu = 0.05,                #type of operation
               const_mu = 0.5,              #constant used in operation
               const_step = 1, 
               recombine1_n = 30,          #number of pairs to recombine1
               recombine2_n = 30,          #number of pairs to recombine2
               initial_mem = bollinger,
               train_start = 0,             #training period start day
               train_end = 365*10,          #training period end day
               trans_cost = 0.01,
               cross_val_splits = 4
               )

    output = (opti_sol, 
            fitness(opti_sol, 0, 365*10, data_streams, 1, True),
            fitness(opti_sol, 365*10, 365*15, data_streams, 1, True)
            )
    
    #pickle the results
    with open('final_proj_results/Output_'+ticker+'.pk', 'wb') as f:
        pickle.dump(output, f)
        
    return output


#DEFINE STOCKS
long_SP = ['CCE', 'BK', 'DHR', 'SCHW', 'MHFI', 'AMGN', 'TSN', 'VLO', 'MOS', 'AMAT', 'USB', 'TMK', 'MO', 'KO', 'IR', 'FAST', 'FRX', 'ROK', 'XL', 'NE', 'LM', 'HUM', 'CSCO', 'LUK', 'MDT', 'STT', 'PNC', 'NEE', 'AXP', 'AET', 'OMC', 'BEN', 'TAP', 'M', 'GPC', 'PRGO', 'BF.B', 'CMA', 'PCG', 'IP', 'EMR', 'KIM', 'TIF', 'HST', 'WMB', 'CTAS', 'LEN', 'HSY', 'TSO', 'GAS', 'PPG', 'TE', 'PCAR', 'PAYX', 'LMT', 'WDC', 'SO', 'ECL', 'CVS', 'VMC', 'FLS', 'HON', 'OI', 'KLAC', 'TSS', 'AVY', 'CERN', 'CSC', 'X', 'ROST', 'VZ', 'APA', 'R', 'JCI', 'PNR', 'CINF', 'HBAN', 'PG', 'PEP', 'CMCSA', 'XRX', 'APD', 'NBL', 'L', 'APH', 'BBT', 'SWY', 'NSC', 'CMI', 'DD', 'ED', 'AN', 'LSI', 'KMB', 'IFF', 'HRS', 'MWV', 'INTC', 'ADBE', 'AIG', 'KEY', 'PPL', 'WM', 'SYY', 'MCD', 'HOG', 'PBI', 'SYK', 'MAT', 'SNA', 'HAR', 'VFC', 'WEC', 'SEE', 'UNP', 'MTB', 'DNB', 'SHW', 'AON', 'ADM', 'MSFT', 'OXY', 'C', 'T', 'ORCL', 'NU', 'CELG', 'AGN', 'UTX', 'NBR', 'TRV', 'MRK', 'XOM', 'AFL', 'CMS', 'BMY', 'NWL', 'HAS', 'PH', 'AAPL', 'BHI', 'IBM', 'VNO', 'PCL', 'WHR', 'IGT', 'MMC', 'LEG', 'CSX', 'HRB', 'BAC', 'PGR', 'JWN', 'CVX', 'MKC', 'SIAL', 'UNM', 'MSI', 'HCP', 'NOC', 'STJ', 'WFC', 'COG', 'BAX', 'PCP', 'TROW', 'TGT', 'FTR', 'LOW', 'IPG', 'LLTC', 'QCOM', 'BIIB', 'COP', 'RF', 'XRAY', 'CA', 'BA', 'DIS', 'TMO', 'SYMC', 'SWK', 'WMT', 'CL', 'HPQ', 'PSA', 'JEC', 'EIX', 'CAT', 'RDC', 'CAH', 'AZO', 'GILD', 'CLX', 'DUK', 'TYC', 'ADP', 'ETN', 'CNP', 'GE', 'NI', 'D', 'FMC', 'HES', 'DE', 'TXN', 'NEM', 'JPM', 'DOW', 'KR', 'GD', 'HP', 'GLW', 'ADI', 'TEG', 'LRCX', 'VAR', 'MU', 'ROP', 'GHC', 'AES', 'BDX', 'FDO', 'UNH', 'LLY', 'FITB', 'ALTR', 'PNW', 'CPB', 'GT', 'MRO', 'HD', 'SWN', 'POM', 'MUR', 'PBCT', 'ETR', 'MMM', 'FDX', 'K', 'WAG', 'HAL', 'EXC', 'AVP', 'AME', 'LB', 'EQT', 'DTE', 'KSU', 'PLL', 'EFX', 'CI', 'SLB', 'GIS', 'EOG', 'BLL', 'PHM', 'AA', 'FISV', 'F', 'NUE', 'LUV', 'XEL', 'VRTX', 'PFE', 'EMC', 'NKE', 'DOV', 'PKI', 'AEP', 'THC', 'WY', 'CTL', 'CAG', 'SPLS', 'RTN', 'SCG', 'LNC', 'XLNX', 'PVH', 'ZION', 'JNJ', 'ABT', 'COST', 'MAS', 'BMS', 'ARG', 'OKE', 'ESV', 'NTRS', 'ADSK', 'TJX', 'CCL', 'REGN', 'STI', 'PEG', 'EA', 'GPS', 'GCI', 'EXPD', 'HRL', 'CB', 'MYL', 'TXT', 'BBY', 'GWW', 'BCR', 'WFM', 'ITW', 'HOT', 'APC', 'LH']

#Looping
pool = Pool(2)
optimal_sols = pool.map_async(complete_GP,long_SP)
optimal_sols.wait()