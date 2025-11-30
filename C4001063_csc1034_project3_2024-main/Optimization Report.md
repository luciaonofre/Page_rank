# **Reflective Report : PageRank Code Optimization**

## Introduction:
This report evaluates the effectiveness of the strategies I applied to optimize the stochastic and distribution 
algorithms, used for PageRank estimation.

## Optimization strategies:

All tested in their default version: 

_1. Stochastic: 1000000 repetitions, adjacency list, top 20 websites_

_2. Distribution: 100 repetitions, adjacency list, top 20 websites_

### 1. Progress Bar Adjustments:

In the start, the progress bar was being updated on every iteration which could significantly increase runtime. 
So, instead of using `progress.show()` in every loop, the code could be changed to use a lower update interval.

* **Optimization**: Update the progress bar every 1000 repetitions or steps

* **Before Optimization**:
            
      a. Stochastic method: 0.94 seconds
    
      b. Distribution method: 0.18 seconds

* **After Optimization**:
            
      a. Stochastic method: 0.43 seconds
    
      b. Distribution method: 0.10 seconds

* **Outcome:** 
Reducing the frequency of progress bar updates optimized the runtime in both methods. Stochastic time ran in half of 
its previous time, leading to faster execution, therefore I implemented this optimization.

### 2. Graph Representation:
In this case, four graph representation methods were implemented and if I wanted to optimize the code I would only 
want to choose one, to prioritize time, memory usage and execution speed. To evaluate this, I ran all the 
representations to see which one would have been the one I would have implemented in a real case scenario.

* **Execution Times for each Graph Representation**: 

        1. Stochastic method :
                a. Adjacency List: 0.01 seconds
                b. Adjacency Matrix: 0.01 seconds
                c. Edge List: 0.00 seconds
                d. Network X: 0.00 seconds

        2. Distribution method :
                a. Adjacency List : 0.17 seconds
                b. Adjacency Matrix : 0.13 seconds
                c. Edge List : 0.16 seconds
                d. Network X : 0.13 seconds

* **Outcome:** 
The Network X representation was quicker in both methods. If I did not have to display 4 different graph methods, 
I would have chose that one or Edge List, for its similar efficiency.

### 3. Default Dictionary Implementation: 
At first, in the `load_graph` function, I was checking if a node existed in the dictionary before appending it a target. 
So, for a moment I decided to use `defaultdict(list)` from the collections module to avoid that extra checking step and 
instead automatically create an empty list for any missing node or target.

* **Optimization**: implement `defaultdict(list)` in the `load_graph` function

* **Before Optimization**:
            
      a. Stochastic method: 0.58 seconds
    
      b. Distribution method: 0.12 seconds

* **After Optimization**:
            
      a. Stochastic method: 0.50 seconds
    
      b. Distribution method: 1.26 seconds

* **Outcome:** 
The implementation `defaultdict(list)` barely reduced any execution time for the Stochastic method and increased
the execution time for Distribution significantly. Because of its minimal gain and negative impact, I kept my original
code.

## Conclusion:
All the optimizations varied in success. Adjusting the progress bar significantly benefited runtime for Stochastic. 
As well, the Graph representation analysis confirmed Network X's efficiency overall. However, the implementation of 
`defaultdict(list)` was not possible because its impact on Distribution runtime. Certainly, some optimizations improved 
runtime, while others could not be implemented due to their mixed results.



