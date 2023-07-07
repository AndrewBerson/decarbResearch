# decarbResearch


### Introduction
California has an ambitious goal to be net-zero emissions by 2045. While the state has enacted many policies
aimed at driving the energy transition, California's exact path to net-zero still has question marks surrounding
it. 

The goal of this project is to evaluate economy-wide decarbonization pathways for the state of California. 
Ultimately, we hope to illuminate what it will take to get to net-zero, as well as identify the most effective, 
economical, and feasible pathways.

### Method
We have built an economy-wide decarbonization model, **DECAL** (**DE**carbonize **CAL**ifornia), 
that evaluates emissions, costs, and resource use. 
Energy supply is balanced with demand such that reduced demand for fuels leads to reduced production of these fuels. 
The model is not an equilibrium model; instead, behavior is dictated by exogenously defined levers that control 
deployment rates, conversions rates, technology choices, and more. The data used to generate this model was 
collected by researchers at the [Stanford Center for Carbon Storage](https://sccs.stanford.edu/california-projects/pathways-carbon-neutrality-california)
who did in-depth research into sectors and/or subsectors of the California economy. 
In this way, most of the data was collected in a ”bottom-up” fashion. 
The model is built using the Stockholm Energy Institute’s 
Low Emissions Analysis Platform ([LEAP](https://leap.sei.org/)).

Running experiments consists of a few different steps:
1. Define scenarios by setting levers in **`master_scenarios.xlsx`**  (not stored on github)
    - eg: scenario A has 100% zero emission vehicle (ZEV) sales by 2035 and scenario B has 100% ZEV sales by 2045
2. Run scenarios using DECAL model (not yet publicly available)
3. Parse DECAL output using **`results_parser.py`**
4. Generate graphics using **`controller.xlsm`** and **`graph_maker.py`**

This infrastructure has allowed us to run hundreds of simulations, evaluating many decarbonization strategies within 
all sectors of the California economy.

### Preliminary Conclusions
- All technologies and resources will be needed to get to net zero by 2045
- Electrification will require major expansion of the grid 
(approximately 225 – 400 GW of capacity depending on clean generation constraint)
- Going from 98% to 100% carbon-free electricity generation is very expensive
- Policies encouraging ZEV sales can be very effective 
- Point source CCS is effective and economically favorable for the industrial sector
- F-Gas mitigation requires innovation
- Expanding use of H2 may be very expensive, especially due to distribution & storage
- Renewable Natural Gas (RNG) and Renewable Diesel (RD) usage may be limited by feedstock availability
- It is very difficult to reach net-zero emissions by 2045 without significant carbon dioxide removal (CDR) 
(>35 Mt/yr by 2045)

### [Further Preliminary Results](https://drive.google.com/file/d/1OParwMYAkYyiFHtbJwB7z3EGIeOMrU_a/view?usp=sharing) 
- All results are preliminary and subject to change