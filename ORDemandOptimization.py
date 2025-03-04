import pulp  
import pandas as pd  
  
# Define the number of days  
num_days = 30  
  
# Define daily demand and penalty costs  
daily_demand = [  
    1400, 1000, 1000, 1004, 1005, 1006, 1007, 1008, 1009, 1010,  
    1011, 1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020,  
    1021, 1022, 1023, 1024, 1025, 1026, 1027, 1028, 1029, 1030  
]  
  
penalty_costs = [  
    100, 100, 100, 100, 1200, 1200, 1200, 1200, 1200, 1200,  
    1500, 1500, 1500, 1500, 1200, 1500, 2500, 2500, 2500, 2500,  
    2200, 2200, 1800, 1600, 2160, 2180, 2200, 2220, 2240, 2260  
]  
  
# Define plant parameters  
plants = {  
    'Plant1': {  
        'capacity': 1500,              # Max capacity per day  
        'cost': 8,                      # Cost per ton  
        'availability': list(range(1, 21)),  # Days 1 to 20  
        'min_capacity': 100,            # Min capacity per day when available  
        'ramp_up': 500,                 # Ramp-up limit per day  
        'ramp_down': 300                # Ramp-down limit per day  
    },  
    'Plant2': {  
        'capacity': 5000,  
        'cost': 12,  
        'availability': list(range(1, 31)),  # Always available  
        'min_capacity': 400,  
        'ramp_up': 500,  
        'ramp_down': 300  
    },  
    'Plant3': {  
        'capacity': 2000,  
        'cost': 15,  
        'availability': list(range(1, 31)),  # Always available  
        'min_capacity': 200,  
        'ramp_up': 300,  
        'ramp_down': 150  
    }  
}  
  
# Initialize the problem  
prob = pulp.LpProblem("Production_Planning", pulp.LpMinimize)  
  
# Define decision variables  
production = {  
    plant: [  
        pulp.LpVariable(f"prod_{plant}_day_{t+1}",  
                        lowBound=plants[plant]['min_capacity'] if (t+1) in plants[plant]['availability'] else 0,  
                        upBound=plants[plant]['capacity'] if (t+1) in plants[plant]['availability'] else 0,  
                        cat='Continuous')  
        for t in range(num_days)  
    ]  
    for plant in plants  
}  
  
unmet = [  
    pulp.LpVariable(f"unmet_day_{t+1}", lowBound=0, cat='Continuous')  
    for t in range(num_days)  
]  
  
# Objective function: Minimize total production cost + penalty costs  
total_cost = pulp.lpSum(  
    plants[plant]['cost'] * production[plant][t]  
    for plant in plants  
    for t in range(num_days)  
) + pulp.lpSum(  
    penalty_costs[t] * unmet[t]  
    for t in range(num_days)  
)  
  
prob += total_cost, "Total_Cost"  
  
# Constraints  
  
# Demand fulfillment  
for t in range(num_days):  
    prob += (  
        pulp.lpSum(production[plant][t] for plant in plants) + unmet[t] == daily_demand[t],  
        f"Demand_Fulfillment_Day_{t+1}"  
    )  
  
# Ramp-up and Ramp-down constraints  
for plant in plants:  
    for t in range(1, num_days):  
        # Check if plant is available on both day t and day t+1  
        if (t+1) in plants[plant]['availability'] and t in plants[plant]['availability']:  
            # Ramp-up  
            prob += (  
                production[plant][t] - production[plant][t-1] <= plants[plant]['ramp_up'],  
                f"Ramp_Up_{plant}_Day_{t+1}"  
            )  
            # Ramp-down  
            prob += (  
                production[plant][t-1] - production[plant][t] <= plants[plant]['ramp_down'],  
                f"Ramp_Down_{plant}_Day_{t+1}"  
            )  
        elif (t+1) in plants[plant]['availability'] and t not in plants[plant]['availability']:  
            # If the plant becomes available on day t+1 after being unavailable on day t  
            # No ramp-down constraint needed as production was 0 on day t  
            prob += (  
                production[plant][t] <= plants[plant]['ramp_up'],  
                f"Ramp_Up_{plant}_Day_{t+1}_From_0"  
            )  
        elif (t+1) not in plants[plant]['availability'] and t in plants[plant]['availability']:  
            # If the plant becomes unavailable on day t+1 after being available on day t  
            # Production must drop to 0  
            prob += (  
                production[plant][t] <= plants[plant]['ramp_down'],  
                f"Ramp_Down_{plant}_Day_{t+1}_To_0"  
            )  
        # Else: both days unavailable, no constraints needed  
  
# Solve the problem  
solver = pulp.PULP_CBC_CMD(msg=True)  # msg=True to display solver messages  
prob.solve(solver)  
  
# Print the status of the solution  
print(f"Status: {pulp.LpStatus[prob.status]}\n")  
  
# Print the total cost  
print(f"Total Optimum Cost (Rs): {pulp.value(prob.objective):,.2f}\n")  
  
# Create a DataFrame to display production schedule  
schedule = pd.DataFrame({  
    'Day': list(range(1, num_days + 1)),  
    'Unmet_Demand': [round(unmet[t].varValue, 2) for t in range(num_days)]  
})  
  
for plant in plants:  
    schedule[plant] = [round(production[plant][t].varValue, 2) if (t+1) in plants[plant]['availability'] else 0.0 for t in range(num_days)]  
  
# Rearrange columns  
cols = ['Day'] + [plant for plant in plants] + ['Unmet_Demand']  
schedule = schedule[cols]  
  
# Display the production schedule  
print("Production Schedule:")  
print(schedule.to_string(index=False))  
  
# Optionally, save the schedule to a CSV file  
# schedule.to_csv("production_schedule.csv", index=False)  