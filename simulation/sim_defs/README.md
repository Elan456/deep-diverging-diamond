# Defining a simulation

Create a csv where it's a list routes

For example:

```csv
1,5,3,2,1,5,7,8
```

This will spawn in 8 cars:

First car will take route #1  
Second car will take route #5  
Third car will take route #3  
etc...   

All cars are spawned at the same time.
If two cars are taking the same route, then the second car will
wait until the first car leaves the initial node.


# Route Definition  
The routes are defines in the `lane_defs.py` file as a list called `routes`.