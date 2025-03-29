import json 

result = json.load(open("results/untrained_qwen_results.json"))
values = list(result.values())
print(len(values))  
print("average MSE" , sum(values[0])/len(values[0]))
print("average MAE" , sum(values[1])/len(values[1]))
print("average R2" , sum(values[2])/len(values[2]))