import torch
from torch import nn
import torch.nn.functional as F

import torch.fx

class MyModule(nn.Module):
    def __init__(self) -> None:
        super(MyModule,self).__init__()
        self.layer1=nn.Linear(28*28,128)
        self.layer2=nn.Linear(128,64)
        self.layer3=nn.Linear(64,10)
    
    def forward(self,x):
        x=x.view(-1,28*28)
        x=F.relu(self.layer1(x))
        x=F.relu(self.layer2(x))
        x=self.layer3(x)
        return x

model = MyModule()
symbolic_tranced:torch.fx.GraphModule=torch.fx.symbolic_trace(model)

print(symbolic_tranced.graph)
print(symbolic_tranced.code)

def transform(m:torch.fx.Graph,tracer_class:type=torch.fx.Tracer)->torch.nn.Module:
    # Step 1: Acquire a Graph representing the code in `m`

    # NOTE: torch.fx.symbolic_trace is a wrapper around a call to
    # fx.Tracer.trace and constructing a GraphModule. We'll
    # split that out in our transform to allow the caller to
    # customize tracing behavior.
    graph : torch.fx.Graph = tracer_class().trace(m)

    # Step 2: Modify this Graph or create a new one
    for node in graph.find_nodes(op="call_module",target="layer3"):
        new_node = graph.call_function(torch.relu, args=tuple([node]))
        graph.inserting_after(new_node)

    # Step 3: Construct a Module to return
    return torch.fx.GraphModule(m, graph)

print(transform(model))

