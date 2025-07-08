import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os 

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x): # forward feeding neural network
        x = F.relu(self.linear1(x)) # linear layer, then actuation function
        x = self.linear2(x)
        return x
    
    def save(self, file_name='model.pth'): # to save the model 
        model_folder_path = './model'  # new folder in current directory
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        
        file_name = os.path.join(model_folder_path, file_name) # join these together
        torch.save(self.state_dict, file_name)

class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimiser = optim.Adam(model.parameters(),lr=self.lr) # optimise the model
        self.criterion = nn.MSELoss()  # Mean Squared Error
    
    def train_step(self, state, action, reward, next_state, done):  # handle diff sizes
        # turn this into a pytorch tensor
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state,dtype=torch.float)
        action = torch.tensor(action,dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        # if this has multiple values this is already as (n,x)

        if len(state.shape) == 1:
            #if this is the case we only have one number (1,x)
            state = torch.unsqueeze(state, 0) # appends one dimension in the beginning
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

            #bellman equation simplified has to be used here 1: predicted Q values w current state
            pred = self.model(state)

            target = pred.clone() # iterate over tensors and apply this formula
            for index in range (len(done)):
                Q_new = reward[index] # reward of current index
                if not done[index]:
                    Q_new = reward[index] + self.gamma * torch.max(self.model(next_state[index]))

                target[index][torch.argmax(action).item()] = Q_new

            
            # 2: Q_new = r + y * max(next_predicted Q value) - this is all on ipad -> only do this if not done
            #code above is exactly the same as the formula here
            # need the Qnew in the same format to execute the formula
            #pred.clone()
            #preds[argmax(action)] = Q_new
            self.optimiser.zero_grad() # empty the gradient
            loss = self.criterion(target, pred) # calculate loss self.criterion (Qnew, Q)
            loss.backward() # back propergation

            self.optimiser.step()

    