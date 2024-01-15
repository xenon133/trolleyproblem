import torch.nn
import data_provider as dp
from torch.utils.data import DataLoader

class TrolleyClassifier(torch.nn.Module):

    def __init__(self, num_risk_values):
        super().__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Linear(in_features=num_risk_values*2, out_features=32, bias=False),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(in_features=32, out_features=64, bias=False),
            torch.nn.ReLU(inplace=True)
        )
        self.output = torch.nn.Sequential(
            torch.nn.Linear(in_features=64, out_features=2, bias=False),
            torch.nn.Softmax(dim=0)
        )

    def forward(self, x):
        output = self.layer1(x)
        output = self.output(output)
        return output

def train_trolley(model, trainloader, epochs, verbose):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = torch.nn.BCELoss()

    for i in range(epochs):
        if(verbose):
            print(f"Epoch: {i+1} -----------------")
        model.train()
        for batch_number, data in enumerate(trainloader):
            risk_values, labels = data
            pred = model(risk_values)
            loss = loss_fn(pred, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if(verbose):
                print(f"[{batch_number+1}/{len(trainloader)}] - Loss: {loss}")

    torch.save(model.state_dict(), "trolley.pth")

def init_cl():
    num_risk_values = 24
    trolley = TrolleyClassifier(num_risk_values=num_risk_values)
    trolley.load_state_dict(torch.load("trolley_49.pth"))
    return trolley