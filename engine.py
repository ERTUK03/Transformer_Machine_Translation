import torch

def train_step(model: torch.nn.Module,
               train_dataloader: torch.utils.data.DataLoader,
               optimizer: torch.optim.Optimizer,
               criterion: torch.nn.Module,
               scheduler: torch.optim.lr_scheduler,
               device: torch.device):

    running_loss = 0.
    last_loss = 0.

    model.train()

    for i, (inputs, targets) in enumerate(train_dataloader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        outputs = model(inputs, targets)

        outputs = outputs.view(-1, outputs.size(-1))
        targets = targets.view(-1)
        loss = criterion(outputs, targets)
        loss.backward()

        optimizer.step()
        scheduler.step()

        running_loss += loss.item()
        if i % 100 == 99:
            last_loss = running_loss /100
            print(f'batch {i+1} loss: {last_loss}')
            running_loss = 0.

    return last_loss

def test_step(model: torch.nn.Module,
              test_dataloader: torch.utils.data.DataLoader,
              criterion: torch.nn.Module,
              device: torch.device):

    running_vloss = 0.
    model.eval()

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(test_dataloader):
            inputs = inputs.to(device)
            annotations = targets.to(device)
            outputs = model(inputs, targets)
            outputs = outputs.view(-1, outputs.size(-1))
            targets = targets.view(-1)
            vloss = criterion(outputs, targets)
            running_vloss += vloss.item()
    avg_loss = running_vloss / (i + 1)
    return avg_loss

def train(epochs: int,
          model: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          criterion: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          model_path: str,
          scheduler: torch.optim.lr_scheduler,
          device: torch.device):

    best_vloss = 1000000

    for epoch in range(1,epochs+1):
        print(f'EPOCH {epoch}')
        avg_loss = train_step(model, train_dataloader, optimizer, criterion, scheduler, device)
        avg_vloss = test_step(model, test_dataloader, criterion, device)
        print(f'LOSS train {avg_loss} test {avg_vloss}')
        if avg_vloss<best_vloss:
            best_vloss = avg_vloss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_vloss
                }, f'{model_path}_{epoch}.pth')
