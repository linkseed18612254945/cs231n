import tqdm


def train(dataloader, model, criterion, optimizer, flat=True, device=None, epoch_size=3):
    for epoch in range(1, epoch_size + 1):
        print(f"Epoch {epoch} start")
        model.train()
        for i, batch in tqdm.tqdm(list(enumerate(dataloader)), desc='Training:'):
            data = batch[0].to_devi
            label = batch[1]
            model.zero_grad()
            if flat:
                data = data.reshape(data.shape[0], -1)
            output = model.forward(data)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
    return model
