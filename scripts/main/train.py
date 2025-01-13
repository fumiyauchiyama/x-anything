def train_one_epoch(
        iter,
        epoch,
        model,
        data_loader,
        optimizer,
        sampler,
        scheduler,
        rank,
        world_size,
):
    model.train()
    sampler.set_epoch(epoch)
    for batch in data_loader:
        iter += 1
        optimizer.zero_grad()
        batch = batch.pin_memory() if batch.is_cuda else batch
        batch = batch.to(rank)
        loss = model(batch)
        loss.backward()
        optimizer.step()    
        scheduler.step()
    
    return iter