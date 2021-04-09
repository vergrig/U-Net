def calc_loss(pred, target, metrics, bce_weight=0.5):
    target = target.type_as(pred)
    bce = F.binary_cross_entropy_with_logits(pred, target)

    pred = torch.sigmoid(pred)
    dice = dice_loss(pred, target)

    loss = bce * bce_weight + dice * (1 - bce_weight)

    metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)

    return loss, metrics

def string_metrics(metrics, epoch_samples):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:.3f}".format(k, metrics[k] / epoch_samples))

    return (", ".join(outputs))
    
def string_time(elapsed):
    return "%im %is" %(int(elapsed / 60), int(elapsed % 60))




BATCH_PRINT = len(train_dataloader) // 5

def train_model(model, optimizer, scheduler, num_epochs=25):
    
    start = time()

    #best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10

    for epoch in range(num_epochs):
        
        print('\n======== Epoch %i / %i ========' %(epoch + 1, num_epochs))
        print('\n======== Epoch %i / %i ========' %(epoch + 1, num_epochs), file=logs)
        print('Training...')
        print('Training...', file=logs)
        
        # =========== training ==========

        metrics = defaultdict(float)
        epoch_samples = 0
        model.train()
        
        for step, batch in enumerate(train_dataloader):
            inputs, labels = batch[0].to(device), batch[1].to(device)

            if step % BATCH_PRINT == 0 and not step == 0:
                print('  Batch %i  of  %i;\tElapsed time: ' %(step + 1, len(train_dataloader)) + string_time(time() - start))
                print('  Batch %i  of  %i;\tElapsed time: ' %(step + 1, len(train_dataloader)) + string_time(time() - start), file=logs)
                #print('      ' + str(step + 1) + " batch training loss: "  + string_metrics(metrics, epoch_samples))
                
            optimizer.zero_grad()

            outputs = model(inputs)
            loss, metrics = calc_loss(outputs, labels, metrics)
            epoch_samples += inputs.size(0)
            
            
            
            loss.backward()
            optimizer.step()
            # del outputs
            # del inputs
            # del labels
            # del loss
    
        print("\n  Average training loss: "  + string_metrics(metrics, epoch_samples))
        print("\n  Average training loss: "  + string_metrics(metrics, epoch_samples), file=logs)
        scheduler.step()
        
        # =========== validating ===========
        
        print("\nValidating...")
        print("\nValidating...", file=logs)
        
        metrics = defaultdict(float)
        epoch_samples = 0
        model.eval()
        
        for step, batch in enumerate(val_dataloader):
          
            inputs, labels = batch[0].to(device), batch[1].to(device)

            with torch.no_grad():
                outputs = model(inputs)
                loss, metrics = calc_loss(outputs, labels, metrics)

            epoch_samples += inputs.size(0)
            
            # del outputs
            # del inputs
            # del labels
            # del loss

        epoch_loss = metrics['loss'] / epoch_samples
        
        # ============= logging ==============
        
        if epoch_loss < best_loss:
            # print("Saving best model")
            best_loss = epoch_loss
            # best_model_wts = copy.deepcopy(model.state_dict())

        print("\n  Average validation loss: "  + string_metrics(metrics, epoch_samples))
        print("\n  Average validation loss: "  + string_metrics(metrics, epoch_samples), file=logs)
        
        print("  Elapsed time: " + string_time(time() - start))
        print("  Elapsed time: " + string_time(time() - start), file=logs)


    print('Best val loss: {:.4f}'.format(best_loss))
    print('Best val loss: {:.4f}'.format(best_loss), file=logs)
    # model.load_state_dict(best_model_wts)
    
    return model