from reward_predictor import RewardPredictor
import torch

observation_dims = (3, 64, 64)
n_actions = 3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = RewardPredictor(observation_dims, n_actions, device, max_len=50).to(device)

observations = torch.rand((50, 32, 3, 64, 64), device=device)
actions = torch.vstack( (torch.tensor([1,0,0], device=device),) * 32)
actions = torch.unsqueeze(actions, 0)
actions = torch.vstack( (actions,) * 50)

out = model.forward(observations, actions)

print(out)
print(out.shape)

# def get_data_loaders(dataset, batch_size=1024, validation_subset=0, seed=42):

#     if validation_subset > 0:
#         n_total_samples = len(dataset)
#         n_train_samples = math.floor(n_total_samples * (1-validation_subset))
#         n_valid_samples = n_total_samples - n_train_samples

#         train_dataset, valid_dataset = random_split(
#             dataset,
#             [n_train_samples, n_valid_samples],
#             generator=torch.Generator().manual_seed(seed)
#         )  # reproducible results

#         train_loader = DataLoader(train_dataset, batch_size=batch_size)
#         valid_loader = DataLoader(valid_dataset, batch_size=batch_size)

#         print('Train set size:', len(train_dataset), 'samples')
#         print('Train set:', len(train_loader), 'batches')
#         print('Validation set size:', len(valid_dataset), 'samples')
#         print('Validation set:', len(valid_loader), 'batches')
#     else:
#         train_loader = DataLoader(dataset, batch_size=batch_size)
#         valid_loader = None
#         print('Prepared:', len(train_loader), 'batches')

#     return train_loader, valid_loader

# # Load dataset

# path_to_dataset = '/content/drive/MyDrive/Self-Attention/datasets/dataset-perfect-agent-1.pt'
# dataset = torch.load(path_to_dataset)

# validation_size = 0.2
# batch_size = 1024
# train_loader, valid_loader = get_data_loaders(dataset,
#                                               batch_size=batch_size,
#                                               validation_subset=validation_size)


# def evaluate(model, criterion, data_loader, device, verbose=False):
#     losses = []
#     accuracies = []

#     for batch_idx, batch in enumerate(data_loader):
#         observations, actions, rewards = batch

#         observations = observations.transpose(0, 1).to(device)
#         actions = actions.transpose(0, 1).to(device)
#         rewards = rewards.transpose(0, 1).to(device)

#         output = model(observations, actions)

#         # reshape for CrossEntropyLoss
#         output = output.permute(1, 2, 0)
#         rewards = rewards.transpose(0, 1)

#         loss = criterion(output, rewards)
#         preds = output.argmax(dim=1)
#         masked_preds = preds[rewards != pad_val]
#         masked_rewards = rewards[rewards != pad_val]
#         accuracy = torch.sum(masked_preds == masked_rewards) / \
#             masked_rewards.numel()

#         losses.append(loss.item())
#         accuracies.append(accuracy.item())

#         del observations
#         del actions
#         del rewards
#         del output

#     mean_loss = sum(losses) / len(losses)
#     mean_acc = sum(accuracies) / len(accuracies)

#     del losses
#     del accuracies

#     if verbose:
#         print('mean_loss:', mean_loss)
#         print('mean_acc:', mean_acc)

#     return mean_loss, mean_acc


# RESUME = True
# CHECKPOINT_PATH = 'checkpoint-vloss-0.0502'
# RUN_PATH = 'runs/run-1'

# # Training hyperparameters
# num_epochs = 10
# learning_rate = 3e-3
# batch_size = 1024
# # TODO: second weight might be a typo in the paper, consider 0.002
# class_weights = torch.tensor([0.499, 0.02, 0.499]).to(device)

# # Model hyperparameters
# max_len = 201

# model = SelfAttentionForRL(
#     observation_size, action_size, device, verbose=False).to(device)

# step = 0
# start_epoch = 0

# optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#     optimizer, factor=0.1, patience=10, verbose=True
# )

# criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=pad_val)

# if RESUME and os.path.isfile(CHECKPOINT_PATH):
#     print(f'> Loading from checkpoint at {CHECKPOINT_PATH}')
#     checkpoint = torch.load(CHECKPOINT_PATH)
#     model.load_state_dict(checkpoint['model_state_dict'])
#     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#     scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
#     start_epoch = checkpoint['epoch']
#     step = checkpoint['step']
#     RUN_PATH = checkpoint['run_path']
#     del checkpoint
#     print(f'> Resuming run at: {RUN_PATH}')
# else:
#     print(f'> Checkpoint not found. Starting new run at: {RUN_PATH}...')

# best_val_loss = 1e9

# for epoch in range(start_epoch, num_epochs):
#     print(f"> Epoch {epoch+1}/{num_epochs}", end=' ')

#     losses = []
#     is_best_model = True

#     pbar = tqdm(train_loader)

#     for batch_idx, batch in enumerate(pbar):
#         observations, actions, rewards = batch

#         observations = observations.transpose(0, 1).to(device)
#         actions = actions.transpose(0, 1).to(device)
#         rewards = rewards.transpose(0, 1).to(device)

#         output = model(observations, actions)

#         # Reshape output for K-dimensional CrossEntropy loss
#         output = output.permute(1, 2, 0)
#         rewards = rewards.transpose(0, 1)

#         # Compute loss
#         optimizer.zero_grad()

#         loss = criterion(output, rewards)
#         losses.append(loss.item())

#         loss.backward()

#         # Clip to avoid exploding gradient issues
#         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

#         optimizer.step()

#         # plot to tensorboard
#         model.eval()
#         val_loss, val_acc = evaluate(model, criterion, valid_loader, device)

#         preds = output.argmax(dim=1)
#         masked_preds = preds[rewards != pad_val]
#         masked_rewards = rewards[rewards != pad_val]
#         acc = torch.sum(masked_preds == masked_rewards) / \
#             masked_rewards.numel()

#         writer.add_scalar("Training loss", loss, global_step=step)
#         writer.add_scalar("Training acc", acc, global_step=step)
#         writer.add_scalar("Validation loss", val_loss, global_step=step)
#         writer.add_scalar("Validation acc", val_acc, global_step=step)

#         pbar.set_postfix_str(
#             f'loss: {loss:0.5f}, acc: {acc:0.5f}, val_loss: {val_loss:0.5f}, val_acc: {val_acc:0.5f}')

#         model.train()

#         step += 1

#         del observations
#         del actions
#         del rewards
#         del output
#         del preds

#     is_best_model = val_loss < best_val_loss

#     if is_best_model:
#         print(f'> Saving checkpoint with val loss: {val_loss:.4f}...')
#         best_val_loss = val_loss
#         torch.save({
#             'model_state_dict': model.state_dict(),
#             'optimizer_state_dict': optimizer.state_dict(),
#             'scheduler_state_dict': scheduler.state_dict(),
#             'epoch': epoch,
#             'run_path': RUN_PATH,
#             'step': step
#         }, f'checkpoint-vloss-{val_loss:.4f}')
#     else:
#         print(f'> NOT saving checkpoint, val loss: {val_loss:.4f}...')

#     # free some GPU memory
#     torch.cuda.empty_cache()

#     mean_loss = sum(losses) / len(losses)
#     scheduler.step(mean_loss)
