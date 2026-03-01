import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from function.load_graph_data_list import HeteroGraphDataList, CrossUrbanData
from function.logger import get_logger
from args import get_default_arguments
from datetime import datetime
from model.DACTModel import DACTModel
from function.llm_agent import LLMAgent
from data.llmargs import get_llm_args
from function.tools import get_device, node_to_graph, combine_inter_region_sub
from function.downstream_task import predict_crime, predict_check

# Set up arguments
args = get_default_arguments()

# Set up logger
log_folder = args['code_test']['log_folder']
logger_name = args['code_test']['logger_name']
log_level = args['code_test']['log_level']
log_file_name = logger_name + '_' + datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + '.log'
logger = get_logger(log_folder, logger_name, log_file_name, level=log_level)

logger.info('Start code test')

# LLM agent
llm_args = get_llm_args()
# llm = LLMAgent(llm_args)
llm = None
logger.info('llm agent loaded')
# test_text = 'I am a good boy'
# embedding = llm.get_embedding(test_text)
# logger.info('embedding: %s', embedding)

# Load graph data
device = get_device()
logger.info('Device: %s', device)


# Load graph data list
hkg_list = HeteroGraphDataList(kg_dir_list=args['data']['kg_dir_list'],
                               reverse=args['data']['kg_reverse'],
                               export=args['data']['kg_export'],
                               graph_device=device, spatial_list=args['data']['spatial_list'])
logger.info('kg data list loaded')
# Initialize graph features
hkg_list.init_hetero_graph_features(image_list=args['data']['image_list'], flow_list=args['data']['flow_list'],
                                    feature_dir_list=args['data']['hg_LLM_feature_dir_list'],
                                    node_feats_dim=args['model']['node_dim'][0],
                                    edge_feats_dim=args['model']['edge_in_dim'],
                                    graph_device=device)
logger.info('kg feature list loaded...')
# Construct multi-level graph coordinator
hkg_list.construct_multi_level_graph_coordinator(llm=llm,
                                                 hg_LLM_feature_dir_list=args['data']['hg_LLM_feature_dir_list'],
                                                 coordinator_link_threshold=args['data']['coordinator_link_threshold'],
                                                 node_feats_dim=args['model']['node_dim'][0], edge_feats_dim=args['model']['edge_dim'], )
logger.info('multi-level graph coordinator constructed...')


cross_urban_data = CrossUrbanData(args=args['data'], hkg_list=hkg_list, graph_device=device)
logger.info('cross urban data loaded...')
sp_pos_samples, sp_neg_samples, mobility_samples, flow_in_samples, flow_out_samples, inter_pair_samples = cross_urban_data.get_all_samples(if_pre_solve=False)
logger.info('all samples loaded...')

city_num = len(args['data']['kg_dir_list'])
# Model
model = DACTModel(args['model'], city_num).to(device)
logger.info('DACTModel loaded')
optimizer = torch.optim.Adam(model.parameters(), lr=args['training']['lr'], weight_decay=args['training']['weight_decay'])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

# Training
epochs = args['training']['epochs']
intra_batch_size = args['model']['intra_batch_size']
intra_batch_num = 0
for city in range(city_num):
    intra_batch_num = max(intra_batch_num, len(sp_pos_samples[city]) // intra_batch_size[city])

inter_batch_size = args['model']['inter_batch_size']
inter_batch_num = 0
for city in range(city_num):
    inter_batch_num = max(inter_batch_num, len(inter_pair_samples[city][city]) // inter_batch_size[city])

batch_num = intra_batch_num

logger.info('Intra batch size: %s Intra batch num: %s Inter batch size: %s Inter batch num: %s',
            intra_batch_size, intra_batch_num, inter_batch_size, inter_batch_num)

train_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

loss_list = []
loss_spatial_list = []
loss_mobility_list = []
loss_cross_list = []
loss_pred_list = []
cuda_memory_list = []
cri_mae_list = []
cri_rmse_list = []
cri_r2_list = []
check_mae_list = []
check_rmse_list = []
check_r2_list = []
for city in range(city_num):
    cri_mae_list.append([])
    cri_rmse_list.append([])
    cri_r2_list.append([])
    check_mae_list.append([])
    check_rmse_list.append([])
    check_r2_list.append([])

for epoch in tqdm(range(epochs)):
    # log empty
    logger.info('')

    sp_pos_samples_batch, sp_neg_samples_batch, mobility_samples_batch, flow_in_samples_batch, flow_out_samples_batch = cross_urban_data.get_all_intra_samples_batch(
        sp_pos_samples, sp_neg_samples,
        mobility_samples, flow_in_samples, flow_out_samples,
        intra_batch_size,
        intra_batch_num, if_pre_solve = False)
    inter_pair_samples_batch = cross_urban_data.get_all_inter_samples_batch(inter_pair_samples, inter_batch_size, inter_batch_num)

    model.train()
    epoch_loss = 0
    epoch_loss_spatial = 0
    epoch_loss_mobility = 0
    epoch_loss_cross = 0
    epoch_loss_pred = 0
    out_emb_list = []
    for city in range(city_num):
        out_emb_list.append(np.empty((0, args['model']['out_dim'])))

    for batch in range(batch_num):
        optimizer.zero_grad()

        # Convert node dict to graph Intra
        sp_pos_samples_graph = []
        sp_neg_samples_graph = []
        for city in range(city_num):
            sp_pos_samples_graph_city = []
            sp_neg_samples_graph_city = []
            for sample in sp_pos_samples_batch[batch][city]:
                sp_pos_samples_graph_city.append(node_to_graph(sample, hkg_list.urban_graph_list[city]))
            for sample in sp_neg_samples_batch[batch][city]:
                sp_neg_samples_graph_city.append(node_to_graph(sample, hkg_list.urban_graph_list[city]))
            sp_pos_samples_graph.append(sp_pos_samples_graph_city)
            sp_neg_samples_graph.append(sp_neg_samples_graph_city)

        # Convert node dict to graph Inter
        inter_samples_graph = []
        for city in range(city_num):
            inter_samples_graph_city = []
            for sample in inter_pair_samples_batch[batch][city]:
                city1 = sample[0][0]
                city2 = sample[0][1]
                region_node_dict_1 = sample[0][4][0]
                region_node_dict_2 = sample[0][4][1]
                inter_samples_graph_city.append(combine_inter_region_sub(hkg_list.urban_graph_list[city1],
                                                                         hkg_list.urban_graph_list[city2],
                                                                         region_node_dict_1, region_node_dict_2, ))
            inter_samples_graph.append(inter_samples_graph_city)

        # Compute loss
        loss, loss_spatial, loss_mobility, loss_pred, out_emb = model(sp_pos_samples_graph,
                                                                                  sp_neg_samples_graph,
                                                                                  sp_pos_samples_batch[batch],
                                                                                  sp_neg_samples_batch[batch],
                                                                                  mobility_samples_batch[batch],
                                                                                  flow_in_samples_batch[batch],
                                                                                  flow_out_samples_batch[batch],
                                                                                  None,
                                                                                  device, args['training'])

        for city in range(city_num):
            out_emb_list[city] = np.concatenate((out_emb_list[city], out_emb[city].detach().cpu().numpy()))

        epoch_loss += loss.item()
        epoch_loss_spatial += loss_spatial.item()
        epoch_loss_mobility += loss_mobility.item()
        epoch_loss_cross += loss_cross.item()
        epoch_loss_pred += loss_pred.item()

        # Backward pass and optimization
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
        optimizer.step()

        # Record CUDA memory usage
        cuda_memory_list.append(torch.cuda.memory_allocated(device))

    # Compute average loss for the epoch
    avg_loss = epoch_loss / batch_num
    loss_list.append(avg_loss)
    avg_loss_spatial = epoch_loss_spatial / batch_num
    loss_spatial_list.append(avg_loss_spatial)
    avg_loss_mobility = epoch_loss_mobility / batch_num
    loss_mobility_list.append(avg_loss_mobility)
    avg_loss_cross = epoch_loss_cross / batch_num
    loss_cross_list.append(avg_loss_cross)
    avg_loss_pred = epoch_loss_pred / batch_num
    loss_pred_list.append(avg_loss_pred)

    # Update learning rate
    scheduler.step(avg_loss)

    # Log epoch loss
    logger.info(f'Epoch {epoch + 1}, '
                f'Loss: {avg_loss}, '
                f'Spatial Loss: {avg_loss_spatial}, '
                f'Mobility Loss: {avg_loss_mobility}, '
                f'Pred Loss: {avg_loss_pred}')

    with torch.no_grad():
        cur_out_emb_list = out_emb_list
        for city in range(city_num):
            cri_mae, cri_rmse, cri_r2 = predict_crime(cur_out_emb_list[city])
            check_mae, check_rmse, check_r2 = predict_check(cur_out_emb_list[city])

            cri_mae_list[city].append(cri_mae)
            cri_rmse_list[city].append(cri_rmse)
            cri_r2_list[city].append(cri_r2)
            check_mae_list[city].append(check_mae)
            check_rmse_list[city].append(check_rmse)
            check_r2_list[city].append(check_r2)

            logger.info(f'Epoch {epoch + 1}, '
                        f'City {city + 1}, '
                        f'Crime MAE: {cri_mae}, '
                        f'Crime RMSE: {cri_rmse}, '
                        f'Crime R2: {cri_r2}, '
                        f'Check MAE: {check_mae}, '
                        f'Check RMSE: {check_rmse}, '
                        f'Check R2: {check_r2}')

# Plot CUDA memory usage
plt.plot(cuda_memory_list)
plt.title('CUDA Memory Usage')
plt.xlabel('Batch')
plt.ylabel('Memory (Bytes)')
plt.show()

# Plot loss curve
plt.plot(loss_list)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# Plot spatial loss curve
plt.plot(loss_spatial_list)
plt.title('Spatial Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# Plot mobility loss curve
plt.plot(loss_mobility_list)
plt.title('Mobility Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# Plot cross loss curve
plt.plot(loss_cross_list)
plt.title('Cross Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# Plot pred loss curve
plt.plot(loss_pred_list)
plt.title('Pred Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# Plot crime MAE, RMSE, R2, check MAE, RMSE, R2 in one figure
fig, axs = plt.subplots(2, 3)
fig.suptitle('Crime and Check Prediction Metrics')
for city in range(city_num):
    axs[0, 0].plot(cri_mae_list[city], label=f'City {city + 1}')
    axs[0, 1].plot(cri_rmse_list[city], label=f'City {city + 1}')
    axs[0, 2].plot(cri_r2_list[city], label=f'City {city + 1}')
    axs[1, 0].plot(check_mae_list[city], label=f'City {city + 1}')
    axs[1, 1].plot(check_rmse_list[city], label=f'City {city + 1}')
    axs[1, 2].plot(check_r2_list[city], label=f'City {city + 1}')
axs[0, 0].set_title('Crime MAE')
axs[0, 1].set_title('Crime RMSE')
axs[0, 2].set_title('Crime R2')
axs[1, 0].set_title('Check MAE')
axs[1, 1].set_title('Check RMSE')
axs[1, 2].set_title('Check R2')
plt.show()

logger.info('End of train')
