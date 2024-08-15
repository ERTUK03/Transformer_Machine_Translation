import torch
from download_dataset import download_dataset
from read_data import read_data
from data_preprocessing import preprocess_data, DataPreProcessor
from data_tokenization import tokenize_data
from get_dataloaders import get_dataloaders
from engine import train
from model.transformer import Transformer
from utils.custom_loss_scheduler import CustomLRScheduler
from utils.config_parser import load_config, type_convert

config_data = load_config('config.ini')

URL = config_data['url']
DIR_NAME = config_data['dir_name']
SRC_DATA = config_data['src_data']
TGT_DATA = config_data['tgt_data']
BATCH_SIZE = config_data['batch_size']
TRAIN_SIZE = config_data['train_size']
TOKEN_NAME = config_data['token_name']
MAX_LEN = config_data['max_length']
START_TOKEN = config_data['start_token']
STOP_TOKEN = config_data['stop_token']
PAD_TOKEN = config_data['pad_token']
MAX_VOCAB = config_data['max_vocab']
D_MODEL = config_data['d_model']
D_FF = config_data['d_ff']
NUM_HEADS = config_data['attn_heads']
NUM_LAYERS = config_data['num_layer']
MAX_SEQ_LENGTH = config_data['max_seq_length']
BETA_1 = config_data['beta_1']
BETA_2 = config_data['beta_2']
EPSILON = config_data['eps']
LABEL_SMOOTHING = config_data['label_smoothing']
WARMUP_STEPS = config_data['warmup_steps']
EPOCHS = config_data['epochs']
MODEL_NAME = config_data['name']

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

download_dataset(URL, DIR_NAME)
data = read_data(f'{DIR_NAME}/{SRC_DATA}', f'{DIR_NAME}/{TGT_DATA}')
preprocessed_data = preprocess_data(data, MAX_LEN, START_TOKEN, STOP_TOKEN)
tokenized_data_pairs = tokenize_data(preprocessed_data, [PAD_TOKEN, START_TOKEN, STOP_TOKEN], MAX_VOCAB, TOKEN_NAME)
train_dataloader, test_dataloader = get_dataloaders(tokenized_data_pairs, BATCH_SIZE, TRAIN_SIZE)
model = Transformer(MAX_VOCAB, D_MODEL, D_FF, NUM_HEADS, NUM_LAYERS, MAX_SEQ_LENGTH, DEVICE).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=0, betas=(BETA_1, BETA_2), eps=EPSILON)
criterion = torch.nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING, ignore_index=0)
scheduler = CustomLRScheduler(optimizer, D_MODEL, WARMUP_STEPS)

train(EPOCHS, model, optimizer, criterion, train_dataloader, test_dataloader, MODEL_NAME, scheduler, DEVICE)
