from pytorch_lightning import Trainer
from src.model import Predictor
from pytorch_lightning.loggers import WandbLogger

def train():
    WINDOW_SIZE = 10
    BATCH_SIZE = 256
    HIDDEN_SIZE = 512
    WEIGHT_DECAY = 0
    MAX_EPOCHS = 128
    VAL_INTERVAL = 8
    HIDDEN_IS_WITH_NOISE = True
    NUM_OF_DATA_TO_TEST = 1000
    LEARNING_RATE = 1e-4
    SCLAER = False
    percentage_to_train = 0.9
    percentage_to_val = 0.05
    total_of_data = 50000


    name = f'batch:{BATCH_SIZE}-hidden_layer_size: {HIDDEN_SIZE}\
	    -weight_decay:{WEIGHT_DECAY}-epochs:{MAX_EPOCHS}\
	    -hidden_is_with_noise:{HIDDEN_IS_WITH_NOISE}\
      -data-v4-with-2-rnn-9-linear\
      overffting - false\
      -learning_rate:{LEARNING_RATE}\
      scaler-{SCLAER}\
      WINDOW_SIZE-{WINDOW_SIZE}'  

    wb_logger = WandbLogger(
		name=name,
		project='rnn-predict-next-positions', 
		log_model='all',
    # settings=wandb.Settings(start_method="fork")
		)

    model = Predictor(
	    input_size = 16, hidden_size = HIDDEN_SIZE, out_size = 16,
	    num_layers1= 1, num_layers2= 1,
	    num_workers= 8, batch_size=BATCH_SIZE,
      num_of_data_to_test = NUM_OF_DATA_TO_TEST,
	    num_of_data_to_train = int(total_of_data * percentage_to_train), 
	    num_of_data_to_val = int(total_of_data * percentage_to_val), 
      data_root= '../all_data/data-v4',
      should_test_overffit = False,
      lr=LEARNING_RATE, weight_decay= WEIGHT_DECAY, 
      window=WINDOW_SIZE
      )
    
    # model = model.load_from_checkpoint('./epoch-127.ckpt')
    trainer = Trainer(gpus = 1, 
    max_epochs = MAX_EPOCHS, 
		logger=wb_logger,
    check_val_every_n_epoch = VAL_INTERVAL)
    
    # trainer = Trainer(gpus = 1, max_epochs = MAX_EPOCHS, val_check_interval   = 1)
    
    # trainer = Trainer(gpus = 1, max_epochs = MAX_EPOCHS, val_check_interval = 1)
    # model = model.load_from_checkpoint('../models/epoch-511.ckpt')
    wb_logger.watch(model)
    trainer.fit(model)
  
    # if(is_to_test):
    #     # trainer.test(model)
    #     test()

if __name__ == '__main__':
    train()
