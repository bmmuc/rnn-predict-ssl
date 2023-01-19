from pytorch_lightning import Trainer
# from src.autoencoder import RecurrentAutoencoder
# from src.new_autoencoder import RecurrentAutoencoder2
# from src.vae_autoencoder import VaeAutoencoder
from src.pos_autoencoder import PositionAutoEncoder
from pytorch_lightning.loggers import WandbLogger


def train():
    MAX_EPOCHS = 128
    VAL_INTERVAL = 8
    WINDOW_SIZE = 10
    BATCH_SIZE = 128
    FEATURES = 40
    HIDDEN_SIZE = 256
    percentage_to_train = 0.9
    percentage_to_val = 0.1
    total_of_data = 5085

    # name = f'batch:{BATCH_SIZE}-hidden_layer_size: {HIDDEN_SIZE}\
    #     -epochs:{MAX_EPOCHS}\
    #     WINDOW_SIZE-{WINDOW_SIZE}' 

    # name = f'new act autoencoder 0.0 sigma hidden size: {HIDDEN_SIZE}, lr: e-3' 
    name = f'new pos autoencoder 0.0 sigma hidden size: {HIDDEN_SIZE}, lr: e-3' 

    wb_logger = WandbLogger(
		name=name,
        save_dir='./autoencoder',
		project='rnn-autoencoder-v3', 
		log_model='all',
		)

    # wb_logger = WandbLogger(
    #               name=name,
    #               project='rnn-predict-next-positions-with-v3', 
    #               save_dir='./next_positions',
    #               log_model='all'
    #             )

    # model = RecurrentAutoencoder(
    # window= WINDOW_SIZE, 
    # input_size= FEATURES , 
    # hidden_size= 512,
    # batch_size=BATCH_SIZE,
    # should_test_the_new_data_set=True,
    # data_root= '../all_data/data-v6',
    # num_of_data_to_train = int(total_of_data * percentage_to_train), 
    # num_of_data_to_val = int(total_of_data * percentage_to_val)
    # # num_of_data_to_val = 256
    # )

    # model = RecurrentAutoencoder2(
    # window= WINDOW_SIZE, 
    # input_size= FEATURES , 
    # hidden_size= HIDDEN_SIZE,
    # batch_size=BATCH_SIZE,
    # should_test_the_new_data_set=True,
    # data_root= '../all_data/data-3v3-v1',
    # num_of_data_to_train = 256, 
    # num_of_data_to_train = 26550, 
    # num_of_data_to_val = 1000
    # num_of_data_to_val = 256
    # )

    model = PositionAutoEncoder(
        window= WINDOW_SIZE, 
        input_size= 22, 
        hidden_size= HIDDEN_SIZE,
        batch_size=BATCH_SIZE,
        should_test_the_new_data_set=True,
        data_root= '../all_data/data-3v3-v2',
        # num_of_data_to_train = 256, 
        output_size=22,
        num_of_data_to_train = 30450, 
        num_of_data_to_val = 1000
        # num_of_data_to_val = 256
    )

    # model = ActAutoEncoder(
    #     window= WINDOW_SIZE, 
    #     input_size= 18 , 
    #     hidden_size= HIDDEN_SIZE,
    #     batch_size=BATCH_SIZE,
    #     should_test_the_new_data_set=True,
    #     data_root= '../all_data/data-3v3-v2',
    #     # num_of_data_to_train = 256, 
    #     output_size=18,
    #     num_of_data_to_train = 30450, 
    #     num_of_data_to_val = 1000,
    #     # num_of_data_to_val = 256
    # )

    # model = ConcatAutoEncoder(
    #     window= WINDOW_SIZE, 
    #     input_size= FEATURES , 
    #     hidden_size= HIDDEN_SIZE,
    #     batch_size=BATCH_SIZE,
    #     should_test_the_new_data_set=True,
    #     data_root= '../all_data/data-3v3-v2',
    #     # num_of_data_to_train = 256,
    #     pos_path = './autoencoder/rnn-autoencoder-v3/2maykc2c/checkpoints/epoch=255-step=60927.ckpt',
    #     act_path_autoencoder='./next_positions/rnn-predict-next-positions-with-v3/2a5vkx8o/checkpoints/epoch=255-step=60927.ckpt',
    #     output_size=40,
    #     num_of_data_to_train = 30450, 
    #     num_of_data_to_val = 1000
    #     # num_of_data_to_val = 256
    # )
    trainer = Trainer(gpus = 1, 
        max_epochs = MAX_EPOCHS, 
        logger=wb_logger,
        check_val_every_n_epoch = VAL_INTERVAL
    )

    # model.load_from_checkpoint('./autoencoder/rnn-autoencoder-v3/uo8bxtt2/checkpoints/epoch=255-step=60927.ckpt')
    wb_logger.watch(model)
    trainer.fit(model)


if __name__ == '__main__':
    train()
