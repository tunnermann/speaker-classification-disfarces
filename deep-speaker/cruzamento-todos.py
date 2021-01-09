import random

import numpy as np

#from audio import read_mfcc
#from batcher import sample_from_mfcc
#from constants import SAMPLE_RATE, NUM_FRAMES
#from conv_models import DeepSpeakerModel
from test import batch_cosine_similarity

import pandas as pd

# Reproducible results.
np.random.seed(123)
random.seed(123)

# Define the model here.
#model = DeepSpeakerModel()

# Load the checkpoint.
#model.m.load_weights('ResCNN_triplet_training_checkpoint_265.h5', by_name=True)

df = pd.read_csv('../dataset_3_consolidado/todos.csv')

resultados = []
for i, linha in df.iterrows():
    for j in range(i, df.last_valid_index()):
        linha2 = df.loc[j]
        embeddings_1 = np.load('../'+linha['path'].replace('.wav', '.npy'))
        embeddings_2 = np.load('../'+linha2['path'].replace('.wav', '.npy'))
        score = batch_cosine_similarity(embeddings_1, embeddings_2)
        resultados.append([linha['voz'], linha['classe'], linha2['voz'], linha2['classe'], score])

df_resultado = pd.DataFrame(resultados, columns=['voz-1', 'classe-1', 'voz-2', 'classe-2','score'])
df_resultado.to_csv('resultado.csv', index=False)





# mfcc_001 = sample_from_mfcc(read_mfcc('../data/NORMAL/GM_NORMAL/GM6_NORMAL.wav', SAMPLE_RATE), NUM_FRAMES)
# mfcc_002 = sample_from_mfcc(read_mfcc('../data/DISFARCE/GM_DISFARCE/GM6_DISFARCE.wav', SAMPLE_RATE), NUM_FRAMES)

# # Call the model to get the embeddings of shape (1, 512) for each file.
# predict_001 = model.m.predict(np.expand_dims(mfcc_001, axis=0))
# predict_002 = model.m.predict(np.expand_dims(mfcc_002, axis=0))

# # Do it again with a different speaker.
# mfcc_003 = sample_from_mfcc(read_mfcc('../data/NORMAL/GM_NORMAL/GM1_NORMAL.wav', SAMPLE_RATE), NUM_FRAMES)
# predict_003 = model.m.predict(np.expand_dims(mfcc_003, axis=0))

# # Compute the cosine similarity and check that it is higher for the same speaker.
# print('SAME SPEAKER', batch_cosine_similarity(predict_001, predict_002)) # SAME SPEAKER [0.81564593]
# print('DIFF SPEAKER', batch_cosine_similarity(predict_001, predict_003)) # DIFF SPEAKER [0.1419204]