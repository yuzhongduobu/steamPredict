from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model

model = load_model('../tmp/model/steam.h5')
model.summary()