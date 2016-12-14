from dataset import Dataset

ds = Dataset.load_wavs(data_folder='data', split=0.9, sample_length=1.0, cross_validation=5, downsampling=100)
