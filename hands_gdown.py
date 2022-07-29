import gdown

# a file
url = "https://drive.google.com/uc?id=1WnV5gdichlXt9MNJAnL3bqtRx18W0ehd"
output = "data/output/HAnDS_figer_types_stage_one_state_two_sentences_stage_three_pp.tar.gz"
gdown.download(url, output, quiet=False)
