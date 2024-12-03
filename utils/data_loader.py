from glob import glob
import pandas as pd

#load subtitles and process text and episode number
def load_subtitles(subtitles_dataset_path):
    paths = glob(subtitles_dataset_path + '/*.ass')
    
    scripts = []
    episodes_num = []

    for path in paths:
        with open(path, 'r', encoding="utf8") as file:
            lines = file.readlines()
            lines = lines[27:]
            lines = [','.join(line.replace('\\N', ' ').strip().split(',')[9:]) for line in lines ]

        script = ' '.join(lines)
        episode_num = int(path.split('-')[-1].split('.')[0].strip())

        scripts.append(script)
        episodes_num.append(episode_num)

    df = pd.DataFrame.from_dict({'Episode' : episodes_num, 'Script' : scripts})

    return df