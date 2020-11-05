import moviepy.editor as mp
import os
#import matplotlib.pyplot as plt
import numpy as np
import librosa
import panns_inference
from panns_inference import AudioTagging, SoundEventDetection, labels
import time
import json
from json.decoder import JSONDecodeError
from collections import defaultdict


def get_audio_tagging_result(clipwise_output, number_of_classes=10, classes_set=set(), threshold=0.2):
    """Visualization of audio tagging result.

    Args:
      clipwise_output: (classes_num,)
    """
    sorted_indexes = np.argsort(clipwise_output)[::-1]
    result = {}
    # Print audio tagging top probabilities
    for k in range(number_of_classes):
        if np.array(labels)[sorted_indexes[k]] in classes_set and clipwise_output[sorted_indexes[k]] > threshold:
            print('{}: {:.3f}'.format(np.array(labels)[sorted_indexes[k]], clipwise_output[sorted_indexes[k]]))
            result[np.array(labels)[sorted_indexes[k]]] = ['{:.3f}'.format(clipwise_output[sorted_indexes[k]])]
    return result
    

def store_result(new_result={}):
    with open('/home/nvr/airesults/ser.json', 'r') as f:
        try:
            prev = json.load(f)
        except JSONDecodeError:
            prev = {}
    result = defaultdict(list)

    for d in (prev, new_result):
        for key, value in d.items():
            result[key].extend(value)

    with open('/home/nvr/airesults/ser.json', 'w') as f:
        json.dump(dict(result), f)

'''
def plot_sound_event_detection_result(framewise_output):
    """Visualization of sound event detection result. 

    Args:
      framewise_output: (time_steps, classes_num)
    """
    out_fig_path = 'results/sed_result.png'
    os.makedirs(os.path.dirname(out_fig_path), exist_ok=True)

    classwise_output = np.max(framewise_output, axis=0) # (classes_num,)

    idxes = np.argsort(classwise_output)[::-1]
    idxes = idxes[0:5]

    ix_to_lb = {i : label for i, label in enumerate(labels)}
    lines = []
    for idx in idxes:
        line, = plt.plot(framewise_output[:, idx], label=ix_to_lb[idx])
        lines.append(line)

    plt.legend(handles=lines)
    plt.xlabel('Frames')
    plt.ylabel('Probability')
    plt.ylim(0, 1.)
    plt.savefig(out_fig_path)
    print('Save fig to {}'.format(out_fig_path))
'''

if __name__ == '__main__':
    """Example of using panns_inferece for audio tagging and sound evetn detection.
    """
    device = 'cpu'  # 'cuda' | 'cpu'
    at = AudioTagging(checkpoint_path=None, device=device)
    audio_path = "/home/nvr/converted_audio_files/"
    with open('/opt/iotistic-mnvr/config/default.json', 'r') as f:
        config = json.load(f)['SERParameters']
    classes = set(config['enabledclasses'].split('|'))
    threshold = config['threshold']

    while True:
        audio_files = os.listdir(audio_path)
        if not audio_files:
            time.sleep(0.1)
            continue
        t0 = time.time()
        # audio_path = 'resources/R9_ZSCveAHg_7s.wav'
        (audio, _) = librosa.core.load(audio_path + audio_files[0], sr=32000, mono=True)
        os.remove(audio_path + audio_files[0])
        audio = audio[None, :]  # (batch_size, segment_samples)

        print('------ Audio tagging ------')
        (clipwise_output, embedding) = at.inference(audio)
        """clipwise_output: (batch_size, classes_num), embedding: (batch_size, embedding_size)"""
        result = get_audio_tagging_result(clipwise_output[0], 10, classes, threshold)
        store_result(result)
        # print('------ Sound event detection ------')
        # sed = SoundEventDetection(checkpoint_path=None, device=device)
        # framewise_output = sed.inference(audio)
        # """(batch_size, time_steps, classes_num)"""

        # plot_sound_event_detection_result(framewise_output[0])

        print('done file within ', time.time() - t0)
        
        ########### run store_result on another thread?
