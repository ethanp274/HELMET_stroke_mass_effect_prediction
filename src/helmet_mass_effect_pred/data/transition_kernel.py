import pandas as pd
import numpy as np

def get_avg_transition_kernel(df, targets, target_transform):
    
    count_0 = 0
    count_1 = 0
    count_2 = 0
    count_3 = 0

    count_0_0 = 0
    count_0_1 = 0
    count_0_2 = 0
    count_0_3 = 0

    count_1_0 = 0
    count_1_1 = 0
    count_1_2 = 0
    count_1_3 = 0

    count_2_0 = 0
    count_2_1 = 0
    count_2_2 = 0
    count_2_3 = 0

    count_3_0 = 0
    count_3_1 = 0
    count_3_2 = 0
    count_3_3 = 0

    for i, mls in enumerate(df['size_mls']):
        current_class = target_transform(mls)
        if current_class == 0:
            count_0 += 1
            if targets[i] == 0:
                count_0_0 += 1
            elif targets[i] == 1:
                count_0_1 += 1
            elif targets[i] == 2:
                count_0_2 += 1
            elif targets[i] == 3:
                count_0_3 += 1
        elif current_class == 1:
            count_1 += 1
            if targets[i] == 0:
                count_1_0 += 1
            elif targets[i] == 1:
                count_1_1 += 1
            elif targets[i] == 2:
                count_1_2 += 1
            elif targets[i] == 3:
                count_1_3 += 1
        elif current_class == 2:
            count_2 += 1
            if targets[i] == 0:
                count_2_0 += 1
            elif targets[i] == 1:
                count_2_1 += 1
            elif targets[i] == 2:
                count_2_2 += 1
            elif targets[i] == 3:
                count_2_3 += 1
        elif current_class == 3:
            count_3 += 1
            if targets[i] == 0:
                count_3_0 += 1
            elif targets[i] == 1:
                count_3_1 += 1
            elif targets[i] == 2:
                count_3_2 += 1
            elif targets[i] == 3:
                count_3_3 += 1
    
    transition_kernel = pd.DataFrame(
        [
            [count_0_0 / count_0, count_0_1 / count_0, count_0_2 / count_0, count_0_3 / count_0],
            [count_1_0 / count_1, count_1_1 / count_1, count_1_2 / count_1, count_1_3 / count_1],
            [count_2_0 / count_2, count_2_1 / count_2, count_2_2 / count_2, count_2_3 / count_2],
            [count_3_0 / count_3, count_3_1 / count_3, count_3_2 / count_3, count_3_3 / count_3]
        ],
        columns = ['0', '1', '2', '3']
    )

    return transition_kernel


