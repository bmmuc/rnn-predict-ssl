import ipdb

def convert_pred_to_obs(arr_pos, arr_act):
    obs = [0] * 40


    index_pos = [0, 1, 2, 3, 4, 5, 6, 7, 11, 12, 13, 14, 18, 19, 20, 21, 25, 26, 30, 31, 35, 36]
    index_act = [8, 9, 10, 15, 16, 17, 22, 23, 24, 27, 28, 29, 32,33, 34, 37, 38, 39]

    last_index_pos_used, las_index_act_used = 0, 0

    for i in range(40):
        if i in index_pos:
            # idx = arr_pos.index(i)
            obs[i] = arr_pos[last_index_pos_used]
            last_index_pos_used += 1

        else:
            # idx = arr_act.index(i)

            obs[i] = arr_act[las_index_act_used]
            las_index_act_used += 1

    return obs


if __name__ == '__main__':
    # ipdb.set_trace()
    case_test = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]

    arr_pos_test = [0, 1, 2, 3, 4, 5, 6, 7, 11, 12, 13, 14, 18, 19, 20, 21, 25, 26, 30, 31, 35, 36]

    arr_act_test = [8, 9, 10, 15, 16, 17, 22, 23, 24, 27, 28, 29, 32,33, 34, 37, 38, 39]

    result = convert_pred_to_obs(arr_pos_test, arr_act_test)

    print(result == case_test)
            
