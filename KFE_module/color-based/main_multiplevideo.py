from utilities import *

#############################################
#############################################
#############################################

if __name__ == '__main__':
    global_path = ".../KFE_deep_learning/dataset"
    save_result_to = ".../KFE_deep_learning/results/time_performance"
    dsample_rate = 10   # take a frame per dsample_rate frames
    clip_period = 60   # second, need to be greater than 120*dsample_rate/fps ~= 4*dsample_rate (svd 60 dimension constraint)
    similarity_threshold = 0.9  # squared
    CLUSTER_THRESHOLD = 30
    BACKGROUND_REMOVAL = False

    time1 = {}
    time2 = {}
    time3 = {}
    num_kfs = {}
    for folderpath in os.listdir(global_path):
        video_path = os.path.join(global_path, folderpath, folderpath) + '.mp4'
        save_to = os.path.join(global_path, folderpath, 'keyframes')  # path to save the key frames
        time_ch, time_kfs, time_save, num_kf = full_KFE_onetime(video_path, save_to, dsample_rate, clip_period,
                                                                similarity_threshold, CLUSTER_THRESHOLD, BACKGROUND_REMOVAL,
                                                                clu_method='dynamic', K=12)
        time1[folderpath] = time_ch
        time2[folderpath] = time_kfs
        time3[folderpath] = time_save
        num_kfs[folderpath] = num_kf
        print(f"{folderpath} keyframes have been extracted.")

    # for saving the time performance
    dict_list = [time1, time2, time3, num_kfs]
    df = pd.DataFrame(dict_list)
    output_path = os.path.join(save_result_to, 'Polo3.xlsx')
    with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Sheet1', index=False)
    print('The output sheet is saved to', output_path)
    print("KFE done.")