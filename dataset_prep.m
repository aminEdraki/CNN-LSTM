clear
close all
clc

dataset_path = 'C:\Users\18ae5\Desktop\Datasets\ASR\Gabor';
% features_path = 'D:\Datasets\Tensorflow_ASR_Pre';

fs = 8000;
train_path = '/train/audio/';
% test_path = '/test/audio/';

% loop over files


folders = dir([dataset_path, train_path]);
for i = 3:length(folders)
    folder = folders(i);
    if ~folder.isdir
        continue
    end
    files = dir([folder.folder, '/', folder.name]);
    
    parfor j = 3:length(files)
        file = files(j);
        if file.isdir
            continue
        end
        write_features(file);
    end
end
%%

% Test set

clear
close all
clc

dataset_path = 'C:\Users\18ae5\Desktop\Datasets';

fs = 8000;
test_path = '/test/audio/';

% loop over files



files = dir([dataset_path, '/', test_path]);
silent_files = {};

parfor j = 3:length(files)
    file = files(j);
    if file.isdir
        continue
    end
    %     write_features(file);
    %     disp([num2str(j-2), ' files processed'])
    try
        [x, file_fs] = audioread([file.folder, '/', file.name]);
    catch
        disp(['file name: ', file.name])
        disp(['fodler name: ', file.folder])
        continue
    end
    x = resample(x, fs, file_fs);
    if length(x) < 8000
        x(end+1:8000) = 0;
    end
    x = x(end-8000+1:end);
    [ activeNum, pos] = vad_YW(x, fs, 0);
    close all
    if activeNum == 0
        silent_files = [{file.name}, silent_files];
    end
   
end

