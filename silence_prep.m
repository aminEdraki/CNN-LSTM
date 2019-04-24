clear
close all
clc

path = 'C:\Users\18ae5\Desktop\Datasets\ASR\Gabor\train\audio\silence\';
file_names = {'doing_the_dishes', 'dude_miaowing', 'exercise_bike', 'pink_noise', 'running_tap', 'white_noise'};
for i = 1:length(file_names)
    fname = [path, file_names{i}, '.wav'];
    [y, fs] = audioread(fname);
    y = resample(y, 8000, fs);
    for j = 1:8000/4:length(y)-8000
        x = y(j:j+8000-1);
        x = sgbfb_feature_extraction(x, 8000);
%         x = mfcc_feature_extraction(x, 8000);
%         x = log_mel_spectrogram(x, 8000);
        x = x - mean(x, 2);
        mag = x.*x;
        x = x ./ sum(mag, 2);
        
        x = x - mean(x, 1);
        mag = x.*x;
        x = x ./ sum(mag, 1);
        
        x = (x - min(x(:)))./(max(x(:)) - min(x(:)));
        
        
        if size(x, 2) < 98
            x(:, end+1:98) = x(:, end);
        end
        assert(size(x, 2) == 98)
        
        save([path, file_names{i}, '_', num2str(j), '.mat'], 'x')
    end
end
