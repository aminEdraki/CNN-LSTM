function write_features(file)
fs = 8000;
try
    [x, file_fs] = audioread([file.folder, '/', file.name]);
catch
    disp(['file name: ', file.name])
    disp(['fodler name: ', file.folder])
    return
end

[~, name, ~] = fileparts(file.name);
x = resample(x, fs, file_fs);
if length(x) < 8000
    x(end+1:8000) = 0;
end
x = x(1:8000);
% sgbfb_feature_extraction(x, fs);

% x = sgbfb_feature_extraction(x, fs);
% x = mfcc_feature_extraction(x, fs);
x = log_mel_spectrogram(x, fs);
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

save([file.folder, '/', name, '.mat'], 'x')
end