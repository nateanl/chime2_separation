addpath('../../../../mimlib');
addpath('../../../../utils');
basedir = '/Users/Near/Desktop/test/';
predictDir ='/Users/Near/Desktop/pred/';
testsavedir = '/Users/Near/Desktop/save/test';
predsavedir = '/Users/Near/Desktop/save/pred';
type = '0dB';
testDir = fullfile(basedir,type);
predictDir = fullfile(predictDir,type);
testsavedir = fullfile(testsavedir,type);
predsavedir = fullfile(predsavedir,type);
testlist = findFiles(testDir,'.mat');
%testlist = testlist(runWithRandomSeed(22, @randperm, length(testlist)));
data = [];
sdrr =0.0;
for f = 1:length(testlist)
    testFile = fullfile(testDir,testlist{f});
    predFile = fullfile(predictDir,testlist{f});
    load(testFile);
    load(predFile);
    spect = data.spect;
    x = convert_save(spect,nsampl);
    y = convert_save(clean.',nsampl);
    testfile = fullfile(testsavedir,strrep(testlist{f}, '.mat', '.wav'));
    predfile = fullfile(predsavedir,strrep(testlist{f}, '.mat', '.wav'));
    ensureDirExists(testfile);
    ensureDirExists(predfile);
    audiowrite(testfile,x,fs);
    audiowrite(predfile,y,fs);
    options.destDir = '/Users/Near/Desktop/save/peass/';
    options.segmentationFactor = 1; % increase this integer if you experienced "out of memory" problems
    res = PEASS_ObjectiveMeasure({testfile},predfile,options);
    fprintf(' - SDR = %.1f dB\n - ISR = %.1f dB\n - SIR = %.1f dB\n - SAR = %.1f dB\n',...
    res.SDR,res.ISR,res.SIR,res.SAR);
    sdrr = sdrr+ res.SDR;
    %fprintf('write audio file: %s\n',testfile);
    %fprintf('write audio file: %s\n',predfile);
end
fprintf('the average sdr is %f',sdrr);