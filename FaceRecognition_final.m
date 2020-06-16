clear;

setDir=fullfile('imgDatabase');
imds = imageDatastore(setDir,'IncludeSubfolders',true,'LabelSource',...
    'foldernames');

%data preprocessing
testFile = imds.Files;
numFiles = length(testFile);

faceDetector = vision.CascadeObjectDetector;
faceDetector.MergeThreshold =7;

for k = 1 : numFiles
    fprintf('Now reading file %s\n', testFile{k});
    I = imread(testFile{k});
    [rows, columns, numberOfColorChannels] = size(I); %add this
    bboxes = step(faceDetector, I);
    for i = 1 : size(bboxes,1)     
      rectangle('Position', bboxes(i,:), 'LineWidth', 3, 'LineStyle', '-', 'EdgeColor', 'r'); 
    end 
    for i = 1 : size(bboxes, 1)
      if numberOfColorChannels == 3
          Jrgb = rgb2gray(I); %change to greyscale
          J = imcrop(Jrgb, bboxes(i, :));  %crop face from picture
          Jr = imresize(J,[112,92]); %resize image into 112 * 92

          imwrite(Jr,testFile{k}); %replace files with processed image
          imshow(Jr); 
      else %image is already grayscale
          J = imcrop(I, bboxes(i, :));  %crop face from picture
          Jr = imresize(J,[112,92]); %resize image into 112 * 92

          imwrite(Jr,testFile{k}); %replace files with processed image
          imshow(Jr); 
      end
    end
end
  
numTrainFiles =4;
[imdsTrain,imdsValidation] = splitEachLabel(imds,numTrainFiles,'randomize'); %split 

%define network architecture
inputSize = [112 92 1];
numClasses = 5;

layers = [
    imageInputLayer(inputSize)
    convolution2dLayer(5,20)
    batchNormalizationLayer
    reluLayer
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];

%train network
options = trainingOptions('sgdm', ...
    'MaxEpochs',4, ...
    'ValidationData',imdsValidation, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress');

facenet = trainNetwork(imdsTrain,layers,options);

%Test Network
YPred = classify(facenet,imdsValidation);
YValidation = imdsValidation.Labels;
accuracy = mean(YPred == YValidation);

%test with webcam
C = webcamlist;
cam=webcam(C{1});
preview(cam);
NotYet = false;
faceDetector = vision.CascadeObjectDetector;
faceDetector.MergeThreshold =7;
while ~NotYet
pause(2);
I = snapshot(cam);
disp('Took a snapshot. Checking to find a face ....')
bboxes = step(faceDetector, I);
if ~isempty(bboxes)
NotYet = true;
disp('Face found!');
break;
end
disp('No face detected :(, Repeating...');
end
closePreview(cam);
clear('cam');

for i = 1 : size(bboxes,1)     
  rectangle('Position', bboxes(i,:), 'LineWidth', 3, 'LineStyle', '-', 'EdgeColor', 'r'); 
end 
for i = 1 : size(bboxes, 1) 
  J = imcrop(I, bboxes(i, :)); 
  Jr = imresize(I,[112,92]);
  Jrgb = rgb2gray(Jr);
  imshow(Jrgb); 
end

    label = classify(facenet,Jrgb);    % Classify the picture
    title(char(label));          % Show the class label
    drawnow