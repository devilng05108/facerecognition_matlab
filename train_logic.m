% Train for classifier 
setDir=fullfile('FaceDatabase');
imds = imageDatastore(setDir,'IncludeSubfolders',true,'LabelSource',...
    'foldernames');
testFile = imds.Files
numFiles = length(testFile);

faceDetector = vision.CascadeObjectDetector;
faceDetector.MergeThreshold =7;

%data preprocessing
for k = 1 : numFiles
    fprintf('Now reading file %s\n', testFile{k});
    I = imread(testFile{k});
    bboxes = step(faceDetector, I);
    for i = 1 : size(bboxes,1)     
      rectangle('Position', bboxes(i,:), 'LineWidth', 3, 'LineStyle', '-', 'EdgeColor', 'r'); 
    end 
    for i = 1 : size(bboxes, 1) 
      J = imcrop(I, bboxes(i, :)); 
      Jr = imresize(J,[112,92]);
      Jrgb = rgb2gray(Jrgb);
      imwrite(Jrgb,testFile{k});
      imshow(Jrgb); 
    end
end

%training
[trainingSet,testSet] = splitEachLabel(imds,0.7,'randomize'); %higher training = higher accuracy
bag = bagOfFeatures(trainingSet);
categoryClassifier = trainImageCategoryClassifier(trainingSet,bag);
confMatrix = evaluate(categoryClassifier,testSet)
mean(diag(confMatrix))

% Train image - requires preprocess
%faceDatabase = imageSet('FaceDatabase','recursive');
%[training,test] = partition(faceDatabase,[0.8 0.2]);
%trainingFeatures = zeros(size(training,2)*training(1).Count,4680);
%featureCount = 1;
%for i=1:size(training,2)
%    for j = 1:training(i).Count
%        trainingFeatures(featureCount,:) = extractHOGFeatures(read(training(i),j));
%        trainingLabel{featureCount} = training(i).Description;    
%        featureCount = featureCount + 1;
%    end
%    personIndex{i} = training(i).Description;
%end
%faceClassifier = fitcecoc(trainingFeatures,trainingLabel);