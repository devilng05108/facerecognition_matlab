%test data
I = imread('testData-ngchunwu.pgm'); %testData-louis / testData-melvin / testData-ngchunwu / testData-yann
faceDetector = vision.CascadeObjectDetector;
faceDetector.MergeThreshold =7;
bboxes = step(faceDetector, I);
    for i = 1 : size(bboxes,1)     
      rectangle('Position', bboxes(i,:), 'LineWidth', 3, 'LineStyle', '-', 'EdgeColor', 'r'); 
    end 
    for i = 1 : size(bboxes, 1) 
      J = imcrop(I, bboxes(i, :)); 
      Jr = imresize(J,[112,92]);
      %Jrgb = rgb2gray(Jr);
      imshow(Jr); 
    end

[labelIdx, score] = predict(categoryClassifier,J);
test = categoryClassifier.Labels(labelIdx);
disp(test);

%queryFeatures = extractHOGFeatures(img);
%personLabel = predict(faceClassifier,queryFeatures);
%booleanIndex = strcmp(personLabel, personIndex);
%integerIndex = find(booleanIndex);
%subplot(1,2,1);imshow(img);title('Query Face');
%subplot(1,2,2);imshow(read(training(integerIndex),1));title('Matched Class');