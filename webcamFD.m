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
  Jr = imresize(J,[112,92]);
  Jrgb = rgb2gray(Jr);
  imshow(Jrgb); 
end
%IFaces = insertObjectAnnotation(I, 'rectangle', bboxes, 'Face');
%figure, imshow(IFaces), title('Detected faces');
                                                                            
%Classifier test
[labelIdx, score] = predict(categoryClassifier,Jrgb);
categoryClassifier.Labels(labelIdx)

%queryFeatures = extractHOGFeatures(Jrgb);
%personLabel = predict(faceClassifier,queryFeatures);
%booleanIndex = strcmp(personLabel, personIndex);
%integerIndex = find(booleanIndex);
%subplot(1,2,1);imshow(Jrgb);title('Query Face');
%subplot(1,2,2);imshow(read(training(integerIndex),1));title('Matched Class');
