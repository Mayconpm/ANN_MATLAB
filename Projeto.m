% =========== Definição dos dados ============= 
Neurons = 8;

Data = load('pulses_uni0-1024_and_noise_zbEBAch01mu40.txt');
X = Data(:,1:7)';
T = Data(:,8)';

%% =========== OF2 =============
OF2 = [-0.3781 -0.3572 .1808 .8125 .2767 -0.2056 -0.3292];

X_OF2 = (OF2' .* X);
outputsOF2 = (OF2 * X);

errorsOF2 = gsubtract(T,outputsOF2);
RMSEOF2 = sqrt(mean((errorsOF2).^2));

% =========== Redes Neurais =============

[net,outputs,errors] = ANN(Neurons,X,T);

[netOF2,outputsOF2,errorsOF2] = ANN(Neurons,X_OF2,T);

% =========== Gráficos =============
[~, name, ~] = fileparts(file);
folder = fullfile('Histogramas', name);

hold on; 
h1= histogram(errors,-100:10:100);
h1.NumBins = 100; 
h2= histogram(errorsOF2,-100:10:100);
h2.NumBins = 100; 
legend('Levenberg–Marquardt', 'OF2'); hold off;
print(fullfile(folder, 'Levenberg–Marquardt e OF2'),'-dpng','-r600');
