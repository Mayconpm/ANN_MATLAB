function [NET, OUTPUTS, ERRORS, RMSE, TR] = ANN(hiddenSizes,X,T,BP,ACT)
% ANN  Recebe 5 vari�veis de entrada, a quantidade de neur�nios e camadas
% da rede hiddenSizes, vari�veis de entrada X, alvos da rede neural T, a
% fun��o de ativa��o ACT e o algoritmo de backpropagation BP, e retorna a
% rede ap�s o treinamento,  a sa�da calculada com a rede neural OUTPUTS, o
% erro em rela��o a saida da rede neural e o alvo ERRORS, a raiz do erro
% m�dio quadr�tico RMSE e, por fim, o registro do treinamento TR.
% 
%   Vari�veis de entrada:
% 
%   hiddenSizes = Neur�nios a serem usados na rede neural, seja com uma
%   ou mais camadas ocultas.Ex: [1 3].
%       
%   X = Uma matriz R x Q.
%   T = Uma matriz U x Q.
%   Onde:
%       Q = N�mero de amostras
%       R = N�mero de elementos de entrada
%       U = Numero de elementos de sa�da
% 
%   BP = Algor�timo de backpropagation. (Pradr�o = 'trainlm')
%   http://bit.ly/MultilayerNeuralNetwork
% 
%   ACT = Fun��o de ativa��o da rede neural.(Pradr�o = 'tansig')
%   http://bit.ly/NeuralNetworkActivation
% 
%   SA�DAS:
%   NET = Retorna a rede neural treinada como um objeto _network_ 

%   OUTPUTS = Retorna os resultados obtido pela rede neural como uma matriz
%   U x Q. 

%   ERRORS = Erro entre os valores alvo e os resultados da rede neural como
%   uma matriz U x Q

%   RMSE = Retorna raiz do erro m�dio quadr�tico como valor
%   num�rico.

% 	TR = Retorna os dados do treinamento como _struct_ 

if nargin < 4
    BP = 'trainlm';
    ACT = 'tansig';
elseif nargin < 5
    ACT = 'tansig';
end

%% =========== ANN =============
%Inicializa a rede neural para regress�o
NET = fitnet(hiddenSizes, BP);

%Define a fun��o de ativa��o
for i = 1:length(hiddenSizes)
    NET.layers{i}.transferFcn = ACT;
end

%Para o caso da escolha da fun��o de ativa��o sigmoid s�o definidos os
%valores da normaliza��o dos dados para o intervalo de saida da fun��o
%sigmoid [0 , 1].
if isequal(ACT,'logsig')
    saida = length(hiddenSizes) + 1;
    NET.inputs{1}.processParams{2}.ymin = 0;
    NET.outputs{saida}.processParams{2}.ymin = 0;
end
% Os valores padr�o j� est�o dentro do intervalo para a fun��o tangente
% hiperb�lica [-1, 1] portanto n�o precosa ser mudado

%Treina a rede neural
[NET,TR] = train(NET,X,T,'useParallel','yes');

%Calculos dos valores obtidos pela rede neural
OUTPUTS = NET(X);
ERRORS = gsubtract(T,OUTPUTS);
RMSE = sqrt(mean((ERRORS).^2));

end